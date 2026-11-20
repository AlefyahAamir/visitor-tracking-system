from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import json

import pandas as pd

from regions import point_in_polygon


@dataclass
class CustomerRecord:
    visitor_id: str
    first_seen: str
    last_seen: str
    counter_first_seen: Optional[str] = None
    counter_last_seen: Optional[str] = None
    counter_visits: int = 0
    provisional_counter_only: bool = False
    entered_later: bool = False
    gate_name: str = ""
    entry_time: Optional[str] = None


class JourneyManager:
    """Full-session counter-to-gate journey state.

    A counter visit is confirmed only after several consecutive frames inside
    the user-drawn counter polygon.  Once that confirmed visitor leaves or
    disappears, the counter-only total increases immediately.  A confirmed
    gate crossing by the same persistent visitor ID later in the same run
    reverses that provisional result, no matter how many hours later it occurs.
    """

    def __init__(
        self,
        counter_polygon,
        missing_grace_frames: int = 25,
        counter_confirm_frames: int = 5,
    ):
        self.counter_polygon = list(counter_polygon)
        self.missing_grace_frames = max(1, int(missing_grace_frames))
        self.counter_confirm_frames = max(1, int(counter_confirm_frames))
        self.records: Dict[str, CustomerRecord] = {}
        self.last_inside: Dict[str, bool] = {}
        self.inside_streak: Dict[str, int] = {}
        self.confirmed_current_visit: Dict[str, bool] = {}
        self.missing_frames: Dict[str, int] = {}
        self.processed_entry_keys: Set[str] = set()
        self.audit: List[dict] = []

    @property
    def provisional_counter_only(self) -> int:
        return sum(1 for r in self.records.values() if r.provisional_counter_only)

    @property
    def later_entered(self) -> int:
        return sum(1 for r in self.records.values() if r.entered_later)

    @property
    def total_counter_visitors(self) -> int:
        return sum(1 for r in self.records.values() if r.counter_visits > 0)

    def _record(self, visitor_id: str, now: datetime) -> CustomerRecord:
        iso = now.isoformat(timespec="seconds")
        if visitor_id not in self.records:
            self.records[visitor_id] = CustomerRecord(visitor_id, iso, iso)
        self.records[visitor_id].last_seen = iso
        return self.records[visitor_id]

    def _confirm_counter_visit(self, visitor_id: str, rec: CustomerRecord, now: datetime) -> None:
        if self.confirmed_current_visit.get(visitor_id, False):
            return
        self.confirmed_current_visit[visitor_id] = True
        rec.counter_visits += 1
        iso = now.isoformat(timespec="seconds")
        rec.counter_first_seen = rec.counter_first_seen or iso
        rec.counter_last_seen = iso
        self.audit.append({"time": now, "visitor_id": visitor_id, "event": "counter_visit_confirmed", "detail": ""})

    def _mark_counter_only(self, visitor_id: str, now: datetime, reason: str) -> None:
        rec = self.records.get(visitor_id)
        if not rec or rec.entered_later or not self.confirmed_current_visit.get(visitor_id, False):
            return
        if not rec.provisional_counter_only:
            rec.provisional_counter_only = True
            self.audit.append({"time": now, "visitor_id": visitor_id, "event": reason, "detail": ""})
        self.confirmed_current_visit[visitor_id] = False
        self.inside_streak[visitor_id] = 0

    def update_tracks(self, tracks: Iterable, frame_index: int, now: datetime) -> None:
        active: Set[str] = set()

        for track in tracks:
            visitor_id = getattr(track, "visitor_id", None)
            if not visitor_id:
                continue

            active.add(visitor_id)
            rec = self._record(visitor_id, now)
            foot = (
                track.history[-1]
                if getattr(track, "history", None)
                else ((track.bbox[0] + track.bbox[2]) // 2, track.bbox[3])
            )
            inside = point_in_polygon(foot, self.counter_polygon)
            was_inside = self.last_inside.get(visitor_id, False)

            if inside:
                self.inside_streak[visitor_id] = self.inside_streak.get(visitor_id, 0) + 1
                if self.inside_streak[visitor_id] >= self.counter_confirm_frames:
                    self._confirm_counter_visit(visitor_id, rec, now)
                    rec.counter_last_seen = now.isoformat(timespec="seconds")
            else:
                self.inside_streak[visitor_id] = 0
                if was_inside:
                    self._mark_counter_only(visitor_id, now, "counter_left_provisional")

            self.last_inside[visitor_id] = inside
            self.missing_frames[visitor_id] = 0

        for visitor_id, was_inside in list(self.last_inside.items()):
            if visitor_id in active:
                continue
            self.missing_frames[visitor_id] = self.missing_frames.get(visitor_id, 0) + 1
            if was_inside and self.missing_frames[visitor_id] >= self.missing_grace_frames:
                self._mark_counter_only(visitor_id, now, "counter_disappeared_provisional")
                self.last_inside[visitor_id] = False

    def consume_entries(self, entry_events: Iterable) -> None:
        for event in entry_events:
            key = f"{event.visitor_id}|{event.gate_name}|{event.timestamp.isoformat()}|{event.track_id}"
            if key in self.processed_entry_keys:
                continue
            self.processed_entry_keys.add(key)

            rec = self.records.get(event.visitor_id)
            if rec and rec.counter_visits > 0:
                rec.provisional_counter_only = False
                rec.entered_later = True
                rec.gate_name = event.gate_name
                rec.entry_time = event.timestamp.isoformat(timespec="seconds")
                self.audit.append({
                    "time": event.timestamp,
                    "visitor_id": event.visitor_id,
                    "event": "later_entered",
                    "detail": event.gate_name,
                })

    def records_dataframe(self) -> pd.DataFrame:
        columns = [field.name for field in CustomerRecord.__dataclass_fields__.values()]
        rows = [asdict(r) for r in self.records.values() if r.counter_visits > 0]
        return pd.DataFrame(rows, columns=columns)

    def audit_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.audit, columns=["time", "visitor_id", "event", "detail"])

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.records_dataframe().to_csv(output_dir / "customer_journeys.csv", index=False)
        self.audit_dataframe().to_csv(output_dir / "journey_audit.csv", index=False)
        (output_dir / "customer_journeys.json").write_text(
            json.dumps([asdict(r) for r in self.records.values()], indent=2),
            encoding="utf-8",
        )
