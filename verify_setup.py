"""Fast package and deterministic entry-logic checks used by run_app.bat."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import cv2  # noqa: F401
import numpy as np
import pandas  # noqa: F401
import streamlit  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import ultralytics  # noqa: F401

from osnet_model import build_osnet_x0_25
from journey import JourneyManager
from vision import EntryEvent, Gate, TrackInfo, VideoProcessor


def _bbox_at(x: int, foot_y: int) -> tuple[int, int, int, int]:
    return (x - 20, foot_y - 100, x + 20, foot_y)


def _make_gate(name: str = "Gate 1") -> Gate:
    return Gate(
        name=name,
        corridor_polygon=[(200, 50), (440, 50), (440, 330), (200, 330)],
        count_line=((200, 160), (440, 160)),
        confirmation_line=((200, 220), (440, 220)),
        entry_from_bottom=False,
        outside_confirm_frames=3,
    )


def _simulate(
    paths: dict[int, list[tuple[int, int]]],
    first_seen_frames: dict[int, int],
    gate_name: str = "Gate 1",
) -> int:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    processor = VideoProcessor()
    processor.configure_gates([_make_gate(gate_name)])
    processor.frame_width = 640
    processor.frame_height = 360

    tracks = {
        track_id: TrackInfo(
            track_id=track_id,
            bbox=_bbox_at(*points[0]),
            confidence=0.90,
            first_seen_frame=first_seen_frames[track_id],
        )
        for track_id, points in paths.items()
    }

    maximum_length = max(len(points) for points in paths.values())
    for index in range(maximum_length):
        processor.frame_index = 40 + index
        active_tracks: list[TrackInfo] = []

        for track_id, points in paths.items():
            if index >= len(points):
                continue
            track = tracks[track_id]
            track.bbox = _bbox_at(*points[index])
            track.history.append(points[index])
            active_tracks.append(track)

        processor.active_track_ids = {
            track.track_id for track in active_tracks
        }
        processor.update_gate_events(frame, active_tracks)

    return processor.total_entries



def _simulate_occluded_tandem() -> int:
    """Second person is seen outside, occluded, then receives a new ID."""

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    processor = VideoProcessor()
    processor.configure_gates([_make_gate("Gate 1")])
    processor.frame_width = 640
    processor.frame_height = 360

    first = TrackInfo(
        track_id=1,
        bbox=_bbox_at(280, 100),
        confidence=0.90,
        first_seen_frame=40,
    )
    second = TrackInfo(
        track_id=2,
        bbox=_bbox_at(360, 100),
        confidence=0.90,
        first_seen_frame=40,
    )
    processor.tracks = {1: first, 2: second}

    for frame_index, foot_y in enumerate([100, 115, 130], start=40):
        processor.frame_index = frame_index
        processor.active_track_ids = {1, 2}
        first.bbox = _bbox_at(280, foot_y)
        second.bbox = _bbox_at(360, foot_y)
        first.history.append((280, foot_y))
        second.history.append((360, foot_y))
        processor.update_gate_events(frame, [first, second])
        processor.cleanup_tracks()

    # The second person is briefly hidden behind the first just before yellow.
    processor.frame_index = 43
    processor.active_track_ids = {1}
    first.bbox = _bbox_at(280, 175)
    first.history.append((280, 175))
    processor.update_gate_events(frame, [first])
    processor.cleanup_tracks()

    # The same second person becomes visible again with a new tracker ID.
    second_reacquired = TrackInfo(
        track_id=3,
        bbox=_bbox_at(360, 180),
        confidence=0.90,
        first_seen_frame=44,
    )
    processor.tracks[3] = second_reacquired

    first_path = [205, 230, 245, 260]
    second_path = [180, 205, 230, 245]
    for offset, (first_y, second_y) in enumerate(
        zip(first_path, second_path),
        start=44,
    ):
        processor.frame_index = offset
        processor.active_track_ids = {1, 3}
        first.bbox = _bbox_at(280, first_y)
        second_reacquired.bbox = _bbox_at(360, second_y)
        first.history.append((280, first_y))
        second_reacquired.history.append((360, second_y))
        processor.update_gate_events(frame, [first, second_reacquired])
        processor.cleanup_tracks()

    return processor.total_entries

def _verify_entry_logic() -> None:
    normal_path = [
        (300, 100),
        (300, 115),
        (300, 130),
        (300, 175),
        (300, 205),
        (300, 230),
        (300, 245),
    ]

    cases = {
        "normal confirmed entry": (
            {1: normal_path},
            {1: 40},
            "Gate 1",
            1,
        ),
        "two separately tracked tandem entries": (
            {
                1: [(280, y) for _x, y in normal_path],
                2: [(360, y) for _x, y in normal_path],
            },
            {1: 40, 2: 40},
            "Gate 2",
            2,
        ),
        "late track first seen after yellow": (
            {3: [(300, 175), (300, 200), (300, 230), (300, 250)]},
            {3: 40},
            "Gate 1",
            0,
        ),
        "standing near yellow": (
            {
                4: [
                    (300, 100),
                    (300, 115),
                    (300, 130),
                    (300, 155),
                    (302, 164),
                    (299, 158),
                    (301, 165),
                ]
            },
            {4: 40},
            "Gate 1",
            0,
        ),
        "cross yellow then turn back": (
            {
                5: [
                    (300, 100),
                    (300, 115),
                    (300, 130),
                    (300, 180),
                    (300, 200),
                    (300, 170),
                    (300, 140),
                    (300, 120),
                ]
            },
            {5: 40},
            "Gate 2",
            0,
        ),
        "sideways movement": (
            {
                6: [
                    (250, 100),
                    (270, 115),
                    (290, 130),
                    (340, 175),
                    (400, 190),
                    (430, 200),
                ]
            },
            {6: 40},
            "Gate 2",
            0,
        ),
        "Gate 3 outward exit": (
            {
                7: [
                    (330, 260),
                    (330, 240),
                    (330, 215),
                    (330, 190),
                    (330, 165),
                    (330, 140),
                    (330, 120),
                    (330, 100),
                ]
            },
            {7: 40},
            "Gate 3",
            0,
        ),
        "appearing deep inside": (
            {8: [(300, 240), (300, 260), (300, 285), (300, 310)]},
            {8: 40},
            "Gate 3",
            0,
        ),
    }

    for name, (paths, first_seen_frames, gate_name, expected) in cases.items():
        actual = _simulate(paths, first_seen_frames, gate_name)
        if actual != expected:
            raise RuntimeError(
                f"Entry-logic check failed for {name}: "
                f"expected {expected}, got {actual}."
            )

    occluded_tandem_count = _simulate_occluded_tandem()
    if occluded_tandem_count != 2:
        raise RuntimeError(
            "Entry-logic check failed for occluded tandem handoff: "
            f"expected 2, got {occluded_tandem_count}."
        )


def _verify_detector_lock() -> None:
    class DummyDetector:
        def __init__(self) -> None:
            self.active_calls = 0
            self.maximum_active_calls = 0
            self.guard = threading.Lock()

        def track(self, *_args, **_kwargs):
            with self.guard:
                self.active_calls += 1
                self.maximum_active_calls = max(
                    self.maximum_active_calls,
                    self.active_calls,
                )
            time.sleep(0.03)
            with self.guard:
                self.active_calls -= 1
            return []

    processor = VideoProcessor()
    detector = DummyDetector()
    processor.detector = detector
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    workers = [
        threading.Thread(target=processor.track_people, args=(frame,))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    if detector.maximum_active_calls != 1:
        raise RuntimeError(
            "YOLO detector lock check failed: concurrent model calls were allowed."
        )



def _verify_user_drawn_line_direction() -> None:
    """Reversing yellow line endpoints must not reverse entry direction."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    processor = VideoProcessor()
    gate = _make_gate("Gate 3")
    gate.count_line = (gate.count_line[1], gate.count_line[0])
    gate.confirmation_line = (gate.confirmation_line[1], gate.confirmation_line[0])
    processor.configure_gates([gate])
    processor.frame_width = 640
    processor.frame_height = 360
    track = TrackInfo(track_id=99, bbox=_bbox_at(320, 100), confidence=0.9, first_seen_frame=1)
    for i, y in enumerate([100, 115, 130, 175, 205, 230, 250], start=1):
        processor.frame_index = i
        track.bbox = _bbox_at(320, y)
        track.history.append((320, y))
        processor.active_track_ids = {99}
        processor.update_gate_events(frame, [track])
    if processor.total_entries != 1:
        raise RuntimeError("User-drawn line direction check failed.")


def _verify_journey_reversal() -> None:
    polygon = [(100, 100), (300, 100), (300, 260), (100, 260)]
    journey = JourneyManager(polygon, missing_grace_frames=2, counter_confirm_frames=2)
    now = datetime(2026, 7, 30, 9, 0, 0)
    track = TrackInfo(track_id=1, bbox=_bbox_at(200, 180), confidence=0.9, first_seen_frame=1)
    track.visitor_id = "V-TEST"
    for frame_index in (1, 2):
        track.history.append((200, 180))
        journey.update_tracks([track], frame_index, now + timedelta(seconds=frame_index))
    track.bbox = _bbox_at(400, 180)
    track.history.append((400, 180))
    journey.update_tracks([track], 3, now + timedelta(seconds=3))
    if journey.provisional_counter_only != 1:
        raise RuntimeError("Counter-only immediate update check failed.")
    event = EntryEvent(now + timedelta(hours=5), "V-TEST", 77, "Gate 2", "entry", 0.9, False)
    journey.consume_entries([event])
    if journey.provisional_counter_only != 0 or journey.later_entered != 1:
        raise RuntimeError("Five-hour later-entry reversal check failed.")


def _verify_counter_identity_counts_first_gate_once() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    processor = VideoProcessor()
    gate = _make_gate("Gate 1")
    processor.configure_gates([gate])
    processor.frame_width = 640
    processor.frame_height = 360
    processor.frame_index = 100
    first_track = TrackInfo(track_id=10, bbox=_bbox_at(300, 250), confidence=0.9, first_seen_frame=1)
    visitor_id = processor.ensure_visitor_identity(first_track, "counter")
    processor._record_entry(frame, first_track, gate)
    if processor.total_entries != 1 or not processor.events[-1].counted_unique:
        raise RuntimeError("A counter-known person's first gate entry was not counted.")
    second_track = TrackInfo(track_id=11, bbox=_bbox_at(300, 250), confidence=0.9, first_seen_frame=200)
    second_track.visitor_id = visitor_id
    processor.frame_index = 1000
    processor._record_entry(frame, second_track, gate)
    if processor.total_entries != 1 or processor.events[-1].counted_unique:
        raise RuntimeError("A returning person's later gate entry was counted twice.")

def main() -> None:
    model = build_osnet_x0_25()
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count <= 0:
        raise RuntimeError("The bundled OSNet model could not be built.")

    _verify_entry_logic()
    _verify_user_drawn_line_direction()
    _verify_journey_reversal()
    _verify_counter_identity_counts_first_gate_once()
    _verify_detector_lock()
    print(
        "Python packages, OSNet, user-drawn line direction, tandem two-line "
        "entry logic, journey reversal, and detector locking are ready."
    )


if __name__ == "__main__":
    main()
