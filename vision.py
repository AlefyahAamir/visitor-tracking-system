"""
VisitorTrackingSystem computer-vision engine.

Features
--------
- YOLO person detection
- Persistent BoT-SORT multi-object tracking
- OSNet person re-identification
- Three narrow, independently calibrated gate corridors
- Entry-only directional line crossing using each person's feet point
- Standing, hesitation, sideways movement, and reverse movement rejection
- Two-stage outer-line and inner-confirmation-line entry validation
- Annotated frame rendering and CSV event export
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from osnet_model import build_osnet_x0_25, load_pretrained_weights


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
CROP_DIR = OUTPUT_DIR / "crops"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"

for directory in (MODEL_DIR, OUTPUT_DIR, CROP_DIR, SNAPSHOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_PATH = MODEL_DIR / "yolo11s.pt"
REID_MODEL_PATH = MODEL_DIR / "osnet_x0_25_msmt17.pth"
EVENT_CSV_PATH = OUTPUT_DIR / "events.csv"


# ============================================================
# SETTINGS
# ============================================================

PERSON_CLASS_ID = 0
# Crowded entrance settings. A lower confidence threshold plus a larger
# inference image makes it less likely that a child or a second overlapping
# adult is swallowed by the first person's detection box.
CONFIDENCE_THRESHOLD = 0.28
IOU_THRESHOLD = 0.70
INFERENCE_IMAGE_SIZE = 960
MIN_PERSON_HEIGHT = 34
MAX_TRACK_AGE = 75
MAX_HISTORY_POINTS = 60

# Re-identification must be conservative. In particular, two people who are
# present at the same time must never be merged just because their clothing
# embeddings look similar.
REID_SIMILARITY_THRESHOLD = 0.88
REID_ID_SWITCH_SIMILARITY_THRESHOLD = 0.80
REID_UPDATE_INTERVAL = 8
REID_MIN_CROP_WIDTH = 18
REID_MIN_CROP_HEIGHT = 40
REID_QUALITY_IMPROVEMENT = 1.03
REID_ID_SWITCH_MAX_GAP_FRAMES = 90
REID_ID_SWITCH_MAX_DISTANCE_RATIO = 0.18
REID_RECENT_DISTINCT_WINDOW_FRAMES = 150

EVENT_COOLDOWN_FRAMES = 45
OUTSIDE_CONFIRM_FRAMES = 3
INSIDE_CONFIRM_FRAMES = 2
CORRIDOR_GRACE_FRAMES = 10
LINE_DEADBAND_RATIO = 0.010
MIN_INWARD_TRAVEL_RATIO = 0.030
MAX_LATERAL_TO_INWARD_RATIO = 3.25

# No person may be inferred from nothing. A count must have a real track
# confirmed outside. The tandem handoff below only reconnects that already-
# observed outside track when short occlusion creates a new tracker ID.
TANDEM_HANDOFF_MAX_GAP_FRAMES = 18
TANDEM_HANDOFF_NEW_TRACK_MAX_AGE_FRAMES = 12
TANDEM_HANDOFF_OUTSIDE_BAND_RATIO = 0.20
TANDEM_HANDOFF_MAX_DISTANCE_RATIO = 0.18
TANDEM_HANDOFF_MIN_APPROACH_RATIO = 0.018
TANDEM_HANDOFF_MIN_EMBEDDING_SIMILARITY = 0.58
SAVE_CROPS = False
SAVE_SNAPSHOTS = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 220, 80)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (70, 70, 70)

ZONE_OUTSIDE = "outside"
ZONE_CORRIDOR = "corridor"
ZONE_INSIDE = "inside"
ZONE_NONE = "none"


# Backward-compatible state constants.
STATE_OUTSIDE = 0
STATE_APPROACHING = 1
STATE_CORRIDOR = 2
STATE_CROSSED = 3
STATE_INSIDE = 4


# ============================================================
# DATA MODELS
# ============================================================

Point = tuple[int, int]
BoundingBox = tuple[int, int, int, int]
Polygon = list[Point]


@dataclass
class GatePassState:
    """Per-track state for one confirmed outside-to-inside gate passage."""

    phase: str = "unarmed"
    outside_frames: int = 0
    confirmation_frames: int = 0
    last_in_corridor_frame: int = -1
    start_point: Optional[Point] = None
    start_progress: Optional[float] = None
    last_progress: Optional[float] = None
    counted: bool = False
    count_method: str = ""
    recent_progress: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_points: deque[Point] = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class PendingTandemCandidate:
    """A real outside track briefly lost while approaching a gate line."""

    gate_name: str
    source_track_id: int
    created_frame: int
    expires_frame: int
    start_point: Point
    start_progress: float
    last_point: Point
    last_progress: float
    recent_points: tuple[Point, ...]
    embedding: Optional[np.ndarray] = None


@dataclass
class TrackInfo:
    """Runtime information for one BoT-SORT track."""

    track_id: int
    bbox: BoundingBox
    confidence: float

    history: deque[Point] = field(
        default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS)
    )

    first_seen: float = field(default_factory=time.time)
    first_seen_frame: int = 0
    last_seen: float = field(default_factory=time.time)
    lost_frames: int = 0

    visitor_id: Optional[str] = None
    best_crop: Optional[np.ndarray] = None
    best_embedding: Optional[np.ndarray] = None
    best_quality: float = 0.0
    last_reid_frame: int = -REID_UPDATE_INTERVAL

    counted: bool = False
    last_gate: str = ""
    gate_last_event_frame: dict[str, int] = field(default_factory=dict)
    gate_pass_state: dict[str, GatePassState] = field(default_factory=dict)


@dataclass
class VisitorIdentity:
    """Appearance information for one unique visitor."""

    visitor_id: str
    embedding: Optional[np.ndarray]
    best_crop: Optional[np.ndarray]
    first_seen: datetime
    last_seen: datetime
    times_seen: int = 1
    last_gate: str = ""
    last_track_id: Optional[int] = None
    last_seen_frame: int = -1
    last_foot: Optional[Point] = None
    has_entered: bool = False


@dataclass
class EntryEvent:
    """One confirmed physical gate crossing.

    ``counted_unique`` is False when a person already known during the same
    analysis crosses a gate later.  The crossing is still emitted so the
    counter-to-gate journey can be corrected, but the unique gate total is not
    incremented again.
    """

    timestamp: datetime
    visitor_id: str
    track_id: int
    gate_name: str
    direction: str
    confidence: float
    counted_unique: bool = True


@dataclass
class Detection:
    """Backward-compatible person-detection result."""

    bbox: BoundingBox
    confidence: float


@dataclass
class Gate:
    """One physical gate corridor with outer and inner directional lines."""

    name: str
    corridor_polygon: Polygon
    count_line: tuple[Point, Point]
    confirmation_line: Optional[tuple[Point, Point]] = None
    entry_from_bottom: bool = True
    entry_count: int = 0
    outside_confirm_frames: int = OUTSIDE_CONFIRM_FRAMES


# ============================================================
# GATE HELPERS
# ============================================================


def create_default_gates(
    frame_width: int,
    frame_height: int,
    entry_from_bottom: bool = True,
    gate_ranges: Optional[Sequence[tuple[float, float]]] = None,
    corridor_y_range: tuple[float, float] = (0.20, 0.82),
    count_line_y: float = 0.51,
    confirmation_gap: float = 0.08,
) -> list[Gate]:
    """Create three narrow gate corridors with two directional boundaries.

    The yellow line is the outer crossing line. The green line is farther
    inward. A person is counted only after the same track is visibly observed
    outside, crosses the yellow line, then reaches and remains beyond the green
    line while continuing inward. No late-track or inferred entry is allowed.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")

    if gate_ranges is None:
        gate_ranges = ((0.41, 0.57), (0.56, 0.73), (0.75, 0.94))
    if len(gate_ranges) != 3:
        raise ValueError("Exactly three gate ranges are required.")

    y1_ratio, y2_ratio = corridor_y_range
    y1_ratio = max(0.0, min(float(y1_ratio), 0.95))
    y2_ratio = max(y1_ratio + 0.08, min(float(y2_ratio), 1.0))

    line_ratio = float(count_line_y)
    gap_ratio = max(0.03, min(float(confirmation_gap), 0.20))

    if entry_from_bottom:
        # Outside is below; inward movement is upward.
        line_ratio = max(y1_ratio + gap_ratio + 0.03, min(line_ratio, y2_ratio - 0.03))
        confirmation_ratio = line_ratio - gap_ratio
        confirmation_ratio = max(y1_ratio + 0.02, confirmation_ratio)
    else:
        # Outside is above; inward movement is downward (reference camera layout).
        line_ratio = max(y1_ratio + 0.03, min(line_ratio, y2_ratio - gap_ratio - 0.03))
        confirmation_ratio = line_ratio + gap_ratio
        confirmation_ratio = min(y2_ratio - 0.02, confirmation_ratio)

    y1 = int(round(y1_ratio * (frame_height - 1)))
    y2 = int(round(y2_ratio * (frame_height - 1)))
    line_y = int(round(line_ratio * (frame_height - 1)))
    confirmation_y = int(round(confirmation_ratio * (frame_height - 1)))

    gates: list[Gate] = []
    for index, (left_ratio, right_ratio) in enumerate(gate_ranges, start=1):
        left_ratio = max(0.0, min(float(left_ratio), 0.98))
        right_ratio = max(left_ratio + 0.02, min(float(right_ratio), 1.0))
        x1 = int(round(left_ratio * (frame_width - 1)))
        x2 = int(round(right_ratio * (frame_width - 1)))
        corridor = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        gates.append(
            Gate(
                name=f"Gate {index}",
                corridor_polygon=corridor,
                count_line=((x1, line_y), (x2, line_y)),
                confirmation_line=((x1, confirmation_y), (x2, confirmation_y)),
                entry_from_bottom=entry_from_bottom,
                outside_confirm_frames=OUTSIDE_CONFIRM_FRAMES,
            )
        )

    return gates


# ============================================================
# VIDEO PROCESSOR
# ============================================================


class VideoProcessor:
    """Main computer-vision engine."""

    def __init__(self) -> None:
        self.detector: Optional[YOLO] = None
        self.reid_model: Optional[torch.nn.Module] = None
        self.tracking_config = "botsort.yaml"
        self.model_load_error: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tracks: dict[int, TrackInfo] = {}
        self.visitors: dict[str, VisitorIdentity] = {}
        self.events: list[EntryEvent] = []
        self.gates: list[Gate] = []
        self.pending_tandem_candidates: dict[str, list[PendingTandemCandidate]] = {}

        self.total_entries = 0

        self.frame_index = 0
        self.active_track_ids: set[int] = set()
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.current_timestamp: Optional[datetime] = None

        # Ultralytics mutates the model during its first predictor setup by
        # fusing Conv and BatchNorm layers. Protect every detector call so one
        # processor can never be fused or used concurrently by two callbacks.
        self._detector_lock = threading.RLock()

    # --------------------------------------------------------
    # State and model management
    # --------------------------------------------------------

    @property
    def total_unique_visitors(self) -> int:
        return len(self.visitors)

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    @property
    def models_device(self) -> str:
        return self.device

    @property
    def is_ready(self) -> bool:
        return self.is_models_loaded() and bool(self.gates)

    @property
    def model_status(self) -> str:
        if self.is_models_loaded():
            return f"YOLO and OSNet are loaded on {self.device}."
        if self.model_load_error:
            return self.model_load_error
        return "Models have not been loaded."

    def is_models_loaded(self) -> bool:
        return self.detector is not None and self.reid_model is not None

    def configure_gates(self, gates: Sequence[Gate]) -> None:
        self.gates = list(gates)

    def add_gate(self, gate: Gate) -> None:
        self.gates.append(gate)

    def clear_gates(self) -> None:
        self.gates.clear()

    def reset(self) -> None:
        """Reset counts and runtime state while keeping loaded models."""

        self.tracks.clear()
        self.visitors.clear()
        self.events.clear()
        self.active_track_ids.clear()
        self.pending_tandem_candidates.clear()

        self.total_entries = 0
        self.frame_index = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.current_timestamp = None

        for gate in self.gates:
            gate.entry_count = 0

        self._reset_ultralytics_tracker()
        self.save_events_csv()

    def _reset_ultralytics_tracker(self) -> None:
        """Reset tracker state without unloading YOLO weights."""

        if self.detector is None:
            return

        predictor = getattr(self.detector, "predictor", None)
        trackers = getattr(predictor, "trackers", None)

        if trackers:
            for tracker in trackers:
                reset_method = getattr(tracker, "reset", None)
                if callable(reset_method):
                    reset_method()

    def load_models(
        self,
        yolo_model_path: Path = YOLO_MODEL_PATH,
        reid_model_path: Path = REID_MODEL_PATH,
    ) -> None:
        """Load YOLO and OSNet model files from the models directory."""

        self.model_load_error = None
        yolo_model_path = Path(yolo_model_path)
        reid_model_path = Path(reid_model_path)

        if not yolo_model_path.is_file():
            message = f"YOLO model file was not found: {yolo_model_path}"
            self.model_load_error = message
            raise FileNotFoundError(message)

        if not reid_model_path.is_file():
            message = f"OSNet model file was not found: {reid_model_path}"
            self.model_load_error = message
            raise FileNotFoundError(message)

        try:
            self.detector = YOLO(str(yolo_model_path))
            self.detector.to(self.device)
            self.reid_model = self._load_reid_model(reid_model_path)
        except Exception as exc:
            self.detector = None
            self.reid_model = None
            message = f"Failed to load computer-vision models: {exc}"
            self.model_load_error = message
            raise RuntimeError(message) from exc

    def _load_reid_model(self, model_path: Path) -> torch.nn.Module:
        # Use the bundled OSNet architecture instead of importing the full
        # torchreid package. This avoids torchreid/pkg_resources installation
        # failures on newer Windows Python environments.
        model = build_osnet_x0_25(num_classes=1000, loss="softmax")
        matched_layers, _ = load_pretrained_weights(
            model,
            model_path,
            map_location=self.device,
        )
        if matched_layers <= 0:
            raise RuntimeError("OSNet weights could not be loaded.")
        model = model.to(self.device)
        model.eval()
        return model

    def unload_models(self) -> None:
        self.detector = None
        self.reid_model = None
        self.model_load_error = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Re-identification
    # --------------------------------------------------------

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        if crop is None or crop.size == 0:
            raise ValueError("Empty crop cannot be preprocessed.")

        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        if self.reid_model is None:
            raise RuntimeError("OSNet model has not been loaded.")

        embedding = self.reid_model(self.preprocess_crop(crop))
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def embedding_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        if embedding_a is None or embedding_b is None:
            return -1.0
        return float(np.dot(embedding_a, embedding_b))

    def is_same_person(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> bool:
        return (
            self.embedding_similarity(embedding_a, embedding_b)
            >= REID_SIMILARITY_THRESHOLD
        )

    @staticmethod
    def crop_quality(crop: np.ndarray, confidence: float) -> float:
        if crop is None or crop.size == 0:
            return 0.0

        height, width = crop.shape[:2]
        if width <= 0 or height <= 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(sharpness / 500.0, 1.0)
        confidence_score = max(0.0, min(float(confidence), 1.0))

        return (
            float(width * height)
            * confidence_score
            * (0.75 + 0.25 * sharpness_score)
        )

    def update_track_appearance(
        self,
        frame: np.ndarray,
        track: TrackInfo,
    ) -> None:
        if self.reid_model is None:
            return

        if self.frame_index - track.last_reid_frame < REID_UPDATE_INTERVAL:
            return

        crop = self.crop_person(frame, track.bbox)
        if crop.size == 0:
            return

        crop_height, crop_width = crop.shape[:2]
        if (
            crop_width < REID_MIN_CROP_WIDTH
            or crop_height < REID_MIN_CROP_HEIGHT
        ):
            return

        quality = self.crop_quality(crop, track.confidence)
        required_quality = track.best_quality * REID_QUALITY_IMPROVEMENT

        if track.best_embedding is not None and quality <= required_quality:
            track.last_reid_frame = self.frame_index
            return

        try:
            embedding = self.extract_embedding(crop)
        except Exception:
            track.last_reid_frame = self.frame_index
            return

        track.best_crop = crop.copy()
        track.best_embedding = embedding
        track.best_quality = quality
        track.last_reid_frame = self.frame_index

        if SAVE_CROPS:
            cv2.imwrite(
                str(CROP_DIR / f"track_{track.track_id}.jpg"),
                track.best_crop,
            )

    def _visitor_has_other_active_track(
        self,
        visitor_id: str,
        current_track_id: int,
    ) -> bool:
        """Return True when this identity is already visible in another box."""

        for active_track_id in self.active_track_ids:
            if active_track_id == current_track_id:
                continue
            active_track = self.tracks.get(active_track_id)
            if active_track and active_track.visitor_id == visitor_id:
                return True
        return False

    def _recent_id_switch_candidate(
        self,
        visitor: VisitorIdentity,
        track: TrackInfo,
    ) -> bool:
        """Allow only a very local, immediate tracker-ID recovery.

        This handles one person briefly losing their tracker box without
        allowing two people entering together to be merged.
        """

        if visitor.last_seen_frame < 0 or visitor.last_foot is None:
            return False

        frame_gap = self.frame_index - visitor.last_seen_frame
        # A tracker-ID recovery must follow at least one missing frame.  A
        # visitor updated earlier in the same frame belongs to another
        # simultaneously visible person and must never be reused.
        if frame_gap < 1 or frame_gap > REID_ID_SWITCH_MAX_GAP_FRAMES:
            return False
        if visitor.last_track_id in self.active_track_ids:
            return False

        current_foot = np.asarray(self.bbox_foot(track.bbox), dtype=np.float32)
        previous_foot = np.asarray(visitor.last_foot, dtype=np.float32)
        distance = float(np.linalg.norm(current_foot - previous_foot))
        frame_diagonal = max(
            float(np.hypot(self.frame_width, self.frame_height)),
            1.0,
        )
        return distance <= frame_diagonal * REID_ID_SWITCH_MAX_DISTANCE_RATIO

    def _resolve_visitor(
        self,
        track: TrackInfo,
        gate_name: str,
        direction: str,
    ) -> str:
        now = datetime.now()
        foot = self.bbox_foot(track.bbox)

        if track.visitor_id and track.visitor_id in self.visitors:
            visitor = self.visitors[track.visitor_id]
            visitor.last_seen = now
            visitor.last_gate = gate_name
            visitor.last_track_id = track.track_id
            visitor.last_seen_frame = self.frame_index
            visitor.last_foot = foot
            if direction == "entry":
                visitor.times_seen += 1
            self._update_visitor_embedding(visitor, track)
            return visitor.visitor_id

        best_visitor: Optional[VisitorIdentity] = None
        best_similarity = -1.0

        # Tracker IDs can change on RTSP streams even when the same person is
        # continuously visible.  Recover a very recent, nearby identity before
        # relying on appearance embeddings.  This also works when OSNet has not
        # produced a usable crop yet.  An identity already assigned to another
        # active box is never reused, so two simultaneous people stay distinct.
        local_candidates: list[tuple[float, VisitorIdentity]] = []
        current_foot = np.asarray(foot, dtype=np.float32)
        frame_diagonal = max(float(np.hypot(self.frame_width, self.frame_height)), 1.0)
        for visitor in self.visitors.values():
            if self._visitor_has_other_active_track(visitor.visitor_id, track.track_id):
                continue
            if visitor.last_seen_frame < 0 or visitor.last_foot is None:
                continue
            frame_gap = self.frame_index - visitor.last_seen_frame
            # Only reconnect an identity after its previous tracker has
            # actually disappeared.  This prevents two people crossing side
            # by side in the same frame from being collapsed into one visitor.
            if frame_gap < 1 or frame_gap > REID_ID_SWITCH_MAX_GAP_FRAMES:
                continue
            if visitor.last_track_id in self.active_track_ids:
                continue
            previous_foot = np.asarray(visitor.last_foot, dtype=np.float32)
            distance = float(np.linalg.norm(current_foot - previous_foot))
            if distance <= frame_diagonal * REID_ID_SWITCH_MAX_DISTANCE_RATIO:
                local_candidates.append((distance, visitor))

        if local_candidates:
            local_candidates.sort(key=lambda item: item[0])
            nearest_distance, nearest_visitor = local_candidates[0]
            # Require the nearest candidate to be clearly better when more than
            # one recently missing person is nearby.
            unambiguous = (
                len(local_candidates) == 1
                or nearest_distance + frame_diagonal * 0.025 < local_candidates[1][0]
            )
            if unambiguous:
                nearest_visitor.last_seen = now
                nearest_visitor.last_gate = gate_name
                nearest_visitor.last_track_id = track.track_id
                nearest_visitor.last_seen_frame = self.frame_index
                nearest_visitor.last_foot = foot
                self._update_visitor_embedding(nearest_visitor, track)
                track.visitor_id = nearest_visitor.visitor_id
                return nearest_visitor.visitor_id

        if track.best_embedding is not None:
            for visitor in self.visitors.values():
                if visitor.embedding is None:
                    continue

                # Two simultaneously visible tracks are two different people.
                if self._visitor_has_other_active_track(
                    visitor.visitor_id,
                    track.track_id,
                ):
                    continue

                frame_gap = (
                    self.frame_index - visitor.last_seen_frame
                    if visitor.last_seen_frame >= 0
                    else REID_RECENT_DISTINCT_WINDOW_FRAMES + 1
                )
                is_id_switch = self._recent_id_switch_candidate(visitor, track)

                # A second person following immediately behind the first is
                # treated as distinct unless this is a tight spatial ID switch.
                if (
                    frame_gap <= REID_RECENT_DISTINCT_WINDOW_FRAMES
                    and not is_id_switch
                ):
                    continue

                similarity = self.embedding_similarity(
                    track.best_embedding,
                    visitor.embedding,
                )
                required_similarity = (
                    REID_ID_SWITCH_SIMILARITY_THRESHOLD
                    if is_id_switch
                    else REID_SIMILARITY_THRESHOLD
                )
                if (
                    similarity >= required_similarity
                    and similarity > best_similarity
                ):
                    best_similarity = similarity
                    best_visitor = visitor

        if best_visitor is not None:
            best_visitor.last_seen = now
            best_visitor.last_gate = gate_name
            best_visitor.last_track_id = track.track_id
            best_visitor.last_seen_frame = self.frame_index
            best_visitor.last_foot = foot
            if direction == "entry":
                best_visitor.times_seen += 1
            self._update_visitor_embedding(best_visitor, track)
            track.visitor_id = best_visitor.visitor_id
            return best_visitor.visitor_id

        visitor_id = f"V-{uuid.uuid4().hex[:8].upper()}"
        visitor = VisitorIdentity(
            visitor_id=visitor_id,
            embedding=(
                track.best_embedding.copy()
                if track.best_embedding is not None
                else None
            ),
            best_crop=(
                track.best_crop.copy()
                if track.best_crop is not None
                else None
            ),
            first_seen=now,
            last_seen=now,
            times_seen=1,
            last_gate=gate_name,
            last_track_id=track.track_id,
            last_seen_frame=self.frame_index,
            last_foot=foot,
        )
        self.visitors[visitor_id] = visitor
        track.visitor_id = visitor_id

        if SAVE_CROPS and visitor.best_crop is not None:
            cv2.imwrite(
                str(CROP_DIR / f"visitor_{visitor_id}.jpg"),
                visitor.best_crop,
            )

        return visitor_id

    def ensure_visitor_identity(
        self,
        track: TrackInfo,
        context: str = "observed",
    ) -> str:
        """Assign or re-identify a persistent visitor ID for a visible track.

        Counter tracking calls this before JourneyManager processes the track.
        This is essential: tracker IDs are short-lived, while visitor IDs are
        the appearance-based IDs used to reconnect somebody who returns later
        in the same 24-hour analysis.
        """

        return self._resolve_visitor(track, context, "observed")

    @staticmethod
    def _update_visitor_embedding(
        visitor: VisitorIdentity,
        track: TrackInfo,
    ) -> None:
        if track.best_embedding is None:
            return

        if visitor.embedding is None:
            visitor.embedding = track.best_embedding.copy()
        else:
            combined = 0.70 * visitor.embedding + 0.30 * track.best_embedding
            norm = float(np.linalg.norm(combined))
            if norm > 1e-8:
                visitor.embedding = (combined / norm).astype(np.float32)

        if track.best_quality > 0 and track.best_crop is not None:
            visitor.best_crop = track.best_crop.copy()

    # --------------------------------------------------------
    # Detection and tracking
    # --------------------------------------------------------

    def detect_people(self, frame: np.ndarray) -> list[Detection]:
        """Run plain YOLO detection for compatibility or debugging."""

        if self.detector is None:
            raise RuntimeError("YOLO model has not been loaded.")

        with self._detector_lock:
            results = self.detector.predict(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=INFERENCE_IMAGE_SIZE,
                max_det=100,
                agnostic_nms=False,
                device=self.device,
                verbose=False,
            )

        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            if y2 - y1 < MIN_PERSON_HEIGHT:
                continue
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(box.conf.item()),
                )
            )

        return detections

    def track_people(self, frame: np.ndarray) -> list[TrackInfo]:
        """Run YOLO with persistent BoT-SORT tracking."""

        if self.detector is None:
            raise RuntimeError("YOLO model has not been loaded.")

        with self._detector_lock:
            results = self.detector.track(
                frame,
                persist=True,
                tracker=self.tracking_config,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=INFERENCE_IMAGE_SIZE,
                max_det=100,
                agnostic_nms=False,
                device=self.device,
                verbose=False,
            )

        self.active_track_ids.clear()

        if not results or results[0].boxes is None:
            self.cleanup_tracks()
            return []

        current_time = time.time()

        for box in results[0].boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            if y2 - y1 < MIN_PERSON_HEIGHT:
                continue

            track_id = int(box.id.item())
            bbox = (int(x1), int(y1), int(x2), int(y2))
            confidence = float(box.conf.item())
            self.active_track_ids.add(track_id)

            track = self.tracks.get(track_id)
            if track is None:
                track = TrackInfo(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    first_seen=current_time,
                    first_seen_frame=self.frame_index,
                    last_seen=current_time,
                )
                self.tracks[track_id] = track
            else:
                track.bbox = bbox
                track.confidence = confidence
                track.last_seen = current_time
                track.lost_frames = 0

            foot = self.bbox_foot(bbox)
            track.history.append(foot)

            # Keep the identity's latest location fresh after it has been
            # confirmed. This lets re-identification distinguish an actual
            # return from another person entering at the same time.
            if track.visitor_id and track.visitor_id in self.visitors:
                visitor = self.visitors[track.visitor_id]
                visitor.last_seen = datetime.now()
                visitor.last_seen_frame = self.frame_index
                visitor.last_track_id = track.track_id
                visitor.last_foot = foot

        self.cleanup_tracks()

        active_tracks = [
            self.tracks[track_id]
            for track_id in sorted(self.active_track_ids)
            if track_id in self.tracks
        ]

        for track in active_tracks:
            self.update_track_appearance(frame, track)

        return active_tracks

    def cleanup_tracks(self) -> None:
        self._expire_tandem_candidates()

        for track_id, track in list(self.tracks.items()):
            if track_id in self.active_track_ids:
                continue

            # On the first missing frame, preserve a strictly qualified armed
            # outside track. This is not a guessed person: the track had to be
            # separately visible outside and moving toward the line first.
            if track.lost_frames == 0:
                self._register_tandem_candidate(track)

            track.lost_frames += 1
            if track.lost_frames > MAX_TRACK_AGE:
                del self.tracks[track_id]

    def _expire_tandem_candidates(self) -> None:
        for gate_name, candidates in list(self.pending_tandem_candidates.items()):
            remaining = [
                candidate
                for candidate in candidates
                if candidate.expires_frame >= self.frame_index
            ]
            if remaining:
                self.pending_tandem_candidates[gate_name] = remaining
            else:
                self.pending_tandem_candidates.pop(gate_name, None)

    def _register_tandem_candidate(self, track: TrackInfo) -> None:
        if track.counted or len(track.history) < 3:
            return

        for gate in self.gates:
            state = track.gate_pass_state.get(gate.name)
            if state is None or state.phase != "armed":
                continue
            if state.start_point is None or state.start_progress is None:
                continue
            if state.outside_frames < max(2, gate.outside_confirm_frames):
                continue
            if state.last_progress is None:
                continue

            deadband = max(7.0, self.frame_height * LINE_DEADBAND_RATIO)
            outside_band = max(
                45.0,
                self.frame_height * TANDEM_HANDOFF_OUTSIDE_BAND_RATIO,
            )
            if not (-outside_band <= state.last_progress <= -deadband):
                continue

            if len(state.recent_points) < 3:
                continue

            _, _tangent, inward_axis = self._line_axes(gate)
            points = np.asarray(state.recent_points, dtype=np.float32)
            total_inward = float(np.dot(points[-1] - points[0], inward_axis))
            minimum_approach = max(
                7.0,
                self.frame_height * TANDEM_HANDOFF_MIN_APPROACH_RATIO,
            )
            if total_inward < minimum_approach:
                continue

            candidate = PendingTandemCandidate(
                gate_name=gate.name,
                source_track_id=track.track_id,
                created_frame=self.frame_index,
                expires_frame=(
                    self.frame_index + TANDEM_HANDOFF_MAX_GAP_FRAMES
                ),
                start_point=state.start_point,
                start_progress=float(state.start_progress),
                last_point=state.recent_points[-1],
                last_progress=float(state.last_progress),
                recent_points=tuple(state.recent_points),
                embedding=(
                    track.best_embedding.copy()
                    if track.best_embedding is not None
                    else None
                ),
            )

            candidates = self.pending_tandem_candidates.setdefault(
                gate.name,
                [],
            )
            if not any(
                existing.source_track_id == track.track_id
                for existing in candidates
            ):
                candidates.append(candidate)

    def _try_tandem_handoff(
        self,
        track: TrackInfo,
        gate: Gate,
        foot: Point,
        progress: float,
        confirmation_threshold: float,
    ) -> bool:
        """Reconnect a new ID to a real outside track lost by occlusion."""

        track_age = max(1, self.frame_index - track.first_seen_frame + 1)
        if track_age > TANDEM_HANDOFF_NEW_TRACK_MAX_AGE_FRAMES:
            return False
        if progress <= 0:
            return False

        # Do not rescue a box that first appears far inside the park.
        maximum_start_progress = confirmation_threshold + max(
            30.0,
            self.frame_height * 0.08,
        )
        if progress > maximum_start_progress:
            return False

        candidates = self.pending_tandem_candidates.get(gate.name, [])
        if not candidates:
            return False

        frame_diagonal = max(
            float(np.hypot(self.frame_width, self.frame_height)),
            1.0,
        )
        maximum_distance = (
            frame_diagonal * TANDEM_HANDOFF_MAX_DISTANCE_RATIO
        )

        best_candidate: Optional[PendingTandemCandidate] = None
        best_score = float("inf")
        current_point = np.asarray(foot, dtype=np.float32)

        for candidate in candidates:
            if candidate.expires_frame < self.frame_index:
                continue
            if candidate.source_track_id == track.track_id:
                continue

            distance = float(
                np.linalg.norm(
                    current_point
                    - np.asarray(candidate.last_point, dtype=np.float32)
                )
            )
            if distance > maximum_distance:
                continue

            similarity_penalty = 0.0
            if (
                candidate.embedding is not None
                and track.best_embedding is not None
            ):
                similarity = self.embedding_similarity(
                    candidate.embedding,
                    track.best_embedding,
                )
                if similarity < TANDEM_HANDOFF_MIN_EMBEDDING_SIMILARITY:
                    continue
                similarity_penalty = (1.0 - similarity) * maximum_distance

            score = distance + similarity_penalty
            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return False

        state = track.gate_pass_state.setdefault(gate.name, GatePassState())
        state.phase = "crossed_outer"
        state.outside_frames = max(2, gate.outside_confirm_frames)
        state.confirmation_frames = 0
        state.start_point = best_candidate.start_point
        state.start_progress = best_candidate.start_progress
        state.last_progress = progress
        state.recent_points.clear()
        state.recent_progress.clear()

        for point in best_candidate.recent_points:
            state.recent_points.append(point)
            _along, previous_progress = self._gate_measurements(point, gate)
            state.recent_progress.append(previous_progress)
        state.recent_points.append(foot)
        state.recent_progress.append(progress)

        candidates.remove(best_candidate)
        if not candidates:
            self.pending_tandem_candidates.pop(gate.name, None)

        # The old tracker ID is replaced by the new one after the handoff.
        self.tracks.pop(best_candidate.source_track_id, None)
        return True

    # --------------------------------------------------------
    # Gate counting
    # --------------------------------------------------------

    @staticmethod
    def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
        if len(polygon) < 3:
            return False
        contour = np.asarray(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(contour, point, False) >= 0

    @staticmethod
    def _line_axes(gate: Gate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(gate.count_line[0], dtype=np.float32)
        b = np.asarray(gate.count_line[1], dtype=np.float32)
        tangent = b - a
        length = float(np.linalg.norm(tangent))
        if length < 1e-6:
            raise ValueError(f"{gate.name} has an invalid count line.")
        tangent /= length
        left_normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float32)

        # With user-drawn lines, line endpoint order must not change the entry
        # direction.  Whenever a green confirmation line exists, derive inward
        # from the yellow-line midpoint toward the green-line midpoint.
        if gate.confirmation_line is not None:
            yellow_mid = (a + b) * 0.5
            c = np.asarray(gate.confirmation_line[0], dtype=np.float32)
            d = np.asarray(gate.confirmation_line[1], dtype=np.float32)
            green_mid = (c + d) * 0.5
            toward_green = green_mid - yellow_mid
            if float(np.dot(toward_green, left_normal)) >= 0:
                inward = left_normal
            else:
                inward = -left_normal
        else:
            inward = -left_normal if gate.entry_from_bottom else left_normal
        return a, tangent, inward

    def _gate_measurements(
        self,
        point: Point,
        gate: Gate,
    ) -> tuple[float, float]:
        a, tangent, inward = self._line_axes(gate)
        vector = np.asarray(point, dtype=np.float32) - a
        along = float(np.dot(vector, tangent))
        inward_progress = float(np.dot(vector, inward))
        return along, inward_progress

    @staticmethod
    def _reset_gate_pass_state(state: GatePassState) -> None:
        state.phase = "unarmed"
        state.outside_frames = 0
        state.confirmation_frames = 0
        state.start_point = None
        state.start_progress = None
        state.last_progress = None
        state.recent_progress.clear()
        state.recent_points.clear()

    def _confirmation_distance(self, gate: Gate) -> float:
        """Return the inward distance from the yellow to the green line."""

        if gate.confirmation_line is None:
            return max(28.0, self.frame_height * 0.08)

        a, _tangent, inward = self._line_axes(gate)
        c = np.asarray(gate.confirmation_line[0], dtype=np.float32)
        d = np.asarray(gate.confirmation_line[1], dtype=np.float32)
        confirmation_midpoint = (c + d) * 0.5
        yellow_midpoint = (
            np.asarray(gate.count_line[0], dtype=np.float32)
            + np.asarray(gate.count_line[1], dtype=np.float32)
        ) * 0.5
        distance = float(np.dot(confirmation_midpoint - yellow_midpoint, inward))
        return max(distance, 1.0)

    def _has_consistent_inward_motion(
        self,
        state: GatePassState,
        gate: Gate,
        *,
        minimum_total_inward: float,
    ) -> bool:
        """Reject box jitter, sideways walking, and reverse travel."""

        if len(state.recent_points) < 4:
            return False

        _, tangent_axis, inward_axis = self._line_axes(gate)
        points = np.asarray(state.recent_points, dtype=np.float32)
        step_vectors = np.diff(points, axis=0)
        inward_steps = step_vectors @ inward_axis

        positive_steps = int(np.sum(inward_steps > 1.5))
        reverse_steps = int(np.sum(inward_steps < -2.5))

        total_vector = points[-1] - points[0]
        total_inward = float(np.dot(total_vector, inward_axis))
        total_lateral = abs(float(np.dot(total_vector, tangent_axis)))

        return (
            positive_steps >= max(2, len(inward_steps) // 2)
            and reverse_steps <= 1
            and total_inward >= minimum_total_inward
            and total_lateral <= total_inward * MAX_LATERAL_TO_INWARD_RATIO
        )

    def is_track_near_any_gate(self, track: TrackInfo) -> bool:
        foot = self.bbox_foot(track.bbox)
        return any(
            self._point_in_polygon(foot, gate.corridor_polygon)
            for gate in self.gates
        )

    def people_near_gates(self, tracks: Iterable[TrackInfo]) -> int:
        return sum(1 for track in tracks if self.is_track_near_any_gate(track))

    def update_gate_events(
        self,
        frame: np.ndarray,
        tracks: Iterable[TrackInfo],
    ) -> None:
        for track in tracks:
            for gate in self.gates:
                self._update_track_gate(frame, track, gate)

    def _update_track_gate(
        self,
        frame: np.ndarray,
        track: TrackInfo,
        gate: Gate,
    ) -> None:
        """Count only a fully observed outside -> yellow -> green passage."""

        state = track.gate_pass_state.setdefault(gate.name, GatePassState())
        if state.counted or track.counted:
            return

        foot = self.bbox_foot(track.bbox)
        in_corridor = self._point_in_polygon(foot, gate.corridor_polygon)

        if not in_corridor:
            if (
                state.last_in_corridor_frame >= 0
                and self.frame_index - state.last_in_corridor_frame
                <= CORRIDOR_GRACE_FRAMES
            ):
                return
            self._reset_gate_pass_state(state)
            return

        state.last_in_corridor_frame = self.frame_index
        _, progress = self._gate_measurements(foot, gate)
        state.last_progress = progress
        state.recent_progress.append(progress)
        state.recent_points.append(foot)

        deadband = max(7.0, self.frame_height * LINE_DEADBAND_RATIO)
        confirmation_distance = self._confirmation_distance(gate)
        confirmation_threshold = confirmation_distance + deadband
        outside_frames_required = max(2, gate.outside_confirm_frames)

        # Region 1: clearly outside the yellow line.
        if progress <= -deadband:
            if state.phase == "crossed_outer":
                # The person crossed the yellow line but reversed before the
                # green line. Cancel that attempt completely.
                self._reset_gate_pass_state(state)
                state.recent_progress.append(progress)
                state.recent_points.append(foot)

            state.confirmation_frames = 0
            state.outside_frames += 1
            if state.outside_frames >= outside_frames_required:
                state.phase = "armed"
                if (
                    state.start_progress is None
                    or progress < state.start_progress
                ):
                    state.start_progress = progress
                    state.start_point = foot
            return

        # Neutral strip around the yellow line. Outside observations must be
        # consecutive before arming; after arming, waiting is allowed.
        if abs(progress) < deadband:
            state.confirmation_frames = 0
            if state.phase == "unarmed":
                state.outside_frames = 0
            return

        # A track first seen on the inward side is normally ignored. The only
        # exception is a constrained tandem handoff: a separate track must
        # already have been visibly confirmed outside, moving toward this same
        # gate, and then disappeared only briefly because of occlusion.
        if state.phase not in {"armed", "crossed_outer"} or state.start_point is None:
            rescued = self._try_tandem_handoff(
                track,
                gate,
                foot,
                progress,
                confirmation_threshold,
            )
            if not rescued:
                state.outside_frames = 0
                state.confirmation_frames = 0
                return
            state = track.gate_pass_state[gate.name]

        # Region 2: crossed yellow but has not yet reached green.
        if progress < confirmation_threshold:
            state.phase = "crossed_outer"
            state.confirmation_frames = 0
            return

        # Region 3: beyond the green confirmation line. Require consecutive
        # frames here plus sustained inward travel from the confirmed outside
        # starting point.
        state.phase = "crossed_outer"
        state.confirmation_frames += 1
        if state.confirmation_frames < INSIDE_CONFIRM_FRAMES:
            return

        _, tangent_axis, inward_axis = self._line_axes(gate)
        displacement = (
            np.asarray(foot, dtype=np.float32)
            - np.asarray(state.start_point, dtype=np.float32)
        )
        inward_travel = float(np.dot(displacement, inward_axis))
        lateral_travel = abs(float(np.dot(displacement, tangent_axis)))

        required_inward_travel = max(
            confirmation_distance + deadband,
            self.frame_height * MIN_INWARD_TRAVEL_RATIO,
        )
        if inward_travel < required_inward_travel:
            return
        if lateral_travel > inward_travel * MAX_LATERAL_TO_INWARD_RATIO:
            return
        if not self._has_consistent_inward_motion(
            state,
            gate,
            minimum_total_inward=max(12.0, confirmation_distance * 0.55),
        ):
            return

        state.count_method = "confirmed_two_line_crossing"
        self._record_entry(frame, track, gate)
        state.counted = True
        state.phase = "counted"
        track.counted = True

    def _event_allowed(self, track: TrackInfo, gate_name: str) -> bool:
        last_event_frame = track.gate_last_event_frame.get(
            gate_name,
            -EVENT_COOLDOWN_FRAMES,
        )
        return self.frame_index - last_event_frame >= EVENT_COOLDOWN_FRAMES

    def _record_entry(
        self,
        frame: np.ndarray,
        track: TrackInfo,
        gate: Gate,
    ) -> None:
        if not self._event_allowed(track, gate.name):
            return

        visitor_id = self._resolve_visitor(track, gate.name, "entry")
        visitor = self.visitors[visitor_id]
        is_new_person = not visitor.has_entered
        if is_new_person:
            visitor.has_entered = True

        track.gate_last_event_frame[gate.name] = self.frame_index
        track.last_gate = gate.name

        # Every confirmed physical crossing is emitted.  A returning known
        # person does not increase the unique visitor total, but the event is
        # required so JourneyManager can change a prior counter-only record to
        # later-entered, even several hours afterward.
        if is_new_person:
            self.total_entries += 1
            gate.entry_count += 1

        event = EntryEvent(
            timestamp=self.current_timestamp or datetime.now(),
            visitor_id=visitor_id,
            track_id=track.track_id,
            gate_name=gate.name,
            direction="entry",
            confidence=track.confidence,
            counted_unique=is_new_person,
        )
        self.events.append(event)
        self.save_events_csv()

        if SAVE_SNAPSHOTS:
            filename = (
                f"{event.timestamp:%Y%m%d_%H%M%S_%f}_"
                f"{gate.name.replace(' ', '_')}_entry.jpg"
            )
            cv2.imwrite(str(SNAPSHOT_DIR / filename), frame)

    # --------------------------------------------------------
    # Rendering and exports
    # --------------------------------------------------------

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackInfo],
        show_boxes: bool = True,
        show_track_ids: bool = False,
        show_confidence: bool = False,
        show_trails: bool = False,
        show_fps: bool = True,
        show_gates: bool = True,
    ) -> np.ndarray:
        canvas = frame.copy()

        if show_gates:
            self._draw_gates(canvas)

        for track in tracks:
            if not self.is_track_near_any_gate(track):
                continue
            x1, y1, x2, y2 = track.bbox

            if show_boxes:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)

            if show_trails and len(track.history) > 1:
                points = np.asarray(track.history, dtype=np.int32)
                cv2.polylines(canvas, [points], False, CYAN, 2)

            label_parts: list[str] = []
            if show_track_ids:
                label_parts.append(f"T{track.track_id}")
            if track.visitor_id:
                label_parts.append(track.visitor_id)
            if show_confidence:
                label_parts.append(f"{track.confidence:.2f}")

            if label_parts:
                self._draw_label(
                    canvas,
                    " | ".join(label_parts),
                    (x1, max(18, y1 - 7)),
                    GREEN,
                )

        self._draw_dashboard(canvas, show_fps)
        return canvas

    def _draw_gates(self, frame: np.ndarray) -> None:
        overlay = frame.copy()

        for gate in self.gates:
            contour = np.asarray(gate.corridor_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [contour], ORANGE)
            cv2.polylines(frame, [contour], True, ORANGE, 2)
            cv2.line(
                frame,
                gate.count_line[0],
                gate.count_line[1],
                YELLOW,
                3,
            )
            if gate.confirmation_line is not None:
                cv2.line(
                    frame,
                    gate.confirmation_line[0],
                    gate.confirmation_line[1],
                    GREEN,
                    3,
                )

            label_point = gate.count_line[0]
            self._draw_label(
                frame,
                f"{gate.name}  ENTRIES:{gate.entry_count}",
                (label_point[0], max(20, label_point[1] - 8)),
                YELLOW,
            )

        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    def _draw_dashboard(self, frame: np.ndarray, show_fps: bool) -> None:
        height = 48
        cv2.rectangle(frame, (0, 0), (frame.shape[1], height), BLACK, -1)

        near_gate_count = sum(
            1
            for track_id in self.active_track_ids
            if track_id in self.tracks
            and self.is_track_near_any_gate(self.tracks[track_id])
        )
        text = (
            f"Entries: {self.total_entries}    "
            f"Unique: {self.total_unique_visitors}    "
            f"Near gates: {near_gate_count}"
        )
        if show_fps:
            text += f"    FPS: {self.fps:.1f}"

        cv2.putText(
            frame,
            text,
            (12, 31),
            FONT,
            0.68,
            WHITE,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        anchor: Point,
        color: tuple[int, int, int],
    ) -> None:
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            FONT,
            0.52,
            1,
        )
        x, y = anchor
        x = max(0, min(x, frame.shape[1] - text_width - 8))
        y = max(text_height + 5, min(y, frame.shape[0] - baseline - 4))

        cv2.rectangle(
            frame,
            (x, y - text_height - 5),
            (x + text_width + 7, y + baseline + 2),
            BLACK,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x + 3, y),
            FONT,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    def events_dataframe(self) -> pd.DataFrame:
        columns = [
            "timestamp",
            "visitor_id",
            "track_id",
            "gate_name",
            "direction",
            "confidence",
            "counted_unique",
        ]

        rows = [
            {
                "timestamp": event.timestamp.isoformat(timespec="seconds"),
                "visitor_id": event.visitor_id,
                "track_id": event.track_id,
                "gate_name": event.gate_name,
                "direction": event.direction,
                "confidence": round(event.confidence, 4),
                "counted_unique": bool(event.counted_unique),
            }
            for event in self.events
        ]
        return pd.DataFrame(rows, columns=columns)

    def gate_summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gate": gate.name,
                    "entries": gate.entry_count,
                }
                for gate in self.gates
            ]
        )

    def save_events_csv(self, path: Path = EVENT_CSV_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.events_dataframe().to_csv(path, index=False)
        return path

    # --------------------------------------------------------
    # Frame processing
    # --------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        annotate: bool = True,
    ) -> tuple[np.ndarray, list[TrackInfo]]:
        """Process one frame and return the output frame and active tracks."""

        if frame is None or frame.size == 0:
            raise ValueError("A valid video frame is required.")

        self.frame_index += 1
        self.frame_height, self.frame_width = frame.shape[:2]

        current_time = time.time()
        elapsed = max(current_time - self.last_frame_time, 1e-6)
        instantaneous_fps = 1.0 / elapsed
        self.fps = (
            instantaneous_fps
            if self.fps <= 0
            else 0.90 * self.fps + 0.10 * instantaneous_fps
        )
        self.last_frame_time = current_time

        tracks = self.track_people(frame)
        self.update_gate_events(frame, tracks)

        output = self.draw_frame(frame, tracks) if annotate else frame.copy()
        return output, tracks

    # --------------------------------------------------------
    # Geometry helpers
    # --------------------------------------------------------

    @staticmethod
    def crop_person(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, frame.shape[1]))
        x2 = max(0, min(x2, frame.shape[1]))
        y1 = max(0, min(y1, frame.shape[0]))
        y2 = max(0, min(y2, frame.shape[0]))

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def bbox_foot(bbox: BoundingBox) -> Point:
        """Use the bottom-centre of the person box as the floor contact point."""
        x1, _y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    @staticmethod
    def bbox_center(bbox: BoundingBox) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


__all__ = [
    "BASE_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "YOLO_MODEL_PATH",
    "REID_MODEL_PATH",
    "EVENT_CSV_PATH",
    "Detection",
    "EntryEvent",
    "Gate",
    "GatePassState",
    "TrackInfo",
    "VideoProcessor",
    "VisitorIdentity",
    "create_default_gates",
]
