from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

Point = Tuple[int, int]
NormPoint = Tuple[float, float]


@dataclass
class RegionConfig:
    frame_width: int
    frame_height: int
    counter_polygon: List[NormPoint]
    gate_polygons: Dict[str, List[NormPoint]]
    outer_lines: Dict[str, List[NormPoint]]
    confirmation_lines: Dict[str, List[NormPoint]]
    ignore_polygons: List[List[NormPoint]]
    version: int = 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RegionConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def normalize_points(points, width: int, height: int):
    return [(max(0.0, min(1.0, float(x) / width)), max(0.0, min(1.0, float(y) / height))) for x, y in points]


def denormalize_points(points, width: int, height: int):
    return [(int(round(float(x) * width)), int(round(float(y) * height))) for x, y in points]


def canvas_object_points(obj: dict) -> List[Point]:
    """Convert the latest Fabric.js canvas object into canvas pixel points.

    The drawable-canvas forks can emit free polygons as either ``polygon``
    objects with a ``points`` array or ``path`` objects with drawing commands.
    This function supports both forms and accounts for Fabric's local object
    coordinate system.
    """
    kind = str(obj.get("type", "")).lower()
    left = float(obj.get("left", 0) or 0)
    top = float(obj.get("top", 0) or 0)
    sx = float(obj.get("scaleX", 1) or 1)
    sy = float(obj.get("scaleY", 1) or 1)

    if kind == "line":
        x1 = float(obj.get("x1", 0) or 0)
        y1 = float(obj.get("y1", 0) or 0)
        x2 = float(obj.get("x2", 0) or 0)
        y2 = float(obj.get("y2", 0) or 0)
        min_x, min_y = min(x1, x2), min(y1, y2)
        return [
            (int(round(left + (x1 - min_x) * sx)), int(round(top + (y1 - min_y) * sy))),
            (int(round(left + (x2 - min_x) * sx)), int(round(top + (y2 - min_y) * sy))),
        ]

    if kind == "rect":
        width = float(obj.get("width", 0) or 0) * sx
        height = float(obj.get("height", 0) or 0) * sy
        x0, y0 = int(round(left)), int(round(top))
        x1, y1 = int(round(left + width)), int(round(top + height))
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    raw_points: list[tuple[float, float]] = []
    if kind == "polygon":
        for point in obj.get("points") or []:
            if isinstance(point, dict) and "x" in point and "y" in point:
                raw_points.append((float(point["x"]), float(point["y"])))

    elif kind == "path":
        # Typical Fabric path commands: ["M", x, y], ["L", x, y],
        # ["Q", cx, cy, x, y], ["C", ..., x, y], and ["Z"].
        for command in obj.get("path") or []:
            if not isinstance(command, (list, tuple)) or len(command) < 3:
                continue
            op = str(command[0]).upper()
            nums = command[1:]
            try:
                values = [float(v) for v in nums]
            except (TypeError, ValueError):
                continue
            if op in {"M", "L", "T"} and len(values) >= 2:
                raw_points.append((values[-2], values[-1]))
            elif op in {"Q", "S", "C"} and len(values) >= 2:
                raw_points.append((values[-2], values[-1]))

    if raw_points:
        min_x = min(x for x, _ in raw_points)
        min_y = min(y for _, y in raw_points)
        points = [
            (int(round(left + (x - min_x) * sx)), int(round(top + (y - min_y) * sy)))
            for x, y in raw_points
        ]
        # Remove adjacent duplicate points and a duplicate closing point.
        deduped: List[Point] = []
        for point in points:
            if not deduped or point != deduped[-1]:
                deduped.append(point)
        if len(deduped) > 1 and deduped[0] == deduped[-1]:
            deduped.pop()
        return deduped

    return []

def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(polygon_to_contour(polygon), point, False) >= 0


def polygon_to_contour(poly):
    import numpy as np
    return np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
