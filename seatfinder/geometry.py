"""Axis-aligned bounding-box geometry helpers.

All boxes are ``(x1, y1, x2, y2)`` float tuples in pixel coordinates.
Every function here is pure and dependency free so the seating logic can be
unit-tested without loading a model or decoding a video.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Box = tuple[float, float, float, float]
Point = tuple[float, float]


def width(box: Box) -> float:
    return max(0.0, box[2] - box[0])


def height(box: Box) -> float:
    return max(0.0, box[3] - box[1])


def area(box: Box) -> float:
    return width(box) * height(box)


def center(box: Box) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bottom_center(box: Box) -> Point:
    return ((box[0] + box[2]) / 2.0, box[3])


def intersection(a: Box, b: Box) -> Box | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def intersection_area(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    return 0.0 if inter is None else area(inter)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner``'s area that falls inside ``outer``."""
    inner_area = area(inner)
    if inner_area <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner_area


def expand(box: Box, fx: float, fy: float | None = None) -> Box:
    """Grow a box by a fraction of its own width (``fx``) and height (``fy``)."""
    fy = fx if fy is None else fy
    dx = width(box) * fx
    dy = height(box) * fy
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def point_inside(box: Box, point: Point) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def clip(box: Box, frame_w: int, frame_h: int) -> Box:
    return (
        min(max(box[0], 0.0), float(frame_w)),
        min(max(box[1], 0.0), float(frame_h)),
        min(max(box[2], 0.0), float(frame_w)),
        min(max(box[3], 0.0), float(frame_h)),
    )


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def split_horizontally(box: Box, parts: int) -> list[Box]:
    """Split a box into ``parts`` equal vertical slices (left to right)."""
    parts = max(1, int(parts))
    if parts == 1:
        return [box]
    span = width(box) / parts
    return [
        (box[0] + i * span, box[1], box[0] + (i + 1) * span, box[3])
        for i in range(parts)
    ]


def mean_point(points: Sequence[Point]) -> Point:
    if not points:
        return (0.0, 0.0)
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def median(values: Iterable[float]) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0
