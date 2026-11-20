"""Matching rules: which person or belonging occupies which seat cell.

The score for a person/seat pair blends bounding-box IoU with a "the person's
feet are inside this seat" test, because a seated person's box is much taller
than the chair and their IoU is small even when they are clearly sitting on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SeatConfig
from .detector import Detection
from .geometry import (
    Box,
    area,
    bottom_center,
    center,
    containment,
    expand,
    iou,
    point_inside,
)
from .tracking import Seat


@dataclass
class CellObservation:
    """What was seen on one seat cell in one frame."""

    seat_id: int
    cell_index: int
    box: Box
    person_id: int | None = None
    objects: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class AssociationResult:
    cells: dict[tuple[int, int], CellObservation]
    unmatched_persons: list[Detection]


def person_seat_score(person_box: Box, cell_box: Box, cfg: SeatConfig) -> float:
    overlap = iou(person_box, cell_box)
    zone = expand(cell_box, cfg.attach_margin_x, cfg.attach_margin_y)
    seated = 1.0 if point_inside(zone, bottom_center(person_box)) else 0.0
    return overlap + 0.35 * seated


def is_carried(
    item: Detection, persons: list[Detection], cfg: SeatConfig
) -> bool:
    """True when a belonging overlaps a person closely enough to be worn/held."""
    item_center = center(item.box)
    for person in persons:
        if iou(item.box, person.box) >= cfg.belonging_held_iou:
            return True
        if point_inside(person.box, item_center):
            return True
    return False


def associate(
    seats: list[Seat],
    persons: list[Detection],
    belongings: list[Detection],
    cfg: SeatConfig,
    table_boxes: list[Box] | None = None,
) -> AssociationResult:
    """Assign people and left-behind objects to seat cells.

    A cell takes at most one person.  A synthetic (person-derived) seat is
    always occupied by the person it was inferred from.
    """
    tables = table_boxes or []
    cells: dict[tuple[int, int], CellObservation] = {}
    for seat in seats:
        for index, box in enumerate(seat.cells):
            cells[(seat.seat_id, index)] = CellObservation(
                seat_id=seat.seat_id, cell_index=index, box=box
            )

    candidates: list[tuple[float, Detection, int, int]] = []
    for person in persons:
        for seat in seats:
            if seat.synthetic:
                continue
            for index, box in enumerate(seat.cells):
                score = person_seat_score(person.box, box, cfg)
                if score >= cfg.person_min_score:
                    candidates.append((score, person, seat.seat_id, index))

    candidates.sort(key=lambda c: c[0], reverse=True)
    occupied: set[tuple[int, int]] = set()
    assigned_persons: set[int] = set()
    for score, person, seat_id, index in candidates:
        key = (seat_id, index)
        if key in occupied:
            continue
        person_key = id(person)
        if person_key in assigned_persons:
            continue
        observation = cells[key]
        observation.person_id = person.track_id
        observation.score = score
        occupied.add(key)
        assigned_persons.add(person_key)

    unmatched = [p for p in persons if id(p) not in assigned_persons]

    for item in belongings:
        if is_carried(item, persons, cfg):
            continue
        if any(
            containment(item.box, table) >= cfg.belonging_table_ratio
            for table in tables
        ):
            continue
        item_center = center(item.box)
        item_area = area(item.box)
        best_key: tuple[int, int] | None = None
        best_ratio = cfg.belonging_inside_ratio
        for key, observation in cells.items():
            cell_box = observation.box
            cell_height = cell_box[3] - cell_box[1]
            if cell_height <= 0:
                continue
            depth = (item_center[1] - cell_box[1]) / cell_height
            if depth < cfg.belonging_min_depth:
                continue
            # A "bag" covering half the chair is almost always a bad detection.
            if item_area > cfg.belonging_max_area_ratio * area(cell_box):
                continue
            ratio = containment(item.box, cell_box)
            if ratio > best_ratio and point_inside(cell_box, item_center):
                best_key, best_ratio = key, ratio
        if best_key is not None:
            cells[best_key].objects.append(item.label)

    return AssociationResult(cells=cells, unmatched_persons=unmatched)
