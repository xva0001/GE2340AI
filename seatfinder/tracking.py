"""Seat inventory (furniture tracking) and person motion bookkeeping.

Two independent pieces live here:

* :class:`SeatTracker` turns per-frame furniture detections into stable seat
  ids, estimates how many people a couch or bench seats, and keeps a seat alive
  for a short while when the detector loses it (someone sitting down hides the
  chair behind them).
* :class:`GlobalMotion` + :class:`PersonHistory` measure how still a person is
  *relative to the scene*, which is what separates a seated person from a
  passer-by even when the camera itself is panning.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .config import SeatConfig
from .detector import Detection
from .geometry import (
    Box,
    area,
    bottom_center,
    center,
    clamp,
    containment,
    distance,
    height,
    iou,
    median,
    split_horizontally,
    width,
)

SYNTHETIC_SEAT_OFFSET = 100_000


@dataclass
class Seat:
    """One piece of furniture that provides one or more seats."""

    seat_id: int
    kind: str
    box: Box
    capacity: int
    first_frame: int
    last_frame: int
    confidence: float
    hits: int = 1
    synthetic: bool = False
    observed: bool = False

    @property
    def cells(self) -> list[Box]:
        """Seat-sized slices of the furniture box, left to right."""
        return split_horizontally(self.box, self.capacity)

    @property
    def age(self) -> int:
        return self.last_frame - self.first_frame


def _blend(old: Box, new: Box, alpha: float) -> Box:
    return (
        old[0] + (new[0] - old[0]) * alpha,
        old[1] + (new[1] - old[1]) * alpha,
        old[2] + (new[2] - old[2]) * alpha,
        old[3] + (new[3] - old[3]) * alpha,
    )


def estimate_capacity(
    kind: str, box: Box, person_widths: list[float], fraction: float
) -> int:
    """Guess how many people a piece of furniture seats.

    Chairs always seat one.  Couches and benches are divided by a typical
    person width; without any person in the frame we fall back to a default so
    an empty couch still shows a plausible number of seats.
    """
    if kind not in ("couch", "bench"):
        return 1

    reference = median([w for w in person_widths if w > 0])
    if reference <= 1.0:
        default = 2 if kind == "couch" else 3
        return default

    divisor = reference * (0.95 if kind == "couch" else 1.05)
    capacity = int(round(width(box) / max(divisor, 1.0)))
    upper = 4 if kind == "couch" else 6
    capacity = int(clamp(capacity, 1, upper))
    # Never produce cells narrower than a few centimetres of the image.
    max_cells = max(1, int(width(box) // 24))
    return max(1, min(capacity, max_cells))


class SeatTracker:
    """Maintains a stable seat inventory across frames."""

    def __init__(self, cfg: SeatConfig) -> None:
        self.cfg = cfg
        self._seats: dict[int, Seat] = {}
        self._synthetic: dict[int, Seat] = {}
        self._next_id = 1

    # -- furniture ---------------------------------------------------------
    def update(
        self,
        furniture: list[Detection],
        frame_idx: int,
        person_widths: list[float],
    ) -> list[Seat]:
        """Match this frame's furniture detections to known seats."""
        detections = [d for d in furniture if d.confidence >= self.cfg.seat_conf]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = self._deduplicate(detections)

        taken: set[int] = set()
        for det in detections:
            seat, score = self._best_match(det.box, taken)
            if seat is None:
                seat = self._create(det, frame_idx, person_widths)
            else:
                taken.add(seat.seat_id)
                seat.box = _blend(seat.box, det.box, 0.35)
                seat.last_frame = frame_idx
                seat.confidence = max(seat.confidence, det.confidence)
                seat.hits += 1
            seat.observed = True

        return self._live(frame_idx)

    def _deduplicate(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for det in detections:
            duplicate = False
            for other in kept:
                if iou(det.box, other.box) > self.cfg.duplicate_iou:
                    duplicate = True
                    break
                small, big = (
                    (det.box, other.box)
                    if area(det.box) < area(other.box)
                    else (other.box, det.box)
                )
                if containment(small, big) > self.cfg.duplicate_containment:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    def _best_match(self, box: Box, taken: set[int]) -> tuple[Seat | None, float]:
        best: Seat | None = None
        best_score = 0.0
        for seat in self._seats.values():
            if seat.seat_id in taken:
                continue
            score = iou(box, seat.box)
            if score < self.cfg.match_iou:
                # A box that jumps (camera pan, detector jitter) still belongs
                # to the same chair when it is nearby and similarly sized.
                scale = max(width(seat.box), height(seat.box), 1.0)
                centre_gap = distance(center(box), center(seat.box)) / scale
                size_ratio = area(box) / max(area(seat.box), 1.0)
                if centre_gap <= 0.35 and 0.5 <= size_ratio <= 2.0:
                    score = self.cfg.match_iou
            if score >= self.cfg.match_iou and score > best_score:
                best, best_score = seat, score
        return best, best_score

    def _create(
        self, det: Detection, frame_idx: int, person_widths: list[float]
    ) -> Seat:
        capacity = estimate_capacity(
            det.label, det.box, person_widths, self.cfg.person_width_fraction
        )
        seat = Seat(
            seat_id=self._next_id,
            kind=det.label,
            box=det.box,
            capacity=capacity,
            first_frame=frame_idx,
            last_frame=frame_idx,
            confidence=det.confidence,
        )
        self._seats[seat.seat_id] = seat
        self._next_id += 1
        return seat

    # -- synthetic (inferred) seats ---------------------------------------
    def upsert_synthetic(
        self, track_id: int, box: Box, frame_idx: int, confidence: float
    ) -> Seat:
        """Register/refresh a seat inferred from a stationary person."""
        seat = self._synthetic.get(track_id)
        if seat is None:
            seat = Seat(
                seat_id=SYNTHETIC_SEAT_OFFSET + track_id,
                kind="inferred",
                box=box,
                capacity=1,
                first_frame=frame_idx,
                last_frame=frame_idx,
                confidence=confidence,
                synthetic=True,
            )
            self._synthetic[track_id] = seat
        else:
            seat.box = _blend(seat.box, box, 0.35)
            seat.last_frame = frame_idx
            seat.confidence = confidence
            seat.hits += 1
        seat.observed = True
        return seat

    def synthetic_for(self, track_id: int) -> Seat | None:
        """The inferred seat belonging to a person track, if any."""
        return self._synthetic.get(track_id)

    def prune_synthetic(
        self, active_track_ids: set[int], frame_idx: int, grace_frames: int = 12
    ) -> None:
        for track_id in list(self._synthetic):
            seat = self._synthetic[track_id]
            if track_id not in active_track_ids and (
                frame_idx - seat.last_frame > grace_frames
            ):
                del self._synthetic[track_id]

    def _live(self, frame_idx: int) -> list[Seat]:
        for seat_id in list(self._seats):
            seat = self._seats[seat_id]
            stale = frame_idx - seat.last_frame
            if stale > self.cfg.ttl_frames:
                del self._seats[seat_id]
            elif seat.hits < self.cfg.confirm_hits and stale > 2:
                del self._seats[seat_id]
        return self.all_seats()

    def all_seats(self) -> list[Seat]:
        seats = [
            seat
            for seat in self._seats.values()
            if seat.hits >= self.cfg.confirm_hits
        ]
        seats += list(self._synthetic.values())
        seats.sort(key=lambda s: s.seat_id)
        return seats

    def reset_frame_flags(self) -> None:
        for seat in self.all_seats():
            seat.observed = False


class GlobalMotion:
    """Estimates camera-induced image translation from person motion.

    The median displacement of all tracked people between two frames is almost
    entirely camera movement (people on average do not all walk the same way).
    Subtracting it lets "is this person still?" work on handheld footage too.
    """

    def __init__(self, max_step_fraction: float = 0.08, min_tracks: int = 3) -> None:
        self.max_step_fraction = max_step_fraction
        #: With fewer tracked people than this the median is just one person's
        #: own walking, so compensating would make a passer-by look still.
        self.min_tracks = min_tracks
        self._previous: dict[int, tuple[float, float]] = {}
        self._offset = (0.0, 0.0)

    @property
    def offset(self) -> tuple[float, float]:
        """Cumulative apparent motion, subtracted from person positions."""
        return self._offset

    def update(self, persons: list[Detection], frame_w: int) -> None:
        current = {
            p.track_id: bottom_center(p.box)
            for p in persons
            if p.track_id is not None
        }
        if self._previous:
            dxs: list[float] = []
            dys: list[float] = []
            for track_id, position in current.items():
                previous = self._previous.get(track_id)
                if previous is None:
                    continue
                dxs.append(position[0] - previous[0])
                dys.append(position[1] - previous[1])
            if len(dxs) >= self.min_tracks:
                limit = frame_w * self.max_step_fraction
                dx = clamp(median(dxs), -limit, limit)
                dy = clamp(median(dys), -limit, limit)
                self._offset = (self._offset[0] + dx, self._offset[1] + dy)
        self._previous = current


@dataclass
class _Track:
    positions: deque = field(default_factory=lambda: deque(maxlen=32))
    last_frame: int = 0


class PersonHistory:
    """Keeps a short motion history per tracked person."""

    def __init__(self, window: int = 24) -> None:
        self.window = window
        self._tracks: dict[int, _Track] = {}

    def update(
        self,
        frame_idx: int,
        persons: list[Detection],
        offset: tuple[float, float],
    ) -> None:
        for person in persons:
            if person.track_id is None:
                continue
            position = bottom_center(person.box)
            compensated = (position[0] - offset[0], position[1] - offset[1])
            track = self._tracks.get(person.track_id)
            if track is None:
                track = _Track(positions=deque(maxlen=self.window))
                self._tracks[person.track_id] = track
            track.positions.append(compensated)
            track.last_frame = frame_idx

        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_idx - track.last_frame > self.window * 2
        ]
        for track_id in stale:
            del self._tracks[track_id]

    def samples(self, track_id: int) -> int:
        track = self._tracks.get(track_id)
        return 0 if track is None else len(track.positions)

    def stillness(self, track_id: int) -> float | None:
        """How much a person drifts from their median position, in pixels.

        The 75th percentile of the deviations is used instead of the maximum:
        a single bad box (an arm swing, one noisy detection) should not make a
        seated person look like they are walking, which used to make inferred
        seats blink in and out.
        """
        track = self._tracks.get(track_id)
        if track is None or len(track.positions) < 3:
            return None
        xs = [p[0] for p in track.positions]
        ys = [p[1] for p in track.positions]
        mx = median(xs)
        my = median(ys)
        deviations = sorted(
            math.hypot(x - mx, y - my) for x, y in track.positions
        )
        index = int(round(0.75 * (len(deviations) - 1)))
        return deviations[index]
