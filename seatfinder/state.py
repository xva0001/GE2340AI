"""Per-seat state machine with hysteresis.

Raw per-frame evidence is noisy: a detector may miss one frame, someone may
walk in front of a chair, a backpack may be picked up.  Every seat therefore
only changes state after the new state has been observed for a number of
consecutive frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import TemporalConfig


class SeatState(str, Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"
    GHOST = "ghost"


@dataclass
class CellStatus:
    """Smoothed state of one seat cell."""

    state: SeatState = SeatState.EMPTY
    pending: SeatState | None = None
    streak: int = 0
    since_frame: int = 0
    person_id: int | None = None
    objects: tuple[str, ...] = ()
    seen_frame: int = 0


class SeatStateMachine:
    """Debounces seat states per ``(seat_id, cell_index)`` key."""

    def __init__(self, cfg: TemporalConfig) -> None:
        self.cfg = cfg
        self._cells: dict[tuple[int, int], CellStatus] = {}

    def update(
        self,
        key: tuple[int, int],
        raw_state: SeatState,
        frame_idx: int,
        person_id: int | None = None,
        objects: tuple[str, ...] = (),
    ) -> CellStatus:
        status = self._cells.get(key)
        if status is None:
            status = CellStatus(
                state=raw_state,
                since_frame=frame_idx,
                person_id=person_id,
                objects=objects,
                seen_frame=frame_idx,
            )
            self._cells[key] = status
            return status

        status.person_id = person_id
        status.objects = objects
        status.seen_frame = frame_idx

        if raw_state == status.state:
            # A pending change loses ground on every frame that disagrees with
            # it, but a single dropout does not wipe the evidence entirely.
            if status.pending is not None:
                status.streak -= 1
                if status.streak <= 0:
                    status.pending = None
                    status.streak = 0
            else:
                status.streak = 0
            return status

        if status.pending is None:
            status.pending = raw_state
            status.streak = 1
        elif status.pending == raw_state:
            status.streak += 1
        else:
            status.streak -= 1
            if status.streak <= 0:
                status.pending = raw_state
                status.streak = 1

        if raw_state is SeatState.EMPTY:
            needed = self.cfg.release_frames
        elif raw_state is SeatState.GHOST:
            needed = self.cfg.ghost_confirm_frames
        else:
            needed = self.cfg.confirm_frames

        if status.streak >= needed:
            status.state = raw_state
            status.pending = None
            status.streak = 0
            status.since_frame = frame_idx
        return status

    def get(self, key: tuple[int, int]) -> CellStatus | None:
        return self._cells.get(key)

    def active_keys(self) -> list[tuple[int, int]]:
        return list(self._cells)

    def prune(self, live_seat_ids: set[int], frame_idx: int, max_age: int = 120) -> None:
        stale = [
            key
            for key, status in self._cells.items()
            if key[0] not in live_seat_ids
            and frame_idx - status.seen_frame > max_age
        ]
        for key in stale:
            del self._cells[key]


def raw_state(person_present: bool, objects_present: bool) -> SeatState:
    if person_present:
        return SeatState.OCCUPIED
    if objects_present:
        return SeatState.GHOST
    return SeatState.EMPTY
