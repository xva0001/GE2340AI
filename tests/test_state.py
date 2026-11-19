import unittest

from seatfinder.config import TemporalConfig
from seatfinder.state import SeatState, SeatStateMachine, raw_state


class RawStateTests(unittest.TestCase):
    def test_person_wins_over_objects(self):
        self.assertIs(raw_state(True, True), SeatState.OCCUPIED)

    def test_objects_only_is_ghost(self):
        self.assertIs(raw_state(False, True), SeatState.GHOST)

    def test_nothing_is_empty(self):
        self.assertIs(raw_state(False, False), SeatState.EMPTY)


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = SeatStateMachine(
            TemporalConfig(
                confirm_frames=2, ghost_confirm_frames=2, release_frames=2
            )
        )

    def test_new_cell_takes_raw_state_immediately(self):
        status = self.machine.update((1, 0), SeatState.EMPTY, 0)
        self.assertIs(status.state, SeatState.EMPTY)

    def test_occupied_needs_confirmation(self):
        self.machine.update((1, 0), SeatState.EMPTY, 0)
        self.machine.update((1, 0), SeatState.OCCUPIED, 1)
        status = self.machine.update((1, 0), SeatState.OCCUPIED, 2)
        self.assertIs(status.state, SeatState.OCCUPIED)

    def test_single_noisy_frame_is_ignored(self):
        self.machine.update((1, 0), SeatState.EMPTY, 0)
        self.machine.update((1, 0), SeatState.OCCUPIED, 1)
        status = self.machine.update((1, 0), SeatState.EMPTY, 2)
        self.assertIs(status.state, SeatState.EMPTY)

    def test_leaving_occupied_uses_release_frames(self):
        self.machine.update((1, 0), SeatState.OCCUPIED, 0)
        self.machine.update((1, 0), SeatState.EMPTY, 1)
        status = self.machine.update((1, 0), SeatState.EMPTY, 2)
        self.assertIs(status.state, SeatState.EMPTY)
        self.assertEqual(status.since_frame, 2)

    def test_ghost_requires_confirmation(self):
        self.machine.update((2, 0), SeatState.EMPTY, 0)
        status = self.machine.update((2, 0), SeatState.GHOST, 1)
        self.assertIs(status.state, SeatState.EMPTY)
        status = self.machine.update((2, 0), SeatState.GHOST, 2)
        self.assertIs(status.state, SeatState.GHOST)

    def test_prune_drops_dead_seats(self):
        self.machine.update((9, 0), SeatState.EMPTY, 10)
        self.machine.prune(set(), 500)
        self.assertIsNone(self.machine.get((9, 0)))

    def test_single_frame_dropout_does_not_reset_progress(self):
        cfg = TemporalConfig(
            confirm_frames=10, ghost_confirm_frames=3, release_frames=10
        )
        machine = SeatStateMachine(cfg)
        machine.update((3, 0), SeatState.EMPTY, 0)
        machine.update((3, 0), SeatState.GHOST, 1)
        machine.update((3, 0), SeatState.GHOST, 2)
        machine.update((3, 0), SeatState.EMPTY, 3)   # one-frame detection gap
        machine.update((3, 0), SeatState.GHOST, 4)
        status = machine.update((3, 0), SeatState.GHOST, 5)
        self.assertIs(status.state, SeatState.GHOST)

    def test_ghost_confirms_faster_than_occupied(self):
        cfg = TemporalConfig(
            confirm_frames=10, ghost_confirm_frames=3, release_frames=10
        )
        machine = SeatStateMachine(cfg)
        machine.update((4, 0), SeatState.EMPTY, 0)
        for frame in range(1, 4):
            status = machine.update((4, 0), SeatState.GHOST, frame)
        self.assertIs(status.state, SeatState.GHOST)


if __name__ == "__main__":
    unittest.main()
