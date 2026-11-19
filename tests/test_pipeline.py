import unittest

from seatfinder.pipeline import _count
from seatfinder.state import CellStatus, SeatState


class CountTests(unittest.TestCase):
    def test_all_three_states_are_counted(self):
        statuses = {
            (1, 0): CellStatus(state=SeatState.EMPTY),
            (1, 1): CellStatus(state=SeatState.OCCUPIED),
            (2, 0): CellStatus(state=SeatState.GHOST),
        }
        counts = _count(statuses)
        self.assertEqual(
            counts, {"seats": 3, "empty": 1, "occupied": 1, "objects": 1}
        )

    def test_ghost_only(self):
        counts = _count({(5, 0): CellStatus(state=SeatState.GHOST)})
        self.assertEqual(counts["objects"], 1)
        self.assertEqual(counts["seats"], 1)

    def test_empty_input(self):
        counts = _count({})
        self.assertEqual(counts["seats"], 0)


if __name__ == "__main__":
    unittest.main()
