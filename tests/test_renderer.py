import unittest

import numpy as np

from seatfinder.config import RenderConfig
from seatfinder.detector import Detection
from seatfinder.renderer import HudInfo, Renderer
from seatfinder.state import CellStatus, SeatState
from seatfinder.tracking import Seat


def make_seat(seat_id: int, box, capacity: int = 1, synthetic: bool = False) -> Seat:
    return Seat(
        seat_id=seat_id,
        kind="chair",
        box=box,
        capacity=capacity,
        first_frame=0,
        last_frame=0,
        confidence=0.9,
        hits=5,
        synthetic=synthetic,
    )


def scene(width: int, height: int):
    seats = [
        make_seat(1, (width * 0.1, height * 0.4, width * 0.3, height * 0.9)),
        make_seat(2, (width * 0.35, height * 0.4, width * 0.6, height * 0.9),
                  capacity=2),
    ]
    statuses = {
        (1, 0): CellStatus(state=SeatState.EMPTY),
        (2, 0): CellStatus(state=SeatState.OCCUPIED, person_id=7),
        (2, 1): CellStatus(state=SeatState.GHOST, objects=("backpack",)),
    }
    persons = [Detection(0, "person", 0.9,
                         (width * 0.36, height * 0.2, width * 0.5, height * 0.9),
                         track_id=7)]
    belongings = [Detection(24, "backpack", 0.6,
                            (width * 0.5, height * 0.6, width * 0.58, height * 0.8))]
    hud = HudInfo(frame_index=10, time_s=0.33, empty=4, occupied=2, ghost=1,
                  fps=21.5, total_frames=40)
    return seats, statuses, persons, belongings, hud


class RendererTests(unittest.TestCase):
    def _draw(self, width: int, height: int) -> np.ndarray:
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        seats, statuses, persons, belongings, hud = scene(width, height)
        return Renderer(RenderConfig()).draw(
            frame, seats, statuses, persons, belongings, hud
        )

    def test_draws_without_changing_shape(self):
        for width, height in ((848, 478), (1920, 1080)):
            with self.subTest(size=(width, height)):
                out = self._draw(width, height)
                self.assertEqual(out.shape, (height, width, 3))

    def test_draw_is_deterministic_across_calls(self):
        renderer = Renderer(RenderConfig())
        frame = np.full((478, 848, 3), 80, dtype=np.uint8)
        seats, statuses, persons, belongings, hud = scene(848, 478)
        first = renderer.draw(frame.copy(), seats, statuses, persons, belongings, hud)
        second = renderer.draw(frame.copy(), seats, statuses, persons, belongings, hud)
        self.assertTrue(np.array_equal(first, second))

    def test_labels_do_not_accumulate_between_frames(self):
        renderer = Renderer(RenderConfig())
        frame = np.full((478, 848, 3), 80, dtype=np.uint8)
        seats, statuses, persons, belongings, hud = scene(848, 478)
        renderer.draw(frame.copy(), seats, statuses, persons, belongings, hud)
        first = len(renderer._labels)
        renderer.draw(frame.copy(), seats, statuses, persons, belongings, hud)
        self.assertGreater(first, 0)
        self.assertEqual(len(renderer._labels), first)


if __name__ == "__main__":
    unittest.main()
