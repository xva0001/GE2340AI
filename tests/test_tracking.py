import unittest

from seatfinder.config import SeatConfig
from seatfinder.detector import Detection
from seatfinder.geometry import area
from seatfinder.tracking import (
    SYNTHETIC_SEAT_OFFSET,
    GlobalMotion,
    PersonHistory,
    SeatTracker,
    estimate_capacity,
)


def chair(box, conf=0.9, label="chair"):
    return Detection(56, label, conf, box)


class CapacityTests(unittest.TestCase):
    def test_chair_is_single_seat(self):
        self.assertEqual(estimate_capacity("chair", (0, 0, 80, 120), [40], 0.55), 1)

    def test_couch_splits_by_person_width(self):
        capacity = estimate_capacity("couch", (0, 0, 300, 120), [50], 0.55)
        self.assertGreaterEqual(capacity, 2)

    def test_couch_without_people_has_default(self):
        self.assertEqual(estimate_capacity("couch", (0, 0, 300, 120), [], 0.55), 2)

    def test_capacity_never_creates_tiny_cells(self):
        capacity = estimate_capacity("bench", (0, 0, 50, 100), [10], 0.55)
        self.assertLessEqual(capacity, 2)


class SeatTrackerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SeatConfig()
        self.tracker = SeatTracker(self.cfg)

    def _confirm(self, box, frames=3):
        seats = []
        for frame in range(frames):
            seats = self.tracker.update([chair(box)], frame, [])
        return seats

    def test_seat_is_not_reported_before_confirmation(self):
        seats = self.tracker.update([chair((10, 10, 100, 200))], 0, [])
        self.assertEqual(seats, [])

    def test_seat_reported_after_confirmation(self):
        seats = self._confirm((10, 10, 100, 200))
        self.assertEqual(len(seats), 1)

    def test_identity_is_stable(self):
        self._confirm((10, 10, 100, 200))
        seats = self.tracker.update([chair((14, 12, 104, 202))], 5, [])
        self.assertEqual(len(seats), 1)
        self.assertEqual(seats[0].seat_id, 1)

    def test_duplicate_boxes_collapse(self):
        seats = self._confirm((10, 10, 100, 200))
        self.tracker.update(
            [chair((10, 10, 100, 200)), chair((12, 12, 102, 202))], 5, []
        )
        self.assertEqual(len(seats), 1)

    def test_seat_expires_after_ttl(self):
        self._confirm((10, 10, 100, 200))
        seats = self.tracker.update([], 10_000, [])
        self.assertEqual(seats, [])

    def test_synthetic_seat_registered(self):
        seat = self.tracker.upsert_synthetic(42, (5, 5, 60, 200), 1, 0.8)
        self.assertEqual(seat.seat_id, SYNTHETIC_SEAT_OFFSET + 42)
        self.assertIn(seat, self.tracker.all_seats())
        self.tracker.prune_synthetic(set(), 100)
        self.assertEqual(self.tracker.all_seats(), [])


class MotionTests(unittest.TestCase):
    def test_person_history_detects_stillness(self):
        history = PersonHistory()
        still = Detection(0, "person", 0.9, (100, 100, 160, 300), track_id=1)
        for frame in range(10):
            history.update(frame, [still], (0.0, 0.0))
        self.assertEqual(history.stillness(1), 0.0)
        self.assertEqual(history.samples(1), 10)

    def test_moving_person_is_not_still(self):
        history = PersonHistory()
        for offset in range(10):
            moving = Detection(
                0, "person", 0.9, (100 + offset * 20, 100, 160 + offset * 20, 300),
                track_id=1,
            )
            history.update(offset, [moving], (0.0, 0.0))
        self.assertGreater(history.stillness(1), 20)

    def test_global_motion_tracks_camera_pan(self):
        motion = GlobalMotion()
        first = [
            Detection(0, "person", 0.9, (100 + 200 * i, 100, 160 + 200 * i, 300),
                      track_id=i + 1)
            for i in range(3)
        ]
        second = [
            Detection(0, "person", 0.9, (80 + 200 * i, 100, 140 + 200 * i, 300),
                      track_id=i + 1)
            for i in range(3)
        ]
        motion.update(first, 1000)
        motion.update(second, 1000)
        self.assertAlmostEqual(motion.offset[0], -20.0)

    def test_single_person_is_not_treated_as_camera_motion(self):
        motion = GlobalMotion()
        a = Detection(0, "person", 0.9, (100, 100, 160, 300), track_id=1)
        b = Detection(0, "person", 0.9, (80, 100, 140, 300), track_id=1)
        motion.update([a], 1000)
        motion.update([b], 1000)
        self.assertEqual(motion.offset, (0.0, 0.0))

    def test_compensation_keeps_panning_people_still(self):
        motion = GlobalMotion()
        history = PersonHistory()
        for frame in range(10):
            shift = -5 * frame
            people = [
                Detection(0, "person", 0.9,
                          (100 + shift + 200 * i, 100, 160 + shift + 200 * i, 300),
                          track_id=i + 1)
                for i in range(3)
            ]
            motion.update(people, 1000)
            history.update(frame, people, motion.offset)
        self.assertLess(history.stillness(1), 1e-6)


if __name__ == "__main__":
    unittest.main()
