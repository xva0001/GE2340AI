import unittest

from seatfinder.association import associate, is_carried, person_seat_score
from seatfinder.config import SeatConfig
from seatfinder.detector import Detection
from seatfinder.tracking import Seat


def person(box, track_id=7):
    return Detection(0, "person", 0.9, box, track_id)


def item(label, box, class_id=24):
    return Detection(class_id, label, 0.5, box)


def seat(box=(100, 100, 200, 300), capacity=1, kind="chair"):
    return Seat(
        seat_id=1,
        kind=kind,
        box=box,
        capacity=capacity,
        first_frame=0,
        last_frame=0,
        confidence=0.9,
        hits=5,
    )


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SeatConfig()

    def test_seated_person_scores_above_threshold(self):
        chair = (100, 100, 200, 300)
        sitting = (110, 60, 190, 290)
        score = person_seat_score(sitting, chair, self.cfg)
        self.assertGreaterEqual(score, self.cfg.person_min_score)

    def test_far_person_does_not_match(self):
        result = associate([seat()], [person((600, 600, 700, 900))], [], self.cfg)
        observation = result.cells[(1, 0)]
        self.assertIsNone(observation.person_id)
        self.assertEqual(len(result.unmatched_persons), 1)

    def test_person_is_assigned_to_seat(self):
        result = associate([seat()], [person((110, 60, 190, 290))], [], self.cfg)
        self.assertEqual(result.cells[(1, 0)].person_id, 7)
        self.assertEqual(result.unmatched_persons, [])

    def test_object_on_seat_makes_ghost_candidate(self):
        bag = item("backpack", (120, 200, 170, 280))
        result = associate([seat()], [], [bag], self.cfg)
        self.assertEqual(result.cells[(1, 0)].objects, ["backpack"])

    def test_sign_at_top_of_chair_is_ignored(self):
        sign = item("book", (120, 105, 170, 150), class_id=73)
        result = associate([seat()], [], [sign], self.cfg)
        self.assertEqual(result.cells[(1, 0)].objects, [])

    def test_object_held_by_person_is_not_left_behind(self):
        bag = item("backpack", (120, 200, 170, 280))
        holder = person((100, 150, 210, 320))
        self.assertTrue(is_carried(bag, [holder], self.cfg))
        result = associate([seat()], [holder], [bag], self.cfg)
        self.assertEqual(result.cells[(1, 0)].objects, [])

    def test_oversized_detection_is_rejected(self):
        huge = item("umbrella", (100, 100, 200, 300))
        result = associate([seat()], [], [huge], self.cfg)
        self.assertEqual(result.cells[(1, 0)].objects, [])

    def test_object_on_a_table_is_not_a_seat_object(self):
        laptop = item("laptop", (120, 200, 170, 280), class_id=63)
        table = (100, 190, 240, 300)
        result = associate([seat()], [], [laptop], self.cfg, [table])
        self.assertEqual(result.cells[(1, 0)].objects, [])

    def test_object_off_the_table_still_counts(self):
        bag = item("backpack", (120, 200, 170, 280))
        table = (600, 600, 800, 700)
        result = associate([seat()], [], [bag], self.cfg, [table])
        self.assertEqual(result.cells[(1, 0)].objects, ["backpack"])

    def test_couch_cells_split_capacity(self):
        couch = seat(box=(0, 100, 300, 300), capacity=3, kind="couch")
        left = person((10, 50, 90, 290))
        right = person((210, 50, 290, 290))
        result = associate([couch], [left, right], [], self.cfg)
        self.assertEqual(result.cells[(1, 0)].person_id, 7)
        self.assertEqual(result.cells[(1, 2)].person_id, 7)
        self.assertIsNone(result.cells[(1, 1)].person_id)

    def test_one_person_per_cell(self):
        people = [person((110, 60, 190, 290), track_id=1),
                  person((115, 65, 195, 295), track_id=2)]
        result = associate([seat()], people, [], self.cfg)
        assigned = [o.person_id for o in result.cells.values() if o.person_id]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(len(result.unmatched_persons), 1)


if __name__ == "__main__":
    unittest.main()
