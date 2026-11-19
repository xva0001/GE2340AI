import unittest

from seatfinder.geometry import (
    area,
    bottom_center,
    center,
    clamp,
    containment,
    expand,
    intersection_area,
    iou,
    split_horizontally,
)


class GeometryTests(unittest.TestCase):
    def test_area_and_center(self):
        box = (0, 0, 100, 50)
        self.assertEqual(area(box), 5000)
        self.assertEqual(center(box), (50, 25))
        self.assertEqual(bottom_center(box), (50, 50))

    def test_iou_identical_and_disjoint(self):
        box = (10, 10, 110, 110)
        self.assertAlmostEqual(iou(box, box), 1.0)
        self.assertEqual(iou(box, (200, 200, 300, 300)), 0.0)

    def test_iou_half_overlap(self):
        a = (0, 0, 10, 10)
        b = (5, 0, 15, 10)
        self.assertAlmostEqual(iou(a, b), 50 / 150)

    def test_intersection_area_none(self):
        self.assertEqual(intersection_area((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_containment(self):
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        self.assertAlmostEqual(containment(inner, outer), 1.0)
        self.assertAlmostEqual(containment(outer, inner), 0.25)

    def test_expand(self):
        self.assertEqual(expand((10, 10, 20, 20), 0.5), (5, 5, 25, 25))

    def test_split_horizontally(self):
        parts = split_horizontally((0, 0, 90, 10), 3)
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], (0, 0, 30, 10))
        self.assertEqual(parts[-1], (60, 0, 90, 10))

    def test_split_single_part_is_identity(self):
        box = (1, 2, 3, 4)
        self.assertEqual(split_horizontally(box, 1), [box])

    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(50, 0, 10), 10)


if __name__ == "__main__":
    unittest.main()
