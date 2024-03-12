import unittest

import numpy as np

from fosco.common import domains


class TestDomains(unittest.TestCase):
    def test_rectangle(self):
        X = domains.Rectangle(vars=["x", "y"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        self.assertEqual(X.dimension, 2)
        self.assertEqual(X.vars, ["x", "y"])
        self.assertEqual(X.lower_bounds, (-5.0, -5.0))
        self.assertEqual(X.upper_bounds, (5.0, 5.0))

        data = X.generate_data(1000)
        self.assertEqual(data.shape, (1000, 2))

        for sample in data:
            x, y = sample
            self.assertGreaterEqual(x, -5.0)
            self.assertLessEqual(x, 5.0)
            self.assertGreaterEqual(y, -5.0)
            self.assertLessEqual(y, 5.0)
            self.assertTrue(
                X.check_containment(sample[None]),
                f"check containement failed for sample = {sample}",
            )

    def test_sphere(self):
        X = domains.Sphere(vars=["x", "y", "z"], center=(0.0, 0.0, 0.0), radius=5.0)
        self.assertEqual(X.dimension, 3)

        data = X.generate_data(1000)
        self.assertEqual(data.shape, (1000, 3))

        for sample in data:
            x, y, z = sample
            self.assertLessEqual(x**2 + y**2 + z**2, 5.0**2)
            self.assertTrue(
                X.check_containment(sample[None]),
                f"check containement failed for sample = {sample}",
            )

    def test_union(self):
        X1 = domains.Rectangle(vars=["x", "y"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        X2 = domains.Sphere(vars=["x", "y"], center=(10.0, 10.0), radius=1.0)

        X = domains.Union(sets=[X1, X2])
        self.assertEqual(X.dimension, 2)
        self.assertEqual(X.vars, ["x", "y"])

        data = X.generate_data(1000)
        self.assertEqual(data.shape, (1000, 2))

        for sample in data:
            is_in_x1 = X1.check_containment(sample[None])
            is_in_x2 = X2.check_containment(sample[None])
            self.assertTrue(
                is_in_x1 or is_in_x2, f"check containement failed for sample = {sample}"
            )

    def test_intersection(self):
        X1 = domains.Rectangle(vars=["x", "y"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        X2 = domains.Sphere(vars=["x", "y"], center=(1.0, 1.0), radius=1.0)

        X = domains.Intersection(sets=[X1, X2])
        self.assertEqual(X.dimension, 2)
        self.assertEqual(X.vars, ["x", "y"])

        data = X.generate_data(1000)
        self.assertEqual(data.shape, (1000, 2))

        for sample in data:
            is_in_x1 = X1.check_containment(sample[None])
            is_in_x2 = X2.check_containment(sample[None])
            self.assertTrue(
                is_in_x1 and is_in_x2,
                f"check containement failed for sample = {sample}",
            )
