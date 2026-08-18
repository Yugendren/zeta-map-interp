import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from zetamap.dyck import (
    all_dyck_paths,
    area,
    area_word,
    bounce,
    dinv,
    zeta,
    zeta_inverse,
)

CATALAN = {
    1: 1,
    2: 2,
    3: 5,
    4: 14,
    5: 42,
    6: 132,
    7: 429,
    8: 1430,
    9: 4862,
    10: 16796,
}


class TestCatalanCounts(unittest.TestCase):
    def test_catalan_counts(self):
        for n, c in CATALAN.items():
            with self.subTest(n=n):
                self.assertEqual(len(list(all_dyck_paths(n))), c)


class TestHandVerifiedAnchors(unittest.TestCase):
    def test_area_anchors(self):
        self.assertEqual(area(('N', 'N', 'E', 'E')), 1)
        self.assertEqual(area_word(('N', 'N', 'E', 'E')), (0, 1))
        self.assertEqual(area(('N', 'E', 'N', 'E')), 0)
        self.assertEqual(area_word(('N', 'E', 'N', 'E')), (0, 0))

    def test_area_multiset_n3(self):
        areas = sorted(area(d) for d in all_dyck_paths(3))
        self.assertEqual(areas, [0, 1, 1, 2, 3])

    def test_dinv_anchors(self):
        self.assertEqual(dinv(('N', 'E', 'N', 'E', 'N', 'E')), 3)
        self.assertEqual(dinv(('N', 'N', 'N', 'E', 'E', 'E')), 0)

    def test_dinv_multiset_n3(self):
        dinvs = sorted(dinv(d) for d in all_dyck_paths(3))
        self.assertEqual(dinvs, [0, 1, 1, 2, 3])

    def test_bounce_anchors(self):
        self.assertEqual(bounce(('N', 'E', 'N', 'E', 'N', 'E')), 3)
        self.assertEqual(bounce(('N', 'N', 'N', 'E', 'E', 'E')), 0)

    def test_bounce_multiset_n3(self):
        bounces = sorted(bounce(d) for d in all_dyck_paths(3))
        self.assertEqual(bounces, [0, 1, 1, 2, 3])


class TestEquidistribution(unittest.TestCase):
    def test_area_dinv_bounce_equidistributed(self):
        for n in range(1, 9):
            with self.subTest(n=n):
                paths = list(all_dyck_paths(n))
                area_multiset = Counter(area(d) for d in paths)
                dinv_multiset = Counter(dinv(d) for d in paths)
                bounce_multiset = Counter(bounce(d) for d in paths)
                self.assertEqual(area_multiset, dinv_multiset)
                self.assertEqual(area_multiset, bounce_multiset)


class TestZetaProperties(unittest.TestCase):
    def test_p1_p2_p3_exhaustive(self):
        for n in range(1, 10):
            with self.subTest(n=n):
                paths = list(all_dyck_paths(n))
                pathset = set(paths)
                images = set()
                for d in paths:
                    zd = zeta(d)
                    # zeta image is a valid Dyck path (part of P1)
                    self.assertIn(zd, pathset)
                    images.add(zd)
                    # P2
                    self.assertEqual(area(zd), dinv(d))
                    # P3
                    self.assertEqual(bounce(zd), area(d))
                # P1: bijection (injective + image covers full Dyck_n,
                # domain/codomain have equal finite size)
                self.assertEqual(images, pathset)


class TestZetaInverse(unittest.TestCase):
    def test_zeta_inverse_roundtrip(self):
        for n in range(1, 9):
            with self.subTest(n=n):
                for d in all_dyck_paths(n):
                    self.assertEqual(zeta_inverse(zeta(d)), d)


if __name__ == '__main__':
    unittest.main()
