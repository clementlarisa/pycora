"""Tests for pycora.zonotope.

Each test uses a known analytic result that can be verified by hand or by
comparing to the corresponding CORA MATLAB output.
"""
import numpy as np
import pytest

from pycora.zonotope import Zonotope, reduce_girard


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_construct_from_box():
    Z = Zonotope.from_box([0, 0], [2, 4])
    np.testing.assert_allclose(Z.c, [1, 2])
    np.testing.assert_allclose(Z.G, np.diag([1, 2]))
    assert Z.n == 2
    assert Z.num_generators == 2


def test_construct_from_center_radii():
    Z = Zonotope.from_center_radii([5, -3], [0.5, 1.0])
    np.testing.assert_allclose(Z.c, [5, -3])
    np.testing.assert_allclose(Z.G, np.diag([0.5, 1.0]))


def test_construct_point():
    Z = Zonotope.point([1, 2, 3])
    assert Z.num_generators == 0
    np.testing.assert_allclose(Z.c, [1, 2, 3])


def test_construct_explicit():
    c = np.array([1, 1])
    G = np.array([[1, 1, 1], [1, -1, 0]], dtype=float)
    Z = Zonotope(c, G)
    np.testing.assert_allclose(Z.c, c)
    np.testing.assert_allclose(Z.G, G)


# ---------------------------------------------------------------------------
# Minkowski sum (plus)
# ---------------------------------------------------------------------------
def test_plus_two_boxes_is_a_box():
    """Minkowski sum of two boxes = box with summed widths."""
    A = Zonotope.from_box([-1, -1], [1, 1])  # 2x2 box at origin
    B = Zonotope.from_box([-2, -3], [2, 3])  # 4x6 box at origin
    S = A.plus(B)
    lb, ub = S.interval()
    # Sum should be a box of width 6 x 8 (2 + 4 in x, 2 + 6 in y)
    np.testing.assert_allclose(lb, [-3, -4])
    np.testing.assert_allclose(ub, [3, 4])


def test_plus_with_vector_shifts_center():
    Z = Zonotope.from_box([0, 0], [2, 2])
    Z2 = Z.plus([10, -5])
    np.testing.assert_allclose(Z2.c, [11, -4])
    np.testing.assert_allclose(Z2.G, Z.G)


def test_plus_concatenates_generators():
    Z = Zonotope([0, 0], [[1], [0]])
    other = Zonotope([0, 0], [[0], [1]])
    S = Z.plus(other)
    assert S.num_generators == 2


# ---------------------------------------------------------------------------
# Linear map (mtimes)
# ---------------------------------------------------------------------------
def test_linear_map_identity():
    Z = Zonotope.from_box([1, 2], [3, 4])
    Z2 = Z.linear_map(np.eye(2))
    np.testing.assert_allclose(Z2.c, Z.c)
    np.testing.assert_allclose(Z2.G, Z.G)


def test_linear_map_rotation():
    """90-degree rotation of a box around origin."""
    Z = Zonotope.point([1, 0])
    R = np.array([[0, -1], [1, 0]])
    Z2 = Z.linear_map(R)
    np.testing.assert_allclose(Z2.c, [0, 1], atol=1e-12)


def test_linear_map_with_generators():
    Z = Zonotope([0, 0], [[1, 0], [0, 1]])
    A = np.array([[2, 0], [0, 3]])
    Z2 = Z.linear_map(A)
    np.testing.assert_allclose(Z2.c, [0, 0])
    np.testing.assert_allclose(Z2.G, [[2, 0], [0, 3]])


# ---------------------------------------------------------------------------
# Cartesian product
# ---------------------------------------------------------------------------
def test_cart_product_doubles_dim():
    A = Zonotope.from_box([0], [2])
    B = Zonotope.from_box([10], [12])
    S = A.cart_product(B)
    assert S.n == 2
    np.testing.assert_allclose(S.c, [1, 11])
    np.testing.assert_allclose(S.G, [[1, 0], [0, 1]])


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
def test_project_drops_dims():
    Z = Zonotope.from_box([1, 2, 3], [4, 5, 6])
    Z2 = Z.project([0, 2])
    assert Z2.n == 2
    np.testing.assert_allclose(Z2.c, [2.5, 4.5])


# ---------------------------------------------------------------------------
# Interval / bounds
# ---------------------------------------------------------------------------
def test_interval_of_box_is_box():
    Z = Zonotope.from_box([-2, -3], [4, 5])
    lb, ub = Z.interval()
    np.testing.assert_allclose(lb, [-2, -3])
    np.testing.assert_allclose(ub, [4, 5])


def test_interval_with_skew_generators():
    """Z = {(beta1 + beta2, beta1 - beta2) | beta in [-1,1]^2}.

    Bounding box is [-2, 2] x [-2, 2].
    """
    Z = Zonotope([0, 0], [[1, 1], [1, -1]])
    lb, ub = Z.interval()
    np.testing.assert_allclose(lb, [-2, -2])
    np.testing.assert_allclose(ub, [2, 2])


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------
def test_contains_point_inside():
    Z = Zonotope.from_box([0, 0], [10, 10])
    assert Z.contains_point([5, 5])
    assert Z.contains_point([0, 0])
    assert Z.contains_point([10, 10])


def test_contains_point_outside():
    Z = Zonotope.from_box([0, 0], [10, 10])
    assert not Z.contains_point([11, 5])
    assert not Z.contains_point([-0.1, 5])


def test_contains_point_skew():
    """Diamond-shaped zonotope { (a+b, a-b) | a,b in [-1,1] }."""
    Z = Zonotope([0, 0], [[1, 1], [1, -1]])
    assert Z.contains_point([0, 0])
    assert Z.contains_point([2, 0])
    assert Z.contains_point([0, 2])
    assert not Z.contains_point([2.1, 0])


# ---------------------------------------------------------------------------
# Box overlap
# ---------------------------------------------------------------------------
def test_overlaps_box_disjoint():
    Z = Zonotope.from_box([0, 0], [1, 1])
    assert not Z.overlaps_box([5, 5], [10, 10])


def test_overlaps_box_intersecting():
    Z = Zonotope.from_box([0, 0], [10, 10])
    assert Z.overlaps_box([5, 5], [15, 15])


def test_overlaps_box_contained():
    Z = Zonotope.from_box([-5, -5], [5, 5])
    assert Z.overlaps_box([-1, -1], [1, 1])


# ---------------------------------------------------------------------------
# Enclose / convex hull
# ---------------------------------------------------------------------------
def test_enclose_two_singletons():
    """Enclose of two points = segment between them, contained in result."""
    A = Zonotope.point([0, 0])
    B = Zonotope.point([4, 0])
    H = A.enclose(B)
    # Both endpoints + midpoint must be in H
    assert H.contains_point([0, 0])
    assert H.contains_point([4, 0])
    assert H.contains_point([2, 0])


def test_enclose_two_boxes_contains_both():
    A = Zonotope.from_box([-1, -1], [1, 1])
    B = Zonotope.from_box([5, 5], [7, 7])
    H = A.enclose(B)
    # Sample points from both A and B should lie in H
    for p in [[0, 0], [-1, -1], [1, 1], [6, 6], [5, 5], [7, 7]]:
        assert H.contains_point(p), f"point {p} should be in enclosure"


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
def test_reduce_below_order_is_noop():
    Z = Zonotope.from_box([-1, -1], [1, 1])  # order = 1
    R = reduce_girard(Z, order=50)
    np.testing.assert_allclose(R.c, Z.c)
    np.testing.assert_allclose(R.G, Z.G)


def test_reduce_keeps_overapproximation():
    """Reduced zonotope must contain the original."""
    rng = np.random.default_rng(42)
    n, p = 4, 80  # 4D with 80 generators (order = 20)
    Z = Zonotope(rng.standard_normal(n), rng.standard_normal((n, p)))
    R = reduce_girard(Z, order=2)  # keep at most 2 * 4 = 8 generators
    assert R.num_generators <= 2 * n + n  # Girard adds up to n box generators
    # check 50 random points from Z lie in R
    for _ in range(50):
        beta = rng.uniform(-1, 1, size=p)
        pt = Z.c + Z.G @ beta
        assert R.contains_point(pt, tol=1e-6), \
            f"reduced zonotope must contain original points"
