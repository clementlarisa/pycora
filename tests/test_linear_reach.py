"""Tests for pycora.linear_sys.LinearSys against analytic results."""
import numpy as np
import pytest

from pycora.linear_sys import LinearSys
from pycora.zonotope import Zonotope


# ---------------------------------------------------------------------------
# Double integrator: ẋ₁ = x₂,  ẋ₂ = u
# ---------------------------------------------------------------------------
@pytest.fixture
def di_sys():
    """Standard double integrator system."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    return LinearSys(A, B)


def test_di_homogeneous_zero_input(di_sys):
    """With no input, ẋ₁ = x₂ → x₁(t) = x₁(0) + t·x₂(0); x₂(t) = x₂(0)."""
    # Initial: x₁ = 1, x₂ = 0  (so we should stay at x₁ = 1)
    X = Zonotope.point([1.0, 0.0])
    Htp, Hti = di_sys.homogeneous_solution(X, dt=0.5)
    # Should still be (1, 0)
    np.testing.assert_allclose(Htp.c, [1.0, 0.0], atol=1e-10)


def test_di_homogeneous_velocity(di_sys):
    """x₁(0) = 0, x₂(0) = 2 → x₁(0.5) = 0 + 0.5·2 = 1, x₂ unchanged."""
    X = Zonotope.point([0.0, 2.0])
    Htp, Hti = di_sys.homogeneous_solution(X, dt=0.5)
    np.testing.assert_allclose(Htp.c, [1.0, 2.0], atol=1e-10)


def test_di_constant_input(di_sys):
    """x₁(0)=x₂(0)=0, u=1 const → x₁(0.5) = ½·1·0.25 = 0.125, x₂(0.5) = 0.5."""
    X = Zonotope.point([0.0, 0.0])
    Rtp, Rti = di_sys.one_step(X, u_const=np.array([1.0]), dt=0.5)
    np.testing.assert_allclose(Rtp.c, [0.125, 0.5], atol=1e-10)


def test_di_full_motion(di_sys):
    """x₁(0) = 0, x₂(0) = 1, u = 0.5 const, dt = 1.0.

    Analytic: x₁(1) = 0 + 1·1 + ½·0.5·1² = 1.25
              x₂(1) = 1 + 0.5·1 = 1.5
    """
    X = Zonotope.point([0.0, 1.0])
    Rtp, _ = di_sys.one_step(X, u_const=np.array([0.5]), dt=1.0)
    np.testing.assert_allclose(Rtp.c, [1.25, 1.5], atol=1e-10)


def test_di_initial_uncertainty_propagates(di_sys):
    """If x₂(0) ∈ [0.9, 1.1], then x₁(dt) ∈ [0.9·dt, 1.1·dt] (up to widening)."""
    X = Zonotope.from_box([0.0, 0.9], [0.0, 1.1])  # x₁ exact, x₂ uncertain
    dt = 0.5
    Rtp, _ = di_sys.one_step(X, u_const=np.array([0.0]), dt=dt)
    lb, ub = Rtp.interval()
    # x₁ should be in [0.45, 0.55], x₂ unchanged in [0.9, 1.1]
    assert 0.44 <= lb[0] <= 0.46, f"x₁ lower bound {lb[0]} not ≈ 0.45"
    assert 0.54 <= ub[0] <= 0.56, f"x₁ upper bound {ub[0]} not ≈ 0.55"
    assert 0.89 <= lb[1] <= 0.91
    assert 1.09 <= ub[1] <= 1.11


def test_di_time_interval_contains_endpoints(di_sys):
    """Time-interval reach Rti must contain both X(0) and Rtp."""
    X = Zonotope.from_center_radii([0.5, 1.0], [0.05, 0.05])
    Rtp, Rti = di_sys.one_step(X, u_const=np.array([0.0]), dt=0.5)
    # Both X.c and Rtp.c lie in Rti
    assert Rti.contains_point(X.c)
    assert Rti.contains_point(Rtp.c, tol=1e-6)


# ---------------------------------------------------------------------------
# Stable spiral: A = [[-1, -4], [4, -1]], known matrix exponential
# ---------------------------------------------------------------------------
def test_spiral_homogeneous_matches_expm():
    """exp(A·dt) · x₀ should match scipy.expm result exactly."""
    from scipy.linalg import expm

    A = np.array([[-1.0, -4.0], [4.0, -1.0]])
    sys = LinearSys(A)
    X = Zonotope.point([1.0, 1.0])
    Htp, _ = sys.homogeneous_solution(X, dt=0.05)
    expected = expm(A * 0.05) @ np.array([1.0, 1.0])
    np.testing.assert_allclose(Htp.c, expected, atol=1e-12)


def test_spiral_time_interval_contains_intermediate():
    """Sample x(τ) for τ in [0, dt] using analytic exp(A·τ)·x₀, all should lie in Hti."""
    from scipy.linalg import expm

    A = np.array([[-1.0, -4.0], [4.0, -1.0]])
    sys = LinearSys(A)
    X = Zonotope.from_center_radii([1.0, 1.0], [0.01, 0.01])
    dt = 0.1
    _, Hti = sys.homogeneous_solution(X, dt)

    for tau in np.linspace(0.0, dt, 11):
        x_tau = expm(A * tau) @ X.c
        assert Hti.contains_point(x_tau, tol=1e-3), \
            f"Hti misses analytic x(tau={tau:.3f}) = {x_tau}"
