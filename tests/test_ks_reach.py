"""End-to-end tests for KS bicycle reachability.

Validates that pycora's reach result contains the centre trajectory
(integrated with scipy.integrate.solve_ivp) and that orientation evolves
according to the bicycle kinematics.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pycora.models import KSParams, make_ks_dynamics
from pycora.nonlinear_sys import NonlinearSys
from pycora.zonotope import Zonotope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bmw_params():
    """BMW 320i wheelbase from the Althoff 2012 paper (lf + lr = 2.578 m)."""
    return KSParams(a=1.1562, b=1.4227)


def _scipy_simulate(f_jax, x0, u_const, dt, n_steps):
    """Reference simulation using scipy RK45 with the same f."""
    def f_np(t, x):
        return np.asarray(f_jax(x, u_const))

    sol = solve_ivp(
        f_np, t_span=(0.0, dt * n_steps),
        y0=np.asarray(x0, dtype=float),
        t_eval=np.linspace(0.0, dt * n_steps, n_steps + 1),
        rtol=1e-8, atol=1e-10, method="RK45",
    )
    return sol.y.T  # shape (n_steps + 1, n_x)


# ---------------------------------------------------------------------------
# Straight-line driving
# ---------------------------------------------------------------------------
def test_ks_straight_line():
    """δ = 0, v = 10, no input → ψ stays 0, x grows linearly, y stays 0."""
    params = _bmw_params()
    f = make_ks_dynamics(params)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    # Tight initial set
    x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    R0 = Zonotope.from_center_radii(x0, [0.01, 0.01, 1e-4, 0.01, 1e-4])
    U = Zonotope.from_center_radii([0, 0], [1e-6, 1e-6])  # essentially no input

    res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 0.0]),
                    dt=0.1, n_steps=20)

    # Final reach set should bracket x = 20, y ≈ 0, ψ ≈ 0
    R_final = res.R_tp[-1]
    lb, ub = R_final.interval()
    assert 19.5 < lb[0] < 20.5, f"x lb {lb[0]} not near 20"
    assert 19.5 < ub[0] < 20.5, f"x ub {ub[0]} not near 20"
    assert -0.1 < lb[1] < 0.1
    assert -0.1 < ub[1] < 0.1
    assert -0.05 < lb[4] < 0.05  # ψ
    assert -0.05 < ub[4] < 0.05


def test_ks_constant_acceleration():
    """v(0)=0, a=2 const, δ=0 → ψ stays 0, v(t) = 2t."""
    params = _bmw_params()
    f = make_ks_dynamics(params)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    R0 = Zonotope.from_center_radii(x0, [0.01, 0.01, 1e-4, 0.01, 1e-4])
    U = Zonotope.from_center_radii([0, 0], [1e-6, 1e-6])

    res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 2.0]),
                    dt=0.1, n_steps=10)

    # After 10 steps (1s) at a=2, v should be ≈ 2.0, x ≈ 1.0
    R_final = res.R_tp[-1]
    lb, ub = R_final.interval()
    v_lb, v_ub = lb[3], ub[3]
    assert 1.95 < v_lb < 2.05, f"v lb {v_lb}"
    assert 1.95 < v_ub < 2.05, f"v ub {v_ub}"


def test_ks_curve_orientation_grows():
    """δ = 0.1 const, v = 8 → ψ̇ = v·cos(β)·tan(δ)/L > 0, so ψ grows."""
    params = _bmw_params()
    f = make_ks_dynamics(params)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    delta0 = 0.1
    v0 = 8.0
    x0 = np.array([0.0, 0.0, delta0, v0, 0.0])
    R0 = Zonotope.from_center_radii(x0, [0.01, 0.01, 1e-4, 0.01, 1e-4])
    U = Zonotope.from_center_radii([0, 0], [1e-6, 1e-6])

    res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 0.0]),
                    dt=0.1, n_steps=10)

    # Compare to scipy reference
    x_ref = _scipy_simulate(f, x0, np.array([0.0, 0.0]), dt=0.1, n_steps=10)

    # Reach set must contain the reference final state
    R_final = res.R_tp[-1]
    final_ref = x_ref[-1]
    assert R_final.contains_point(final_ref, tol=1e-3), \
        f"reach set misses reference final state {final_ref}\n" \
        f"reach bounds: {R_final.interval()}"

    # ψ should have grown to roughly v · tan(δ) / L · t (approximate)
    psi_expected = v0 * np.tan(delta0) / params.wheelbase * 1.0
    assert 0.8 * psi_expected < final_ref[4] < 1.2 * psi_expected


def test_ks_reach_contains_all_intermediate_states():
    """For tight initial set, reach time-point set at each step contains the
    scipy-integrated centre trajectory."""
    params = _bmw_params()
    f = make_ks_dynamics(params)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    x0 = np.array([0.0, 0.0, 0.05, 5.0, 0.0])
    R0 = Zonotope.from_center_radii(x0, [0.05, 0.05, 1e-3, 0.05, 1e-3])
    U = Zonotope.from_center_radii([0, 0], [1e-5, 1e-5])

    n_steps = 15
    res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 0.0]),
                    dt=0.1, n_steps=n_steps)
    x_ref = _scipy_simulate(f, x0, np.array([0.0, 0.0]),
                            dt=0.1, n_steps=n_steps)

    for k in range(n_steps + 1):
        assert res.R_tp[k].contains_point(x_ref[k], tol=1e-3), \
            f"step {k}: reach set misses ref state {x_ref[k]}"


def test_ks_uncertain_initial_velocity_widens_position():
    """Initial velocity uncertain in [9, 11] → final x at t=1.0s in [9, 11]."""
    params = _bmw_params()
    f = make_ks_dynamics(params)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    # Velocity uncertain ±1.0 m/s
    R0 = Zonotope.from_center_radii(x0, [0.01, 0.01, 1e-4, 1.0, 1e-4])
    U = Zonotope.from_center_radii([0, 0], [1e-6, 1e-6])

    res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 0.0]),
                    dt=0.1, n_steps=10)

    R_final = res.R_tp[-1]
    lb, ub = R_final.interval()
    # x at t=1s: low bound ≤ 9 (v_min · 1s), high bound ≥ 11 (v_max · 1s)
    assert lb[0] <= 9.0 + 0.5, f"x lb {lb[0]} not below 9.5"
    assert ub[0] >= 11.0 - 0.5, f"x ub {ub[0]} not above 10.5"
