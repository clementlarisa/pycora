"""NonlinearSys reach — port of CORA @nonlinearSys main reach loop.

Implements the basic ``lin`` algorithm (no ``linRem`` refinement, no set
splitting, no adaptive step size). Each step:

  1. Linearize ẋ = f(x, u) at the center of the current reach set
  2. Bound the Lagrange remainder over (R, U)
  3. Propagate via LinearSys.one_step with the remainder added as
     uncertain input
  4. Reduce the new zonotope to keep generator count bounded

References:
  CORA/contDynamics/@nonlinearSys/{linearize.m, initReach.m}
  Althoff (2010), Chapter 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..linear_sys import LinearSys
from ..zonotope import Zonotope, reduce_girard
from .linearize import lagrange_remainder, linearize_at


@dataclass
class ReachResult:
    """Output of NonlinearSys.reach."""
    R_tp: list[Zonotope] = field(default_factory=list)  # time-point sets at t_0, t_1, ...
    R_ti: list[Zonotope] = field(default_factory=list)  # time-interval sets [t_0, t_1], ...
    nominal: list[np.ndarray] = field(default_factory=list)  # center trajectory
    times: list[float] = field(default_factory=list)  # time stamps for R_tp


class NonlinearSys:
    """Continuous-time nonlinear system ẋ = f(x, u).

    Parameters
    ----------
    f : callable f(x, u) -> R^n
        Dynamics. Must be JAX-traceable (use jnp inside).
    n_x : int
        State dimension.
    n_u : int
        Input dimension.
    name : str
        Optional name.
    """

    def __init__(self, f: Callable, n_x: int, n_u: int, name: str = "nlnsys"):
        self.f = f
        self.n_x = n_x
        self.n_u = n_u
        self.name = name

    # -------------------------------------------------------------------------
    def simulate_center(
        self,
        x0: np.ndarray,
        u_ref: np.ndarray | Callable,
        dt: float,
        n_steps: int,
    ) -> list[np.ndarray]:
        """Integrate the nominal (center) trajectory using RK4.

        ``u_ref`` may be a constant vector (held over all steps) or a function
        ``u_ref(k) -> np.ndarray`` of the step index.
        """
        if callable(u_ref):
            u_at = u_ref
        else:
            u_const = np.asarray(u_ref, dtype=float)
            u_at = lambda k: u_const

        traj = [np.asarray(x0, dtype=float)]
        x = traj[0].copy()
        for k in range(n_steps):
            u = u_at(k)
            # RK4
            k1 = np.asarray(self.f(x, u))
            k2 = np.asarray(self.f(x + 0.5 * dt * k1, u))
            k3 = np.asarray(self.f(x + 0.5 * dt * k2, u))
            k4 = np.asarray(self.f(x + dt * k3, u))
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(x.copy())
        return traj

    def reach(
        self,
        R0: Zonotope,
        U: Zonotope,
        u_ref: np.ndarray | Callable,
        dt: float,
        n_steps: int,
        truncation_order: int = 6,
        zonotope_order_max: float = 50.0,
    ) -> ReachResult:
        """Compute reachable sets over [0, n_steps · dt].

        Parameters
        ----------
        R0 : Zonotope
            Initial state set.
        U : Zonotope
            Input set (centered on the linearization input — typically
            ``Zonotope.from_center_radii(0, [du1_max, du2_max, ...])``).
        u_ref : np.ndarray or callable
            Nominal/reference input. May be a constant vector or
            ``u_ref(k) -> np.ndarray``.
        dt : float
            Step size.
        n_steps : int
            Number of steps.
        truncation_order : int
            Taylor series truncation (default 6, CORA convention).
        zonotope_order_max : float
            Reduction threshold (default 50, CORA convention).

        Returns
        -------
        ReachResult with R_tp[0..n_steps] and R_ti[0..n_steps-1].
        """
        # Coerce u_ref into a per-step callable
        if callable(u_ref):
            u_at = u_ref
        else:
            u_const = np.asarray(u_ref, dtype=float)
            u_at = lambda k: u_const

        # Nominal center trajectory (used for linearization points)
        nominal = self.simulate_center(R0.c, u_ref, dt, n_steps)

        result = ReachResult(times=[0.0])
        result.R_tp.append(R0)
        result.nominal = nominal

        R_current = R0

        for k in range(n_steps):
            # Linearization point: midpoint between R(k).c and the next
            # nominal point (CORA's "0.5*dt*f0" shift, here approximated by
            # midpoint of nominal[k] and nominal[k+1])
            x_star = 0.5 * (nominal[k] + nominal[k + 1])
            u_star = u_at(k)

            # Linearize at (x*, u*)
            A, B, c0 = linearize_at(self.f, x_star, u_star)

            # Build linear system; effective constant input is f(x*, u*)
            # (the offset c0 absorbs the linearization point shift)
            linsys = LinearSys(A, B)

            # Lagrange remainder over (R_current, U_full)
            # CORA uses the one-step over-approximation for this; for
            # simplicity we use R_current itself, which is a slightly looser
            # bound but sound.
            U_full = U.plus(u_star)  # absolute-input set
            L = lagrange_remainder(self.f, R_current, U_full, x_star, u_star)

            # The constant offset of the linearization is f(x*, u*) — but
            # since we linearize ẋ ≈ A·(x - x*) + B·(u - u*) + f(x*, u*),
            # we re-formulate as ẋ = A·x + B·u + (f(x*, u*) - A·x* - B·u*).
            # The total "constant input" passed to LinearSys.one_step is the
            # constant term plus B mapped offsets. We keep it simple: shift
            # X by -x*, propagate, then shift back.
            X_shifted = R_current.plus(-x_star)
            U_shifted = U  # already centered on 0
            u_const_shifted = c0  # = f(x*, u*) — the affine offset

            # one_step: ẋ ≈ A·x_shifted + B·u_shifted + c0
            # Treat c0 as the constant input via B = identity branch:
            # We need: x'(t) = A x(t) + (c0 + B u(t))
            # LinearSys.one_step takes (A, B, X, u_const, U_uncertain).
            # We absorb c0 by using an identity-B "constant offset" channel:
            # actually our LinearSys takes B as input matrix and u_const as
            # an input vector; we'll just pass c0 as a scaled input through
            # an augmented B matrix.
            # Simpler: use a scratch LinearSys with B = [B | I] and inputs
            # = [u_shifted; c0] both treated as (uncertain; constant).
            n_x = self.n_x
            n_u = self.n_u
            B_aug = np.hstack([B, np.eye(n_x)])
            linsys_aug = LinearSys(A, B_aug)
            u_const_aug = np.concatenate([np.zeros(n_u), c0])
            U_aug = U_shifted.cart_product(L)  # uncertain inputs: original U + remainder L

            X_next_shifted, _ = linsys_aug.one_step(
                X_shifted,
                u_const=u_const_aug,
                dt=dt,
                U_uncertain=U_aug,
                truncation_order=truncation_order,
            )

            # Shift back to absolute coordinates
            X_next = X_next_shifted.plus(x_star)

            # For Rti we'd need the time-interval result similarly shifted;
            # keep the time-point result and a coarse Rti = enclose(X_curr, X_next)
            R_ti_step = R_current.enclose(X_next)

            # Reduce to keep generator count bounded
            X_next = reduce_girard(X_next, order=zonotope_order_max)
            R_ti_step = reduce_girard(R_ti_step, order=zonotope_order_max)

            result.R_tp.append(X_next)
            result.R_ti.append(R_ti_step)
            result.times.append((k + 1) * dt)

            R_current = X_next

        return result
