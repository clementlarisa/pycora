"""LinearSys — port of CORA's @linearSys.

Continuous-time linear time-invariant system: ẋ = A·x + B·u + c
(for our purposes ``c=0`` since the linearization point is encoded
directly in the affine offset around the nominal trajectory).

Reachability follows Althoff (2010), Chapter 3. We implement the basic
``oneStep`` propagation:

  Htp = exp(A·dt) · X                       homogeneous time-point
  Hti = enclose(X, Htp) ⊕ F·X               homogeneous time-interval
  Pu  = ∫₀^dt exp(A·τ) dτ · u_const         constant input
  PU  = ∫₀^dt exp(A·τ) · U_uncertain        time-varying input (zonotope U)
  Rtp = Htp ⊕ Pu ⊕ PU                       full time-point
  Rti = Hti ⊕ enclose(0, Pu) ⊕ G·U ⊕ PU     full time-interval

where F, G are correction-matrix interval matrices (Prop. 3.1).

We omit block decomposition (CORA optimization for very large systems).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import expm

from ..zonotope import Zonotope
from ._interval_matrix import interval_matrix_times_zonotope
from ._taylor import (
    correction_matrix_input,
    correction_matrix_state,
    eAdt_taylor,
    expm_remainder,
    particular_solution_constant,
)


class LinearSys:
    """Continuous-time linear system ẋ = A·x + B·u (+ optional constant).

    Cached values per-time-step to mirror CORA's ``taylorLinSys`` lazy cache.
    """

    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None):
        self.A = np.asarray(A, dtype=float)
        n = self.A.shape[0]
        if self.A.shape[1] != n:
            raise ValueError("A must be square")
        if B is None:
            self.B = np.eye(n)
        else:
            self.B = np.asarray(B, dtype=float)
            if self.B.shape[0] != n:
                raise ValueError(f"B has {self.B.shape[0]} rows, expected {n}")
        self.n = n
        self.m = self.B.shape[1]

        # caches keyed by dt
        self._eAdt_cache: dict[float, np.ndarray] = {}
        self._F_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._G_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._psol_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    # -------------------------------------------------------------------------
    # Cached primitives
    # -------------------------------------------------------------------------
    def get_eAdt(self, dt: float, truncation_order: int = 6) -> np.ndarray:
        """exp(A·dt) — exact via scipy.linalg.expm (matches CORA's getTaylor cache)."""
        key = (dt, truncation_order)
        if key not in self._eAdt_cache:
            # Use exact matrix exponential rather than truncated Taylor — CORA
            # also caches the exact matrix exp here under getTaylor 'eAdt'.
            self._eAdt_cache[key] = expm(self.A * dt)
        return self._eAdt_cache[key]

    def get_F(self, dt: float, truncation_order: int = 6):
        key = (dt, truncation_order)
        if key not in self._F_cache:
            self._F_cache[key] = correction_matrix_state(
                self.A, dt, truncation_order
            )
        return self._F_cache[key]

    def get_G(self, dt: float, truncation_order: int = 6):
        key = (dt, truncation_order)
        if key not in self._G_cache:
            self._G_cache[key] = correction_matrix_input(
                self.A, dt, truncation_order
            )
        return self._G_cache[key]

    def get_psol_constant(self, dt: float, truncation_order: int = 6):
        key = (dt, truncation_order)
        if key not in self._psol_cache:
            self._psol_cache[key] = particular_solution_constant(
                self.A, dt, truncation_order
            )
        return self._psol_cache[key]

    # -------------------------------------------------------------------------
    # Reachability primitives
    # -------------------------------------------------------------------------
    def homogeneous_solution(
        self,
        X: Zonotope,
        dt: float,
        truncation_order: int = 6,
    ) -> tuple[Zonotope, Zonotope]:
        """Compute (Htp, Hti) — homogeneous time-point and time-interval reach.

        Htp = exp(A·dt) · X
        Hti = enclose(X, Htp) ⊕ F·X
        """
        eAdt = self.get_eAdt(dt, truncation_order)
        Htp = X.linear_map(eAdt)

        # Curvature error: F · X (interval matrix times zonotope)
        F_min, F_max = self.get_F(dt, truncation_order)
        C_state = interval_matrix_times_zonotope(F_min, F_max, X)

        Hti = X.enclose(Htp).plus(C_state)
        return Htp, Hti

    def particular_constant(
        self,
        u_const: np.ndarray | Zonotope,
        dt: float,
        truncation_order: int = 6,
    ) -> tuple[Zonotope, Zonotope]:
        """Particular solution for a constant input.

        Returns (Pu_tp, C_input):
          Pu_tp = M · (B · u_const)
          C_input = G · (B · u_const)         (curvature error)

        Then Pti = enclose(0, Pu_tp) ⊕ C_input  is left for the caller.
        """
        # Convert input to zonotope if numeric
        if isinstance(u_const, Zonotope):
            U = u_const
        else:
            u_arr = np.asarray(u_const, dtype=float).reshape(-1)
            U = Zonotope.point(u_arr)

        # Apply input matrix B: BU
        BU = U.linear_map(self.B)

        # Time-point particular solution
        M, W = self.get_psol_constant(dt, truncation_order)
        Pu_tp_center = M @ BU.c
        Pu_tp = Zonotope(Pu_tp_center, M @ BU.G)

        # Add Taylor remainder (W is per-element, treat as box zonotope on BU's bounds)
        if np.any(W > 0):
            BU_lb, BU_ub = BU.interval()
            BU_c = (BU_lb + BU_ub) / 2.0
            BU_r = (BU_ub - BU_lb) / 2.0
            extra_radius = W @ (np.abs(BU_c) + BU_r)
            Pu_tp = Pu_tp.plus(Zonotope.from_center_radii(
                np.zeros(self.n), extra_radius))

        # Curvature error for input: G · BU
        G_min, G_max = self.get_G(dt, truncation_order)
        C_input = interval_matrix_times_zonotope(G_min, G_max, BU)

        return Pu_tp, C_input

    def one_step(
        self,
        X: Zonotope,
        u_const: np.ndarray | Zonotope,
        dt: float,
        U_uncertain: Optional[Zonotope] = None,
        truncation_order: int = 6,
    ) -> tuple[Zonotope, Zonotope]:
        """One step of reachable-set propagation.

        Parameters
        ----------
        X : Zonotope
            Reachable set at time t.
        u_const : Zonotope or np.ndarray
            Constant input over [t, t+dt] (zonotope means uncertain-but-fixed).
        dt : float
            Step size.
        U_uncertain : Zonotope, optional
            Time-varying uncertain input (e.g., for the Lagrange remainder
            additive-disturbance set in nonlinear reach). Treated as constant
            within [t, t+dt] but allowed to vary across steps.
        truncation_order : int
            Taylor series truncation order (default 6, CORA convention).

        Returns
        -------
        (Rtp, Rti) : (Zonotope, Zonotope)
            Reach set at t+dt and over [t, t+dt].
        """
        # Homogeneous part
        Htp, Hti = self.homogeneous_solution(X, dt, truncation_order)

        # Particular part for constant input
        Pu_tp, C_input = self.particular_constant(u_const, dt, truncation_order)

        # Affine time-point: Htp + Pu_tp
        Atp = Htp.plus(Pu_tp)

        # Affine time-interval: Hti + enclose(0, Pu_tp) + C_input
        Pu_ti_approx = Zonotope.point(np.zeros(self.n)).enclose(Pu_tp)
        Ati = Hti.plus(Pu_ti_approx).plus(C_input)

        # Time-varying uncertain input (e.g. Lagrange remainder)
        if U_uncertain is not None:
            PU_tp, PU_ci = self.particular_constant(U_uncertain, dt, truncation_order)
            PU_ti = Zonotope.point(np.zeros(self.n)).enclose(PU_tp).plus(PU_ci)
            Atp = Atp.plus(PU_tp)
            Ati = Ati.plus(PU_ti)

        return Atp, Ati
