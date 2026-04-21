"""Taylor-series helpers for the matrix exponential and CORA correction matrices.

Faithful port of:
  CORA/contDynamics/@linearSys/private/{priv_expmRemainder,
                                         priv_correctionMatrixState,
                                         priv_correctionMatrixInput}.m

Reference:
  M. Althoff, "Reachability analysis and its application to the safety
  assessment of autonomous cars", PhD dissertation, TU München, 2010.
  Chapter 3, Eqs. (3.6), (3.7), Prop. 3.1.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm


def eAdt_taylor(A: np.ndarray, dt: float, truncation_order: int = 6) -> np.ndarray:
    """Truncated Taylor series for exp(A·dt).

    Returns the sum sum_{i=0}^{order} (A·dt)^i / i!  (no remainder term).
    """
    n = A.shape[0]
    M = np.eye(n)
    Apower = np.eye(n)
    for i in range(1, truncation_order + 1):
        Apower = Apower @ A
        M = M + Apower * (dt ** i) / math.factorial(i)
    return M


def expm_remainder(A: np.ndarray, dt: float, truncation_order: int = 6) -> np.ndarray:
    """Remainder of the truncated matrix exponential (CORA priv_expmRemainder.m).

    Returns the half-width W of the interval matrix [-W, W] s.t.
        exp(A·dt) ∈ truncated_taylor(A, dt) + [-W, W]   element-wise.

    Computed via the conservative bound from Althoff (2010), eq. (3.7):
        W = | exp(|A|·dt) - sum_{i=0}^{order} (|A|·dt)^i / i! |
    """
    A_abs = np.abs(A)
    M = eAdt_taylor(A_abs, dt, truncation_order)
    W = np.abs(expm(A_abs * dt) - M)
    return W


def correction_matrix_state(
    A: np.ndarray, dt: float, truncation_order: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Correction matrix F for the state (Althoff Prop. 3.1).

    Bounds the difference between the convex enclosure of x(t) over [0, dt]
    and the true reachable set. Used by ``priv_curvatureState``.

    Returns (F_min, F_max) — the interval matrix bounds.
    """
    n = A.shape[0]
    Asum_pos = np.zeros((n, n))
    Asum_neg = np.zeros((n, n))

    Apower = np.eye(n)  # A^0 (we start the loop at i=2 so iterate from i=1)
    for i in range(1, truncation_order + 1):
        Apower = Apower @ A
        if i == 1:
            continue  # the formula starts at eta = 2
        # factor = (eta^(-eta/(eta-1)) - eta^(-1/(eta-1))) * dt^eta / eta!
        eta = i
        e1 = -eta / (eta - 1.0)
        e2 = -1.0 / (eta - 1.0)
        dtoverfac = (dt ** eta) / math.factorial(eta)
        factor = (eta ** e1 - eta ** e2) * dtoverfac  # negative

        # positive and negative parts of A^eta
        A_pos = np.maximum(Apower, 0.0)
        A_neg = np.minimum(Apower, 0.0)

        # factor < 0, so factor * A_pos is negative (added to lower bound)
        Asum_pos = Asum_pos + factor * A_neg  # contributes to upper bound
        Asum_neg = Asum_neg + factor * A_pos  # contributes to lower bound

    # add remainder of the matrix exponential
    W = expm_remainder(A, dt, truncation_order)
    F_min = Asum_neg - W
    F_max = Asum_pos + W
    return F_min, F_max


def correction_matrix_input(
    A: np.ndarray, dt: float, truncation_order: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Correction matrix G for the constant input (Althoff p. 38).

    Bounds the difference between the convex enclosure of input integral and
    the exact value. Used by ``priv_curvatureInput``.

    Returns (G_min, G_max) — the interval matrix bounds.
    """
    n = A.shape[0]
    Asum_pos = np.zeros((n, n))
    Asum_neg = np.zeros((n, n))

    Apower = np.eye(n)
    for i in range(2, truncation_order + 2):  # eta = 2 to order+1 inclusive
        eta = i
        e1 = -eta / (eta - 1.0)
        e2 = -1.0 / (eta - 1.0)
        dtoverfac = (dt ** eta) / math.factorial(eta)
        factor = (eta ** e1 - eta ** e2) * dtoverfac  # negative

        # uses A^(eta-1)
        A_pow = np.linalg.matrix_power(A, eta - 1)
        A_pos = np.maximum(A_pow, 0.0)
        A_neg = np.minimum(A_pow, 0.0)

        Asum_pos = Asum_pos + factor * A_neg
        Asum_neg = Asum_neg + factor * A_pos

    # remainder term scaled by dt
    W = expm_remainder(A, dt, truncation_order)
    G_min = Asum_neg - W * dt
    G_max = Asum_pos + W * dt
    return G_min, G_max


def particular_solution_constant(
    A: np.ndarray, dt: float, truncation_order: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the linear map M such that M·u = ∫₀^dt exp(A·τ)·u dτ.

    Returns (M_center, W_radius) — center matrix and bounding radii from
    the Taylor truncation remainder. The actual particular solution for input
    u (which may be a zonotope) is M·u with elementwise uncertainty ±W·dt·|u|.

    If A is invertible: M = A^{-1}·(exp(A·dt) - I) (closed form, exact —
    in this case the remainder reduces to the matrix exp remainder).
    Else: truncated power series.
    """
    n = A.shape[0]
    # Try inverse path first (fast & exact up to expm remainder)
    try:
        Ainv = np.linalg.inv(A)
        eAdt = expm(A * dt)
        M = Ainv @ (eAdt - np.eye(n))
        # No additional remainder — expm is exact
        return M, np.zeros((n, n))
    except np.linalg.LinAlgError:
        pass

    # Singular A: use power series sum_{j=0}^{order} A^j · dt^{j+1} / (j+1)!
    M = dt * np.eye(n)
    Apower = np.eye(n)
    for j in range(1, truncation_order + 1):
        Apower = Apower @ A
        M = M + Apower * (dt ** (j + 1)) / math.factorial(j + 1)

    # Remainder of expm scaled by dt (CORA priv_curvatureInput uses E*dt)
    W = expm_remainder(A, dt, truncation_order) * dt
    return M, W
