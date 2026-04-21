"""Interval matrix x zonotope multiplication helper.

CORA represents these as ``matZonotope``. Here, the simpler approach of
computing the bounding box of the matrix-product over the interval matrix
range, returning the result as a box zonotope.

For an interval matrix M with center M_c and radius M_d, and a zonotope X:
  M*X is over-approximated by:
    center: M_c * X.c
    box-radius (per row i):
      sum_j (|M_c[i,j]| * X_radius[j] + M_d[i,j] * |X.c[j]| + M_d[i,j] * X_radius[j])
  where X_radius is the bounding-box half-width of X.
"""
from __future__ import annotations

import numpy as np

from ..zonotope import Zonotope


def interval_matrix_times_zonotope(
    M_min: np.ndarray, M_max: np.ndarray, X: Zonotope
) -> Zonotope:
    """Over-approximate M*X by a box zonotope where M ∈ [M_min, M_max]."""
    M_c = (M_min + M_max) / 2.0
    M_d = (M_max - M_min) / 2.0

    # Bounding box of X
    X_lb, X_ub = X.interval()
    X_c = (X_lb + X_ub) / 2.0
    X_r = (X_ub - X_lb) / 2.0

    new_c = M_c @ X_c
    # Row-wise radius
    new_r = (
        np.abs(M_c) @ X_r
        + M_d @ np.abs(X_c)
        + M_d @ X_r
    )
    return Zonotope.from_center_radii(new_c, new_r)
