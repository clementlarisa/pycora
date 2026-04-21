"""Zonotope class — Python port of CORA's @zonotope.

A zonotope is the set { c + sum_i(beta_i * g_i) | beta_i in [-1, 1] }
where c in R^n is the center and g_i in R^n are the generators.

Faithful port of:
  CORA/contSet/@zonotope/{zonotope, plus, mtimes, convHull_, enclose,
                          cartProd_, interval, project}.m

Reference:
  M. Althoff, "Reachability analysis and its application to the safety
  assessment of autonomous cars", PhD dissertation, TU München, 2010.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class Zonotope:
    """Zonotope { c + G * beta | beta in [-1, 1]^p }.

    Attributes:
        c (np.ndarray): center vector, shape (n,).
        G (np.ndarray): generator matrix, shape (n, p). May have zero columns
            if the set is a single point.
    """

    __slots__ = ("c", "G")

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(self, c: np.ndarray | Sequence[float], G: np.ndarray | None = None):
        c = np.asarray(c, dtype=float).reshape(-1)
        if G is None:
            G = np.zeros((c.shape[0], 0))
        else:
            G = np.asarray(G, dtype=float)
            if G.ndim == 1:
                # single generator passed as 1-D vector
                G = G.reshape(-1, 1)
            if G.shape[0] != c.shape[0]:
                raise ValueError(
                    f"Generator matrix has {G.shape[0]} rows, expected {c.shape[0]}"
                )
        self.c = c
        self.G = G

    # -------- alternative constructors ---------------------------------------
    @classmethod
    def from_box(cls, lb: Sequence[float], ub: Sequence[float]) -> "Zonotope":
        """Create a zonotope representing the axis-aligned box [lb, ub]."""
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape != ub.shape:
            raise ValueError("lb and ub must have same shape")
        c = (lb + ub) / 2.0
        radii = (ub - lb) / 2.0
        G = np.diag(radii)
        # drop zero-radius columns to keep generator count minimal
        nonzero = np.abs(radii) > 1e-15
        G = G[:, nonzero]
        return cls(c, G)

    @classmethod
    def from_center_radii(cls, c: Sequence[float], radii: Sequence[float]) -> "Zonotope":
        """Create a box-shaped zonotope centered at c with axis radii."""
        c = np.asarray(c, dtype=float).reshape(-1)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if c.shape != radii.shape:
            raise ValueError("c and radii must have same shape")
        return cls.from_box(c - radii, c + radii)

    @classmethod
    def point(cls, c: Sequence[float]) -> "Zonotope":
        """Singleton zonotope (no generators)."""
        c = np.asarray(c, dtype=float).reshape(-1)
        return cls(c, np.zeros((c.shape[0], 0)))

    # -------------------------------------------------------------------------
    # Basic properties
    # -------------------------------------------------------------------------
    @property
    def n(self) -> int:
        """Ambient dimension."""
        return self.c.shape[0]

    @property
    def num_generators(self) -> int:
        return self.G.shape[1]

    @property
    def order(self) -> float:
        """Zonotope order = num_generators / dim."""
        return self.num_generators / self.n if self.n > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"Zonotope(n={self.n}, generators={self.num_generators}, "
            f"c={np.array2string(self.c, precision=3)})"
        )

    # -------------------------------------------------------------------------
    # Operations (port of CORA @zonotope methods)
    # -------------------------------------------------------------------------
    def plus(self, other) -> "Zonotope":
        """Minkowski sum (CORA plus.m).

        Z + Z': center is c + c', generators are concatenated [G G'].
        Z + v (vector): only the center shifts.
        """
        if isinstance(other, Zonotope):
            if other.n != self.n:
                raise ValueError(
                    f"Dimension mismatch: {self.n} vs {other.n}"
                )
            new_c = self.c + other.c
            new_G = np.hstack([self.G, other.G]) if self.G.size or other.G.size \
                else np.zeros((self.n, 0))
            return Zonotope(new_c, new_G)
        v = np.asarray(other, dtype=float).reshape(-1)
        if v.shape[0] != self.n:
            raise ValueError(f"Vector dim {v.shape[0]} != zonotope dim {self.n}")
        return Zonotope(self.c + v, self.G.copy())

    def linear_map(self, A: np.ndarray) -> "Zonotope":
        """Linear transformation A * Z (CORA mtimes.m).

        A: shape (m, n). Returns zonotope of dim m.
        """
        A = np.asarray(A, dtype=float)
        if A.ndim == 0:
            # scalar
            return Zonotope(A * self.c, A * self.G)
        if A.ndim != 2:
            raise ValueError("A must be 2-D matrix or scalar")
        if A.shape[1] != self.n:
            raise ValueError(f"A has {A.shape[1]} cols, expected {self.n}")
        new_c = A @ self.c
        new_G = A @ self.G if self.G.size else np.zeros((A.shape[0], 0))
        return Zonotope(new_c, new_G)

    def cart_product(self, other: "Zonotope") -> "Zonotope":
        """Cartesian product (CORA cartProd_.m): block-diagonal generator matrix."""
        if not isinstance(other, Zonotope):
            raise TypeError("cart_product requires another Zonotope")
        new_c = np.concatenate([self.c, other.c])
        # block-diagonal G
        n1, p1 = self.G.shape
        n2, p2 = other.G.shape
        new_G = np.zeros((n1 + n2, p1 + p2))
        if p1 > 0:
            new_G[:n1, :p1] = self.G
        if p2 > 0:
            new_G[n1:, p1:] = other.G
        return Zonotope(new_c, new_G)

    def project(self, dims: Sequence[int]) -> "Zonotope":
        """Project onto a subset of coordinate dimensions."""
        idx = list(dims)
        return Zonotope(self.c[idx], self.G[idx, :])

    def enclose(self, other: "Zonotope") -> "Zonotope":
        """Enclose union of self and other in a zonotope (CORA enclose.m).

        Computes a zonotope containing { a*x1 + (1-a)*x2 | x1 in Z, x2 in Z2,
                                          a in [0, 1] }.

        This is used as the convex-hull-like enclosure during reachable-set
        propagation (one_step combines R(t) and R^h(t+dt) via this).
        """
        Z2 = other
        g1 = self.num_generators
        g2 = Z2.num_generators

        # decide which side has more generators
        if g2 <= g1:
            cG = ((self.c - Z2.c) / 2.0).reshape(-1, 1)
            Gcut = self.G[:, :g2]
            Gadd = self.G[:, g2:]
            Gequal = Z2.G
        else:
            cG = ((Z2.c - self.c) / 2.0).reshape(-1, 1)
            Gcut = Z2.G[:, :g1]
            Gadd = Z2.G[:, g1:]
            Gequal = self.G

        new_c = (self.c + Z2.c) / 2.0
        # new_G = [(Gcut+Gequal)/2, cG, (Gcut-Gequal)/2, Gadd]
        parts = [(Gcut + Gequal) / 2.0, cG, (Gcut - Gequal) / 2.0, Gadd]
        new_G = np.hstack([p for p in parts if p.size > 0])
        return Zonotope(new_c, new_G)

    def convex_hull(self, other: "Zonotope") -> "Zonotope":
        """Convex hull enclosure (CORA convHull_.m delegates to enclose)."""
        return self.enclose(other)

    # -------------------------------------------------------------------------
    # Geometric queries
    # -------------------------------------------------------------------------
    def interval(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box (CORA interval.m).

        Returns (lb, ub) — tightest axis-aligned interval enclosing the zonotope.
        """
        delta = np.sum(np.abs(self.G), axis=1)
        return self.c - delta, self.c + delta

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for interval()."""
        return self.interval()

    def contains_point(self, p: Sequence[float], tol: float = 1e-9) -> bool:
        """Test if point p lies in the zonotope.

        Solves the LP: find beta in [-1, 1]^p s.t. G * beta = p - c.
        """
        from scipy.optimize import linprog

        p = np.asarray(p, dtype=float).reshape(-1)
        if p.shape[0] != self.n:
            return False
        if self.num_generators == 0:
            return bool(np.allclose(p, self.c, atol=tol))

        # min 0  s.t.  G beta = p - c,  -1 <= beta <= 1
        target = p - self.c
        bounds = [(-1.0, 1.0)] * self.num_generators
        c_obj = np.zeros(self.num_generators)
        res = linprog(
            c_obj, A_eq=self.G, b_eq=target, bounds=bounds, method="highs"
        )
        return bool(res.success)

    def overlaps_box(self, lb: Sequence[float], ub: Sequence[float]) -> bool:
        """Test if zonotope overlaps the axis-aligned box [lb, ub].

        Sound: returns False only if disjoint. May return True for tight cases
        even when geometric overlap is empty (uses bounding-box test first as
        cheap pre-filter, then point-containment LP at the box center).
        """
        from scipy.optimize import linprog

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape[0] != self.n or ub.shape[0] != self.n:
            raise ValueError("box bounds dim mismatch")

        # 1) cheap bounding-box disjointness pre-filter
        z_lb, z_ub = self.interval()
        if np.any(z_ub < lb) or np.any(z_lb > ub):
            return False

        # 2) exact LP: find p in box and beta in [-1,1]^p s.t. c + G beta = p
        # Variables: [beta_1..beta_p, p_1..p_n]
        p_dim = self.num_generators
        n_dim = self.n
        n_var = p_dim + n_dim

        # Equality: G beta - I p = -c, i.e. [G | -I] [beta; p] = -c
        A_eq = np.hstack([self.G, -np.eye(n_dim)])
        b_eq = -self.c

        bounds = [(-1.0, 1.0)] * p_dim + list(zip(lb.tolist(), ub.tolist()))
        c_obj = np.zeros(n_var)

        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return bool(res.success)

    def overlaps_interval(self, lo: float, hi: float, dim: int) -> bool:
        """Test if the projection onto coordinate `dim` overlaps [lo, hi]."""
        z_lb, z_ub = self.interval()
        return not (z_ub[dim] < lo or z_lb[dim] > hi)

    # -------------------------------------------------------------------------
    # Operator overloads
    # -------------------------------------------------------------------------
    def __add__(self, other) -> "Zonotope":
        return self.plus(other)

    def __radd__(self, other) -> "Zonotope":
        return self.plus(other)

    def __rmatmul__(self, A) -> "Zonotope":
        # A @ Z
        return self.linear_map(A)

    def __mul__(self, scalar: float) -> "Zonotope":
        if not np.isscalar(scalar):
            return NotImplemented
        return Zonotope(scalar * self.c, scalar * self.G)

    __rmul__ = __mul__
