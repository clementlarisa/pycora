"""Girard zonotope reduction see CORA's priv_reduceGirard.m.

Reduces the number of generators while keeping the zonotope an
over-approximation of the original.

Algorithm (Girard 2005):
  1. Compute metric h_i = ||g_i||_1 - ||g_i||_inf for each generator.
  2. Keep the (d * (order - 1)) generators with the LARGEST h (highest
     "non-axis-aligned" content).
  3. Box-enclose the rest: replace them with diag(sum(|g_i|)).
"""
from __future__ import annotations

import numpy as np

from .zonotope import Zonotope


def reduce_girard(Z: Zonotope, order: float = 50.0) -> Zonotope:
    """Reduce zonotope so its order does not exceed ``order``.

    Order = num_generators / dim. Default 50 to match CORA's default.

    Parameters
    ----------
    Z : Zonotope
        Input zonotope.
    order : float
        Target maximum order (num_generators / dim).

    Returns
    -------
    Zonotope with at most ``int(dim * order)`` generators, over-approximating Z.
    """
    G = Z.G
    d = Z.n
    p = Z.num_generators

    # nothing to reduce
    if p == 0 or p <= d * order:
        return Zonotope(Z.c.copy(), G.copy())

    # drop zero-length generators
    norms = np.linalg.norm(G, axis=0)
    G = G[:, norms > 1e-15]
    p = G.shape[1]
    if p == 0:
        return Zonotope(Z.c.copy(), np.zeros((d, 0)))

    if p <= d * order:
        return Zonotope(Z.c.copy(), G)

    # number of generators to keep unreduced
    n_unreduced = int(np.floor(d * (order - 1)))
    n_unreduced = max(0, n_unreduced)

    # generator metric: h = ||g||_1 - ||g||_inf
    h = np.linalg.norm(G, ord=1, axis=0) - np.linalg.norm(G, ord=np.inf, axis=0)

    if n_unreduced == 0:
        G_unred = np.zeros((d, 0))
        G_red = G
    elif n_unreduced >= p:
        return Zonotope(Z.c.copy(), G)
    else:
        # keep generators with the largest h (those that are most "non-aligned"
        # are LESS informative when boxed, so we keep them as-is)
        idx_unred = np.argpartition(-h, n_unreduced)[:n_unreduced]
        idx_unred = np.sort(idx_unred)
        mask = np.zeros(p, dtype=bool)
        mask[idx_unred] = True
        G_unred = G[:, mask]
        G_red = G[:, ~mask]

    # box-enclose the reduced generators: each row sums |g_ij|, becomes
    # an axis-aligned generator
    if G_red.shape[1] > 0:
        d_box = np.sum(np.abs(G_red), axis=1)
        G_box = np.diag(d_box)
        # drop zero rows of G_box (no contribution to that dim)
        nonzero = np.abs(d_box) > 1e-15
        G_box = G_box[:, nonzero]
    else:
        G_box = np.zeros((d, 0))

    G_new = np.hstack([G_unred, G_box]) if G_unred.size or G_box.size \
        else np.zeros((d, 0))
    return Zonotope(Z.c.copy(), G_new)
