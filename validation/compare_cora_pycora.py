"""Compare CORA MATLAB and pycora reach outputs side by side.

Reads ks_curve_reach_bounds.csv from validation/cora_outputs/ and
validation/pycora_outputs/ and prints per-step bounds + relative error.

Both CSVs have columns:
  t, lb_x, ub_x, lb_y, ub_y, lb_delta, ub_delta, lb_v, ub_v, lb_psi, ub_psi
"""
from __future__ import annotations

import os
import sys

import numpy as np


HERE = os.path.dirname(__file__)
CORA_CSV = os.path.join(HERE, "cora_outputs", "ks_curve_reach_bounds.csv")
PYCORA_CSV = os.path.join(HERE, "pycora_outputs", "ks_curve_reach_bounds.csv")

DIM_NAMES = ["x", "y", "delta", "v", "psi"]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: missing {path}\n"
                 f"Run the corresponding script first.")
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main():
    cora = load(CORA_CSV)
    pyco = load(PYCORA_CSV)

    if cora.shape != pyco.shape:
        print(f"WARNING: shape mismatch — CORA {cora.shape}, pycora {pyco.shape}")
        n = min(cora.shape[0], pyco.shape[0])
        cora = cora[:n]
        pyco = pyco[:n]

    n_steps = cora.shape[0]
    print(f"Comparing {n_steps} time-point reach sets")
    print(f"  CORA  : {CORA_CSV}")
    print(f"  pycora: {PYCORA_CSV}")
    print()

    # ----- per-step relative error in widths -----
    print(f"{'step':>4} {'t':>5}", end="")
    for d in DIM_NAMES:
        print(f" {('|' + d + ' Δlb|'):>10} {('|' + d + ' Δub|'):>10}", end="")
    print()
    print("-" * 110)
    max_abs_err = np.zeros(5)
    for k in range(n_steps):
        t = cora[k, 0]
        print(f"{k:>4} {t:>5.2f}", end="")
        for d in range(5):
            lb_diff = abs(cora[k, 1 + 2*d] - pyco[k, 1 + 2*d])
            ub_diff = abs(cora[k, 2 + 2*d] - pyco[k, 2 + 2*d])
            print(f" {lb_diff:>10.2e} {ub_diff:>10.2e}", end="")
            max_abs_err[d] = max(max_abs_err[d], lb_diff, ub_diff)
        print()

    print()
    print("Maximum absolute bound error per dimension:")
    for d, name in enumerate(DIM_NAMES):
        print(f"  {name:5s}: {max_abs_err[d]:.4e}")

    # ----- final-step side-by-side -----
    print()
    print(f"Final-step bounds (t={cora[-1, 0]:.2f}):")
    print(f"{'dim':<6} {'CORA lb':>12} {'CORA ub':>12} "
          f"{'pycora lb':>12} {'pycora ub':>12}  {'Δ width':>10}")
    print("-" * 70)
    for d, name in enumerate(DIM_NAMES):
        cora_w = cora[-1, 2 + 2*d] - cora[-1, 1 + 2*d]
        pyco_w = pyco[-1, 2 + 2*d] - pyco[-1, 1 + 2*d]
        print(f"{name:<6} {cora[-1, 1+2*d]:>12.6f} {cora[-1, 2+2*d]:>12.6f} "
              f"{pyco[-1, 1+2*d]:>12.6f} {pyco[-1, 2+2*d]:>12.6f}  "
              f"{(pyco_w - cora_w):>+10.4e}")


if __name__ == "__main__":
    main()
