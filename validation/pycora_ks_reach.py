"""Run pycora reach with the setup cora_ks_reach.m and save bounds CSV.

Then `compare_cora_pycora.py`
"""
from __future__ import annotations

import os

import numpy as np

from pycora import KSParams, NonlinearSys, Zonotope, make_ks_dynamics


# Vehicle params (BMW 320i)
PARAMS = KSParams(a=1.1562, b=1.4227)

# Initial state
DELTA0 = 0.1
V0 = 8.0
X0 = np.array([0.0, 0.0, DELTA0, V0, 0.0])
INIT_RADII = np.array([0.01, 0.01, 1e-4, 0.01, 1e-4])

# Input set centered at zero
U_CENTER = np.array([0.0, 0.0])
U_RADII = np.array([1e-6, 1e-6])

# Time params
DT = 0.1
N_STEPS = 10
TAYLOR_ORDER = 6
ZONOTOPE_ORDER = 50.0

OUT_DIR = os.path.join(os.path.dirname(__file__), "pycora_outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    f = make_ks_dynamics(PARAMS)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    R0 = Zonotope.from_center_radii(X0, INIT_RADII)
    U = Zonotope.from_center_radii(U_CENTER, U_RADII)

    print(f"Running pycora reach: {N_STEPS} steps at dt={DT}...")
    res = sys.reach(
        R0=R0, U=U, u_ref=U_CENTER,
        dt=DT, n_steps=N_STEPS,
        truncation_order=TAYLOR_ORDER,
        zonotope_order_max=ZONOTOPE_ORDER,
    )
    print(f"Got {len(res.R_tp)} time-point sets.")

    csv_path = os.path.join(OUT_DIR, "ks_curve_reach_bounds.csv")
    header = "t,lb_x,ub_x,lb_y,ub_y,lb_delta,ub_delta,lb_v,ub_v,lb_psi,ub_psi"

    with open(csv_path, "w") as fh:
        fh.write(header + "\n")
        for k, Rt in enumerate(res.R_tp):
            t = res.times[k]
            lb, ub = Rt.interval()
            row = [t]
            for d in range(5):
                row.append(lb[d])
                row.append(ub[d])
            fh.write(",".join(f"{x:.10f}" for x in row) + "\n")
    print(f"Wrote: {csv_path}")

    # Print summary
    Rt_final = res.R_tp[-1]
    lb, ub = Rt_final.interval()
    print(f"\nFinal reach set bounds (t={res.times[-1]:.2f}):")
    print(f"  x:     [{lb[0]:.4f}, {ub[0]:.4f}]")
    print(f"  y:     [{lb[1]:.4f}, {ub[1]:.4f}]")
    print(f"  delta: [{lb[2]:.4f}, {ub[2]:.4f}]")
    print(f"  v:     [{lb[3]:.4f}, {ub[3]:.4f}]")
    print(f"  psi:   [{lb[4]:.4f}, {ub[4]:.4f}]")


if __name__ == "__main__":
    main()
