"""End-to-end example: KS bicycle reach over 2 seconds on a straight road.

Run from the repo root:
    PYTHONPATH=src python src/pycora/examples/ks_straight_example.py
"""
import numpy as np

from pycora import KSParams, NonlinearSys, Zonotope, make_ks_dynamics


def main():
    # BMW 320i wheelbase
    params = KSParams(a=1.1562, b=1.4227)
    f = make_ks_dynamics(params)
    sys = NonlinearSys(f, n_x=5, n_u=2)

    # Initial: at origin, facing +x, v = 10 m/s, no steering
    x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    R0 = Zonotope.from_center_radii(x0, [0.2, 0.2, 0.02, 0.2, 0.05])

    # Inputs: bounded steering rate ±0.4 rad/s, bounded accel ±2 m/s²
    U = Zonotope.from_center_radii([0.0, 0.0], [0.4, 2.0])

    print(f"Initial state center: {x0}")
    print(f"Initial uncertainty radii: {[0.2, 0.2, 0.02, 0.2, 0.05]}")
    print(f"Input bounds: ±[0.4, 2.0] (steering rate, acceleration)")
    print()

    # Reach for 2 seconds (20 steps at dt=0.1)
    res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 0.0]),
                    dt=0.1, n_steps=20)

    print(f"{'step':>5} {'time':>6} {'x range':>20} {'y range':>16} "
          f"{'v range':>14} {'psi range':>16}")
    print("-" * 80)
    for k in [0, 1, 5, 10, 15, 20]:
        Rt = res.R_tp[k]
        lb, ub = Rt.interval()
        print(f"{k:>5} {res.times[k]:>6.2f} "
              f"[{lb[0]:>7.2f}, {ub[0]:>7.2f}]  "
              f"[{lb[1]:>5.2f}, {ub[1]:>5.2f}]  "
              f"[{lb[3]:>5.2f}, {ub[3]:>5.2f}]  "
              f"[{lb[4]:>+5.3f}, {ub[4]:>+5.3f}]")


if __name__ == "__main__":
    main()
