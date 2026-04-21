# pycora

Minimal Python port of [CORA](https://tumcps.github.io/CORA/) (Continuous
Reachability Analyzer) focused on autonomous-vehicle reachability with
bicycle kinematics.

Goal: provide reachability with **orientation tracking** (which the
production [`commonroad-reach`](https://commonroad.in.tum.de/tools/commonroad-reach)
package omits in favor of a 4D point-mass model).

## Features

- `Zonotope` set representation with CORA's core operations (`plus`,
  `linear_map`, `enclose`, `convex_hull`, `cart_product`, `project`,
  `interval`, `contains_point`)
- Girard zonotope reduction (`reduce_girard`)
- Linear-system reach with truncated-Taylor matrix exponential and
  Althoff's correction matrices F (state) and G (input)
- Nonlinear-system reach via per-step linearization + Lagrange remainder
  bound (uses JAX `jacfwd` and `hessian` for analytical derivatives)
- 5D kinematic single-track (KS) bicycle model

## Status

v0.0.1 — minimum viable port for checking reachability of CommonRoad
scenarios with goal orientation constraints. Skipped (vs full CORA):
matrix zonotopes, adaptive step size, set splitting, hybrid systems,
DAEs, dynamic single-track (ST) model with friction.

## Usage

```python
import numpy as np
from pycora import (
    Zonotope, NonlinearSys, KSParams, make_ks_dynamics,
)

# 5D KS bicycle dynamics
params = KSParams(a=1.1562, b=1.4227)  # BMW 320i
f = make_ks_dynamics(params)
sys = NonlinearSys(f, n_x=5, n_u=2)

# Initial state with uncertainty
x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0])  # [x, y, delta, v, psi]
R0 = Zonotope.from_center_radii(x0, [0.2, 0.2, 0.02, 0.2, 0.05])

# Input set centered on (delta_dot=0, a=0)
U = Zonotope.from_center_radii([0.0, 0.0], [0.5, 5.0])

# Reach
res = sys.reach(R0=R0, U=U, u_ref=np.array([0.0, 0.0]),
                dt=0.1, n_steps=20)

# Project final reach set onto (x, y) plane and check orientation
R_final = res.R_tp[-1]
xy = R_final.project([0, 1])
psi = R_final.project([4])
psi_lb, psi_ub = psi.interval()
print(f"final orientation range: [{psi_lb[0]:.3f}, {psi_ub[0]:.3f}] rad")
```

## Tests

```bash
PYTHONPATH=src python -m pytest src/pycora/tests/ -v
```

## Reference

M. Althoff, "Reachability analysis and its application to the safety
assessment of autonomous cars", PhD dissertation, TU München, 2010.

CORA toolbox: https://tumcps.github.io/CORA/
