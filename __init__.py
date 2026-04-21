"""Python port of CORA reachability for autonomous driving.

Implements:
  - Zonotope set representation (CORA zonotope)
  - Girard zonotope reduction (priv_reduceGirard)
  - LinearSys reach with one-step propagation (CORA linearSys)
  - NonlinearSys reach via linearization + Lagrange remainder (CORA nonlinearSys)
  - Kinematic single-track (KST) bicycle model (vehicleDynamics_KS_cog)

Uses JAX for analytical Jacobians and Hessians for symbolic-derivative MATLAB match.

Reference: M. Althoff, "Reachability analysis and its application to the safety assessment of autonomous cars", PhD dissertation, TU München, 2010.
"""
from .linear_sys import LinearSys
from .models import KSParams, from_cr_vehicle, make_ks_dynamics
from .nonlinear_sys import NonlinearSys, ReachResult, lagrange_remainder, linearize_at
from .zonotope import Zonotope, reduce_girard

__all__ = [
    "Zonotope",
    "reduce_girard",
    "LinearSys",
    "NonlinearSys",
    "ReachResult",
    "linearize_at",
    "lagrange_remainder",
    "KSParams",
    "make_ks_dynamics",
    "from_cr_vehicle",
]
__version__ = "0.0.1"
