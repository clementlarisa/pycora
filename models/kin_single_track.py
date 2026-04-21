"""Kinematic single-track (KS) bicycle model — port of CORA's
vehicleDynamics_KS_cog.m.

State (5D):
    x[0] = s_x        x-position in global frame
    x[1] = s_y        y-position in global frame
    x[2] = δ          steering angle of front wheels
    x[3] = v          velocity at center of gravity
    x[4] = Ψ          yaw angle (vehicle heading)

Input (2D):
    u[0] = δ̇         steering rate
    u[1] = a          longitudinal acceleration

Reference point: center of gravity. Slip angle β is computed from steering
angle and the rear-axle distance (kinematic, no tire forces).

For JAX compatibility, this version omits the steering/acceleration
constraint clipping that CORA does — caller should pass U bounded
appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class KSParams:
    """Vehicle parameters for the kinematic single-track model."""
    a: float          # distance from c.o.g. to front axle [m]
    b: float          # distance from c.o.g. to rear axle [m]

    @property
    def wheelbase(self) -> float:
        return self.a + self.b


def make_ks_dynamics(params: KSParams):
    """Return f(x, u) for the kinematic single-track model.

    The returned function is JAX-traceable (uses jax.numpy).
    """
    l_wb = params.wheelbase
    b = params.b

    def f(x, u):
        # state
        delta = x[2]
        v = x[3]
        psi = x[4]
        # input
        delta_dot = u[0]
        a_long = u[1]

        # slip angle (kinematic): β = atan(tan(δ) · b/L) · sign(v)
        # We use a smooth sign approximation for differentiability:
        sign_v = jnp.tanh(100.0 * v)
        beta = jnp.arctan(jnp.tan(delta) * b / l_wb) * sign_v

        sx_dot = v * jnp.cos(beta + psi)
        sy_dot = v * jnp.sin(beta + psi)
        delta_dot_state = delta_dot
        v_dot = a_long
        psi_dot = v * jnp.cos(beta) * jnp.tan(delta) / l_wb

        return jnp.array([sx_dot, sy_dot, delta_dot_state, v_dot, psi_dot])

    return f


def from_cr_vehicle(vehicle_id: int = 2) -> KSParams:
    """Build KS parameters from a CommonRoad vehicle type.

    Uses commonroad_dc.feasibility.vehicle_dynamics.VehicleParameterMapping.

    Parameters
    ----------
    vehicle_id : int
        CR vehicle type ID (1=Ford Escort, 2=BMW 320i, 3=VW Vanagon, 4=Truck).
    """
    from commonroad.common.solution import VehicleType
    from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

    vp = VehicleParameterMapping.from_vehicle_type(VehicleType(vehicle_id))
    return KSParams(a=vp.a, b=vp.b)
