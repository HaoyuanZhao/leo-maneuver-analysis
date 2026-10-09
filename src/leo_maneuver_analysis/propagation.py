from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from .config import DynamicsConfig, positive


EARTH_MU_M3_S2 = 3.986004418e14
EARTH_RADIUS_M = 6_378_136.3
EARTH_J2 = 1.08262668e-3


@dataclass(frozen=True)
class PropagationResult:
    times_s: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    backend: str
    model: str


class Propagator(Protocol):
    name: str
    model: str

    def propagate(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        impulse_eci_mps: np.ndarray | None = None,
    ) -> PropagationResult: ...

    def propagate_finite_burn(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        thrust_direction_eci: np.ndarray,
        burn_duration_s: float,
        thrust_n: float,
        mass_kg: float,
    ) -> PropagationResult: ...


def _as_vector(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return out


def _validate_sample_times(sample_times_s) -> np.ndarray:
    times = np.asarray(sample_times_s, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("sample_times_s must contain finite non-negative values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample_times_s must be strictly increasing")
    return times


def rtn_basis(position_m: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    """Return a matrix whose columns are radial, transverse and normal unit vectors."""

    position = _as_vector(position_m, "position_m")
    velocity = _as_vector(velocity_mps, "velocity_mps")
    position_norm = float(np.linalg.norm(position))
    if position_norm == 0:
        raise ValueError("position_m must be non-zero when building an RTN basis")
    radial = position / position_norm
    normal_raw = np.cross(position, velocity)
    normal_norm = np.linalg.norm(normal_raw)
    if normal_norm <= 0:
        raise ValueError("position and velocity cannot be collinear when building an RTN basis")
    normal = normal_raw / normal_norm
    transverse = np.cross(normal, radial)
    return np.column_stack((radial, transverse, normal))


def rtn_to_eci(dv_rtn_mps: np.ndarray, position_m: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    return rtn_basis(position_m, velocity_mps) @ _as_vector(dv_rtn_mps, "dv_rtn_mps")


class ScipyJ2Propagator:
    """Earth point-mass plus J2 gravity with constant spacecraft mass."""

    name = "scipy-j2"
    model = "earth_point_mass+j2_constant_mass"

    def __init__(self, max_step_s: float = 30.0) -> None:
        positive(max_step_s, "max_step_s")
        self.max_step_s = float(max_step_s)

    @staticmethod
    def _gravity(position: np.ndarray) -> np.ndarray:
        x, y, z = position
        radius = float(np.linalg.norm(position))
        if radius <= EARTH_RADIUS_M * 0.5:
            raise ValueError("invalid Earth-orbit position")
        base = -EARTH_MU_M3_S2 * position / radius**3
        z2_r2 = (z * z) / (radius * radius)
        factor = 1.5 * EARTH_J2 * EARTH_MU_M3_S2 * EARTH_RADIUS_M**2 / radius**5
        j2 = factor * np.array(
            [x * (5.0 * z2_r2 - 1.0), y * (5.0 * z2_r2 - 1.0), z * (5.0 * z2_r2 - 3.0)]
        )
        return base + j2

    def _integrate(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        acceleration,
    ) -> PropagationResult:
        position = _as_vector(position_m, "position_m")
        velocity = _as_vector(velocity_mps, "velocity_mps")
        times = _validate_sample_times(sample_times_s)
        if float(times[-1]) == 0.0:
            return PropagationResult(
                times_s=times,
                positions_m=np.repeat(position[None, :], len(times), axis=0),
                velocities_mps=np.repeat(velocity[None, :], len(times), axis=0),
                backend=self.name,
                model=self.model,
            )

        def derivative(t_s: float, state: np.ndarray) -> np.ndarray:
            return np.concatenate((state[3:], self._gravity(state[:3]) + acceleration(t_s)))

        result = solve_ivp(
            derivative,
            (0.0, float(times[-1])),
            np.concatenate((position, velocity)),
            t_eval=times,
            rtol=1e-9,
            atol=1e-6,
            max_step=self.max_step_s,
        )
        if not result.success or result.y.shape[1] != len(times):
            raise RuntimeError(f"orbit propagation failed: {result.message}")
        return PropagationResult(
            times_s=times,
            positions_m=result.y[:3].T.copy(),
            velocities_mps=result.y[3:].T.copy(),
            backend=self.name,
            model=self.model,
        )

    def propagate(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        impulse_eci_mps: np.ndarray | None = None,
    ) -> PropagationResult:
        velocity = _as_vector(velocity_mps, "velocity_mps").copy()
        if impulse_eci_mps is not None:
            velocity += _as_vector(impulse_eci_mps, "impulse_eci_mps")
        return self._integrate(position_m, velocity, sample_times_s, lambda _time: np.zeros(3))

    def propagate_finite_burn(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        thrust_direction_eci: np.ndarray,
        burn_duration_s: float,
        thrust_n: float,
        mass_kg: float,
    ) -> PropagationResult:
        direction = _as_vector(thrust_direction_eci, "thrust_direction_eci")
        direction_norm = float(np.linalg.norm(direction))
        positive(burn_duration_s, "burn_duration_s", zero_allowed=True)
        positive(thrust_n, "thrust_n")
        positive(mass_kg, "mass_kg")
        if direction_norm <= 0:
            raise ValueError("thrust_direction_eci must be non-zero")
        acceleration = direction / direction_norm * (float(thrust_n) / float(mass_kg))
        times = _validate_sample_times(sample_times_s)
        if burn_duration_s == 0:
            return self.propagate(position_m, velocity_mps, times)
        if burn_duration_s >= times[-1]:
            return self._integrate(position_m, velocity_mps, times, lambda _time: acceleration)
        # Integrate the discontinuity explicitly, rather than letting an
        # adaptive step straddle the instant when thrust turns off.
        during = times <= burn_duration_s
        burn_times = np.unique(np.append(times[during], burn_duration_s))
        burned = self._integrate(position_m, velocity_mps, burn_times, lambda _time: acceleration)
        coasted = self.propagate(burned.positions_m[-1], burned.velocities_mps[-1],
                                 times[~during] - burn_duration_s)
        count = int(np.count_nonzero(during))
        return PropagationResult(times,
                                 np.concatenate((burned.positions_m[:count], coasted.positions_m)),
                                 np.concatenate((burned.velocities_mps[:count], coasted.velocities_mps)),
                                 self.name, self.model)


class BasiliskPropagator:
    """Thin adapter around the Basilisk spacecraft simulation module."""

    name = "basilisk"
    model = "basilisk_spacecraft_earth_point_mass_constant_mass"

    def __init__(self, step_s: float = 30.0) -> None:
        positive(step_s, "step_s")
        try:
            from Basilisk import __version__ as bsk_version
            from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
            from Basilisk.simulation import extForceTorque, spacecraft
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The optional Basilisk backend is not installed. "
                "Install it with `python -m pip install '.[basilisk]'`."
            ) from exc
        self.version = str(bsk_version)
        self.step_s = float(step_s)
        self._SimulationBaseClass = SimulationBaseClass
        self._macros = macros
        self._simIncludeGravBody = simIncludeGravBody
        self._spacecraft = spacecraft
        self._extForceTorque = extForceTorque

    def _run(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        thrust_direction_eci: np.ndarray | None = None,
        burn_duration_s: float = 0.0,
        thrust_n: float = 0.0,
        mass_kg: float = 575.0,
    ) -> PropagationResult:
        position = _as_vector(position_m, "position_m")
        velocity = _as_vector(velocity_mps, "velocity_mps")
        requested_times = _validate_sample_times(sample_times_s)
        positive(mass_kg, "mass_kg")
        if not np.all(np.isclose(requested_times / self.step_s,
                                 np.round(requested_times / self.step_s), atol=1e-8, rtol=0)):
            raise ValueError("Basilisk sample times must be multiples of step_s; use a smaller step")
        if thrust_direction_eci is not None and not np.isclose(
                burn_duration_s / self.step_s, round(burn_duration_s / self.step_s), atol=1e-8, rtol=0):
            raise ValueError("Basilisk burn duration must be a multiple of step_s; use a smaller step")
        if requested_times[-1] == 0:
            return PropagationResult(requested_times, position[None, :].copy(),
                                     velocity[None, :].copy(), self.name, self.model)
        simulation = self._SimulationBaseClass.SimBaseClass()
        process = simulation.CreateNewProcess("orbitProcess")
        task = simulation.CreateNewTask("orbitTask", self._macros.sec2nano(self.step_s))
        process.addTask(task)

        spacecraft = self._spacecraft.Spacecraft()
        spacecraft.ModelTag = "maneuveringSpacecraft"
        spacecraft.hub.mHub = float(mass_kg)
        spacecraft.hub.r_CN_NInit = position.tolist()
        spacecraft.hub.v_CN_NInit = velocity.tolist()
        gravity_factory = self._simIncludeGravBody.gravBodyFactory()
        earth = gravity_factory.createEarth()
        earth.isCentralBody = True
        gravity_factory.addBodiesTo(spacecraft)
        simulation.AddModelToTask("orbitTask", spacecraft)

        force = None
        if thrust_direction_eci is not None and burn_duration_s > 0.0:
            direction = _as_vector(thrust_direction_eci, "thrust_direction_eci")
            norm = float(np.linalg.norm(direction))
            if norm <= 0 or thrust_n <= 0:
                raise ValueError("finite burn requires non-zero direction and positive thrust")
            force = self._extForceTorque.ExtForceTorque()
            force.ModelTag = "constantInertialThrust"
            force.extForce_N = (direction / norm * float(thrust_n)).tolist()
            spacecraft.addDynamicEffector(force)
            simulation.AddModelToTask("orbitTask", force)

        recorder = spacecraft.scStateOutMsg.recorder(self._macros.sec2nano(self.step_s))
        simulation.AddModelToTask("orbitTask", recorder)
        simulation.InitializeSimulation()

        stop_s = float(requested_times[-1])
        if force is not None and 0.0 < burn_duration_s < stop_s:
            simulation.ConfigureStopTime(self._macros.sec2nano(float(burn_duration_s)))
            simulation.ExecuteSimulation()
            force.extForce_N = [0.0, 0.0, 0.0]
        simulation.ConfigureStopTime(self._macros.sec2nano(stop_s))
        simulation.ExecuteSimulation()

        raw_times = np.asarray(recorder.times(), dtype=float) * self._macros.NANO2SEC
        raw_positions = np.asarray(recorder.r_BN_N, dtype=float)
        raw_velocities = np.asarray(recorder.v_BN_N, dtype=float)
        if raw_times.size == 0:
            raise RuntimeError("Basilisk produced no spacecraft state samples")
        positions = np.column_stack(
            [np.interp(requested_times, raw_times, raw_positions[:, axis]) for axis in range(3)]
        )
        velocities = np.column_stack(
            [np.interp(requested_times, raw_times, raw_velocities[:, axis]) for axis in range(3)]
        )
        return PropagationResult(
            times_s=requested_times,
            positions_m=positions,
            velocities_mps=velocities,
            backend=self.name,
            model=self.model,
        )

    def propagate(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        impulse_eci_mps: np.ndarray | None = None,
    ) -> PropagationResult:
        velocity = _as_vector(velocity_mps, "velocity_mps").copy()
        if impulse_eci_mps is not None:
            velocity += _as_vector(impulse_eci_mps, "impulse_eci_mps")
        return self._run(position_m, velocity, sample_times_s)

    def propagate_finite_burn(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        sample_times_s: np.ndarray,
        *,
        thrust_direction_eci: np.ndarray,
        burn_duration_s: float,
        thrust_n: float,
        mass_kg: float,
    ) -> PropagationResult:
        direction = _as_vector(thrust_direction_eci, "thrust_direction_eci")
        positive(burn_duration_s, "burn_duration_s", zero_allowed=True)
        positive(thrust_n, "thrust_n")
        positive(mass_kg, "mass_kg")
        if np.linalg.norm(direction) <= 0:
            raise ValueError("thrust_direction_eci must be non-zero")
        return self._run(
            position_m,
            velocity_mps,
            sample_times_s,
            thrust_direction_eci=direction,
            burn_duration_s=float(burn_duration_s),
            thrust_n=float(thrust_n),
            mass_kg=float(mass_kg),
        )


def make_propagator(config: DynamicsConfig) -> Propagator:
    config.validate()
    if config.backend == "basilisk":
        return BasiliskPropagator(step_s=config.integrator_step_seconds)
    return ScipyJ2Propagator(max_step_s=config.integrator_step_seconds)
