"""Compare nominal and maneuvered trajectories on the same prescribed route."""

from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig
from .constellation import Satellite, initial_constellation
from .network import LinkMetrics, PathMetrics, build_topology, evaluate_links, evaluate_path, select_route
from .propagation import make_propagator, rtn_to_eci


@dataclass(frozen=True)
class Snapshot:
    time_s: float
    displacement_m: float
    nominal: PathMetrics
    maneuvered: PathMetrics
    nominal_links: dict[tuple[int, int], LinkMetrics]
    maneuvered_links: dict[tuple[int, int], LinkMetrics]

    @property
    def path_pointing_angle_rad(self) -> float | None:
        if self.maneuvered.path is None:
            return None
        edges = [tuple(sorted(edge)) for edge in zip(self.maneuvered.path[:-1], self.maneuvered.path[1:])]
        return max((self.maneuvered_links[edge].pointing_angle_rad for edge in edges), default=0.0)


@dataclass(frozen=True)
class AnalysisResult:
    config: ScenarioConfig
    satellites: tuple[Satellite, ...]
    times_s: np.ndarray
    nominal_positions_m: np.ndarray
    nominal_velocities_mps: np.ndarray
    maneuvered_positions_m: np.ndarray
    maneuvered_velocities_mps: np.ndarray
    route: tuple[int, ...] | None
    backend: str
    model: str
    snapshots: tuple[Snapshot, ...]


def run_analysis(config: ScenarioConfig) -> AnalysisResult:
    """Run offline; position/velocity arrays have shape (time, satellite, xyz).

    The nominal graph and route are chosen at the maneuver time and frozen
    for both cases. Each edge's range and Earth clearance are rechecked at
    each snapshot. A missing link never triggers implicit route switching.
    """
    config.validate()
    satellites, positions, velocities = initial_constellation(config)
    times = np.unique(np.concatenate((np.arange(0.0, config.duration_s, config.sample_step_s),
                                      [config.maneuver_time_s, config.duration_s])))
    propagator = make_propagator(config.dynamics)
    nominal = [propagator.propagate(position, velocity, times)
               for position, velocity in zip(positions, velocities)]
    nominal_positions = np.stack([result.positions_m for result in nominal], axis=1)
    nominal_velocities = np.stack([result.velocities_mps for result in nominal], axis=1)
    maneuvered_positions = nominal_positions.copy()
    maneuvered_velocities = nominal_velocities.copy()
    start = int(np.flatnonzero(times == config.maneuver_time_s)[0])
    sat = config.maneuver_satellite_id
    if np.any(np.asarray(config.impulse_rtn_mps) != 0):
        impulse = rtn_to_eci(np.array(config.impulse_rtn_mps),
                             nominal_positions[start, sat], nominal_velocities[start, sat])
        moved = propagator.propagate(
            nominal_positions[start, sat], nominal_velocities[start, sat],
            times[start:] - config.maneuver_time_s, impulse_eci_mps=impulse)
        maneuvered_positions[start:, sat] = moved.positions_m
        maneuvered_velocities[start:, sat] = moved.velocities_mps

    graph = build_topology(satellites, nominal_positions[start],
                           max_distance_m=config.max_isl_distance_m, degree_cap=config.isl_degree_cap)
    route = select_route(graph, config.source_satellite_id, config.destination_satellite_id)
    snapshots = []
    for index, time_s in enumerate(times):
        baseline = evaluate_links(graph, nominal_positions[index], nominal_positions[index], config)
        effective = evaluate_links(graph, nominal_positions[index], maneuvered_positions[index], config)
        displacement = float(np.linalg.norm(maneuvered_positions[index, sat] - nominal_positions[index, sat]))
        snapshots.append(Snapshot(float(time_s), displacement,
                                  evaluate_path(route, baseline), evaluate_path(route, effective),
                                  baseline, effective))
    return AnalysisResult(config, satellites, times, nominal_positions, nominal_velocities,
                          maneuvered_positions, maneuvered_velocities, route,
                          propagator.name, propagator.model, tuple(snapshots))
