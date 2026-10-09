"""Range-limited ISL topology, nominal routing, and analytical link metrics."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

from .config import ScenarioConfig, integer, positive
from .constellation import Satellite
from .pointing import los_pointing_angle_rad, pointing_link_quality
from .propagation import EARTH_RADIUS_M


SPEED_OF_LIGHT_MPS = 299_792_458.0
Edge = tuple[int, int]


def _positions(value: np.ndarray, count: int) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(f"positions must be a finite ({count}, 3) array")
    if np.any(np.linalg.norm(positions, axis=1) <= EARTH_RADIUS_M):
        raise ValueError("satellite positions must lie outside Earth")
    return positions


def line_of_sight_clear(a_m: np.ndarray, b_m: np.ndarray) -> bool:
    """Check a straight segment against a spherical Earth (no atmosphere)."""
    delta = b_m - a_m
    length_squared = float(np.dot(delta, delta))
    if length_squared <= 0:
        return False
    fraction = float(np.clip(-np.dot(a_m, delta) / length_squared, 0.0, 1.0))
    return bool(np.linalg.norm(a_m + fraction * delta) > EARTH_RADIUS_M)


def build_topology(satellites: Iterable[Satellite], positions_m: np.ndarray,
                   *, max_distance_m: float, degree_cap: int) -> nx.Graph:
    """Connect slot neighbors and nearest satellites on adjacent planes.

    Intra-plane edges have priority. Each node gets at most two intra-plane
    and two cross-plane edges; total degree is also capped. Ties are resolved
    by distance and satellite IDs. Satellite IDs index the position rows.
    """
    satellites = tuple(satellites)
    if [sat.id for sat in satellites] != list(range(len(satellites))):
        raise ValueError("satellite IDs must match position row order (0, 1, ...)")
    positive(max_distance_m, "max_distance_m")
    integer(degree_cap, "degree_cap", 1)
    positions = _positions(positions_m, len(satellites))
    groups: dict[int, list[Satellite]] = defaultdict(list)
    for sat in satellites:
        groups[sat.plane].append(sat)
    candidates: dict[str, set[Edge]] = {"intra_plane": set(), "cross_plane": set()}
    planes = sorted(groups)

    def add(kind: str, a: int, b: int) -> None:
        if a != b:
            candidates[kind].add(tuple(sorted((a, b))))

    for plane, members in groups.items():
        ordered = sorted(members, key=lambda sat: (sat.slot, sat.id))
        if len(ordered) > 1:
            for index, sat in enumerate(ordered):
                add("intra_plane", sat.id, ordered[(index + 1) % len(ordered)].id)
        if len(planes) > 1:
            index = planes.index(plane)
            for neighbor in {planes[(index - 1) % len(planes)], planes[(index + 1) % len(planes)]}:
                ids = sorted(sat.id for sat in groups[neighbor])
                for sat in members:
                    distances = np.linalg.norm(positions[ids] - positions[sat.id], axis=1)
                    add("cross_plane", sat.id, ids[int(np.argmin(distances))])

    graph = nx.Graph()
    graph.add_nodes_from(sat.id for sat in satellites)
    degree: dict[int, int] = defaultdict(int)
    for kind in ("intra_plane", "cross_plane"):
        typed_degree: dict[int, int] = defaultdict(int)
        ordered_edges = sorted(candidates[kind], key=lambda edge: (
            float(np.linalg.norm(positions[edge[0]] - positions[edge[1]])), *edge))
        for a, b in ordered_edges:
            distance = float(np.linalg.norm(positions[a] - positions[b]))
            if distance > max_distance_m or not line_of_sight_clear(positions[a], positions[b]):
                continue
            if degree[a] >= degree_cap or degree[b] >= degree_cap:
                continue
            if typed_degree[a] >= 2 or typed_degree[b] >= 2:
                continue
            graph.add_edge(a, b, kind=kind, distance_m=distance,
                           propagation_delay_s=distance / SPEED_OF_LIGHT_MPS)
            degree[a] += 1
            degree[b] += 1
            typed_degree[a] += 1
            typed_degree[b] += 1
    return graph


def select_route(graph: nx.Graph, source: int, destination: int) -> tuple[int, ...] | None:
    """Select by nominal propagation delay; return None when disconnected."""
    for _, _, attributes in graph.edges(data=True):
        positive(attributes.get("propagation_delay_s"), "edge propagation_delay_s", zero_allowed=True)
    try:
        _, path = nx.single_source_dijkstra(graph, source, destination, weight="propagation_delay_s")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return tuple(path)


@dataclass(frozen=True)
class LinkMetrics:
    a: int
    b: int
    available: bool
    distance_m: float
    pointing_angle_rad: float
    pointing_gain: float
    propagation_delay_s: float
    retry_weighted_delay_s: float | None
    capacity_proxy_mbps: float
    packet_error_rate: float


@dataclass(frozen=True)
class PathMetrics:
    status: str
    path: tuple[int, ...] | None
    propagation_delay_s: float | None
    retry_weighted_delay_s: float | None
    capacity_proxy_mbps: float | None
    packet_error_rate: float | None


def evaluate_links(graph: nx.Graph, nominal_positions_m: np.ndarray,
                   positions_m: np.ndarray, config: ScenarioConfig) -> dict[Edge, LinkMetrics]:
    """Evaluate a fixed set of edges with terminals following nominal LOS.

    Availability is purely geometric. Capacity and errors are analytical
    proxies; a geometrically available link can still have near-zero capacity.
    """
    nominal = _positions(nominal_positions_m, len(graph))
    effective = _positions(positions_m, len(graph))
    output = {}
    for a, b in sorted(tuple(sorted(edge)) for edge in graph.edges):
        distance = float(np.linalg.norm(effective[a] - effective[b]))
        available = (distance <= config.max_isl_distance_m
                     and line_of_sight_clear(effective[a], effective[b]))
        angle = los_pointing_angle_rad(nominal[a], nominal[b], effective[a], effective[b])
        quality = pointing_link_quality(
            angle, beam_width_rad=config.beam_width_rad,
            nominal_snr_linear=config.nominal_snr_linear, packet_bits=config.packet_bits,
            max_retransmission_factor=config.max_retransmission_factor)
        delay = distance / SPEED_OF_LIGHT_MPS
        output[a, b] = LinkMetrics(
            a, b, available, distance, angle, quality["pointing_gain"], delay,
            delay * quality["retransmission_factor"] if available else None,
            config.nominal_capacity_mbps * quality["throughput_factor"] if available else 0.0,
            quality["packet_error_rate"] if available else 1.0)
    return output


def evaluate_path(path: tuple[int, ...] | None, links: dict[Edge, LinkMetrics]) -> PathMetrics:
    """Sum delays, take bottleneck capacity, and multiply per-hop success."""
    if path is None:
        return PathMetrics("unreachable", None, None, None, None, None)
    if len(path) == 0:
        raise ValueError("path must contain at least one satellite")
    if len(path) == 1:
        return PathMetrics("connected", path, 0.0, 0.0, None, 0.0)
    selected = []
    for a, b in zip(path[:-1], path[1:]):
        edge = tuple(sorted((a, b)))
        if edge not in links:
            raise ValueError(f"Missing metrics for path edge {edge}")
        selected.append(links[edge])
    if any(not link.available for link in selected):
        return PathMetrics("unavailable", path, None, None, None, None)
    success = float(np.prod([1.0 - link.packet_error_rate for link in selected]))
    return PathMetrics("connected", path,
                       sum(link.propagation_delay_s for link in selected),
                       sum(link.retry_weighted_delay_s for link in selected),
                       min(link.capacity_proxy_mbps for link in selected), 1.0 - success)
