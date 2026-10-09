from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from leo_maneuver_analysis import ScenarioConfig
from leo_maneuver_analysis.constellation import initial_constellation
from leo_maneuver_analysis.network import LinkMetrics, build_topology, evaluate_links, evaluate_path, line_of_sight_clear, select_route
from leo_maneuver_analysis.pointing import los_pointing_angle_rad, pointing_link_quality
from leo_maneuver_analysis.propagation import EARTH_RADIUS_M


def test_topology_obeys_distance_earth_and_degree_constraints():
    config = ScenarioConfig()
    satellites, positions, _ = initial_constellation(config)
    graph = build_topology(satellites, positions, max_distance_m=config.max_isl_distance_m, degree_cap=3)
    assert len(graph) == 36
    assert graph.number_of_edges() > 0
    assert max(dict(graph.degree).values()) <= 3
    for a, b in graph.edges:
        assert a != b
        assert np.linalg.norm(positions[a] - positions[b]) <= config.max_isl_distance_m
        assert line_of_sight_clear(positions[a], positions[b])
    for node in graph:
        for kind in ("intra_plane", "cross_plane"):
            assert sum(graph[node][neighbor]["kind"] == kind for neighbor in graph[node]) <= 2


def test_earth_blocks_diameter_link_even_when_range_allows_it():
    config = replace(ScenarioConfig(), num_planes=1, satellites_per_plane=2,
                     source_satellite_id=0, destination_satellite_id=1, maneuver_satellite_id=0)
    satellites, positions, _ = initial_constellation(config)
    graph = build_topology(satellites, positions, max_distance_m=20_000_000.0, degree_cap=4)
    assert graph.number_of_edges() == 0
    assert select_route(graph, 0, 1) is None


def test_weighted_routing_prefers_lower_delay_over_fewer_hops():
    graph = nx.Graph()
    graph.add_edge(0, 1, propagation_delay_s=10.0)
    graph.add_edge(1, 3, propagation_delay_s=10.0)
    graph.add_edge(0, 2, propagation_delay_s=1.0)
    graph.add_edge(2, 4, propagation_delay_s=1.0)
    graph.add_edge(4, 3, propagation_delay_s=1.0)
    graph.add_node(5)
    assert select_route(graph, 0, 3) == (0, 2, 4, 3)
    assert select_route(graph, 0, 5) is None
    assert select_route(graph, 0, 99) is None


def test_link_quality_is_bounded_and_worsens_with_pointing_error():
    kwargs = dict(beam_width_rad=50e-6, nominal_snr_linear=100.0,
                  packet_bits=12000, max_retransmission_factor=20.0)
    qualities = [pointing_link_quality(angle, **kwargs) for angle in (0.0, 30e-6, 50e-6, 100e-6)]
    assert qualities[0]["pointing_gain"] == 1.0
    assert qualities[0]["throughput_factor"] == pytest.approx(1.0)
    assert [quality["throughput_factor"] for quality in qualities] == sorted(
        (quality["throughput_factor"] for quality in qualities), reverse=True)
    assert all(0.0 <= quality["packet_error_rate"] <= 1.0 for quality in qualities)
    assert all(1.0 <= quality["retransmission_factor"] <= 20.0 for quality in qualities)
    assert qualities[-1]["throughput_factor"] == 1e-12


def test_coincident_los_and_invalid_model_fail():
    with pytest.raises(ValueError, match="non-zero"):
        los_pointing_angle_rad(np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="angle_rad"):
        pointing_link_quality(float("nan"), beam_width_rad=50e-6, nominal_snr_linear=100,
                              packet_bits=12000, max_retransmission_factor=20)


def test_path_aggregation_and_unavailable_metrics():
    a = LinkMetrics(0, 1, True, 1000.0, 0.0, 1.0, 0.01, 0.02, 80.0, 0.1)
    b = LinkMetrics(1, 2, True, 1000.0, 0.0, 1.0, 0.03, 0.06, 50.0, 0.2)
    result = evaluate_path((0, 1, 2), {(0, 1): a, (1, 2): b})
    assert result.propagation_delay_s == pytest.approx(0.04)
    assert result.retry_weighted_delay_s == pytest.approx(0.08)
    assert result.capacity_proxy_mbps == 50.0
    assert result.packet_error_rate == pytest.approx(0.28)
    unavailable = evaluate_path((0, 1, 2), {(0, 1): a, (1, 2): replace(b, available=False)})
    assert unavailable.status == "unavailable"
    assert unavailable.propagation_delay_s is None
    assert unavailable.packet_error_rate is None
    assert evaluate_path(None, {}).status == "unreachable"
    assert evaluate_path((0,), {}).capacity_proxy_mbps is None
    with pytest.raises(ValueError, match="Missing metrics"):
        evaluate_path((0, 1), {})


def test_frozen_link_is_marked_unavailable_when_earth_blocks_it():
    config = replace(ScenarioConfig(), num_planes=1, satellites_per_plane=2,
                     source_satellite_id=0, destination_satellite_id=1, maneuver_satellite_id=0)
    positions = np.array([[EARTH_RADIUS_M + 550000, 0, 0], [EARTH_RADIUS_M + 550000, 1000, 0]], dtype=float)
    graph = nx.Graph([(0, 1)])
    blocked = positions.copy()
    blocked[1] = -positions[0]
    links = evaluate_links(graph, positions, blocked, config)
    assert not links[0, 1].available
    assert evaluate_path((0, 1), links).status == "unavailable"
