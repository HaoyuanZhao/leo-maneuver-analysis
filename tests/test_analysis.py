from dataclasses import replace
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from leo_maneuver_analysis import ScenarioConfig, load_config, run_analysis
from leo_maneuver_analysis.export import write_results
from leo_maneuver_analysis.propagation import rtn_to_eci


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "maneuver_impact.json"


@pytest.fixture(scope="module")
def default_result():
    return run_analysis(load_config(EXAMPLE))


def test_example_is_connected_with_physical_trajectory_change(default_result):
    result = default_result
    assert result.route == (0, 1, 2, 3, 4, 5, 6)
    assert result.nominal_positions_m.shape == (61, 36, 3)
    final = result.snapshots[-1]
    assert final.displacement_m == pytest.approx(272.626, abs=0.05)
    assert final.path_pointing_angle_rad * 1e6 == pytest.approx(74.164, abs=0.05)
    assert final.maneuvered.status == "connected"
    assert final.maneuvered.capacity_proxy_mbps < final.nominal.capacity_proxy_mbps
    start = int(np.flatnonzero(result.times_s == result.config.maneuver_time_s)[0])
    np.testing.assert_array_equal(result.nominal_positions_m[:start + 1], result.maneuvered_positions_m[:start + 1])
    sat = result.config.maneuver_satellite_id
    expected = rtn_to_eci(np.array(result.config.impulse_rtn_mps), result.nominal_positions_m[start, sat],
                          result.nominal_velocities_mps[start, sat])
    np.testing.assert_allclose(result.maneuvered_velocities_mps[start, sat] - result.nominal_velocities_mps[start, sat],
                               expected, atol=1e-10)


def test_zero_maneuver_exactly_matches_baseline():
    result = run_analysis(replace(ScenarioConfig(), impulse_rtn_mps=(0.0, 0.0, 0.0)))
    np.testing.assert_array_equal(result.nominal_positions_m, result.maneuvered_positions_m)
    for snapshot in result.snapshots:
        assert snapshot.nominal == snapshot.maneuvered
        assert snapshot.displacement_m == 0.0
        assert snapshot.path_pointing_angle_rad == 0.0


def test_non_grid_maneuver_and_terminal_times_are_included():
    config = replace(ScenarioConfig(), maneuver_time_s=613.5, duration_s=1801.0)
    result = run_analysis(config)
    assert 613.5 in result.times_s
    assert result.times_s[-1] == 1801.0
    assert np.all(np.diff(result.times_s) > 0)
    assert result.snapshots[-1].displacement_m > 0


def test_disconnection_is_explicit_in_csv(tmp_path):
    result = run_analysis(replace(ScenarioConfig(), max_isl_distance_m=1.0, duration_s=60.0, maneuver_time_s=30.0))
    assert result.route is None
    assert all(snapshot.maneuvered.status == "unreachable" for snapshot in result.snapshots)
    write_results(result, tmp_path)
    with (tmp_path / "path_metrics.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "unreachable"
    assert row["propagation_delay_s"] == ""
    assert row["capacity_proxy_mbps"] == ""


def test_exports_are_repeatable_and_include_resolved_config(default_result, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = write_results(default_result, first)
    b = write_results(run_analysis(default_result.config), second)
    assert [path.name for path in a] == [path.name for path in b]
    for left, right in zip(a, b):
        assert left.read_bytes() == right.read_bytes()
    assert json.loads((first / "scenario.json").read_text())["backend"] == "scipy-j2"
    with (first / "link_metrics.csv").open(newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) > 1000


def test_cli_runs_from_an_unrelated_directory(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "leo_maneuver_analysis", "--config", str(EXAMPLE),
                                "--output", str(tmp_path / "outputs")], cwd=tmp_path,
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Fixed nominal route" in completed.stdout
    assert (tmp_path / "outputs" / "path_metrics.csv").exists()


@pytest.mark.parametrize("change", [
    {"sample_step_s": 0}, {"maneuver_time_s": -1}, {"maneuver_time_s": 2000},
    {"impulse_rtn_mps": [0, float("nan"), 0]}, {"maneuver_satellite_id": 99},
    {"max_retransmission_factor": 0.5}, {"backend": []}, {"packet_bits": True},
])
def test_invalid_configurations_fail(change):
    with pytest.raises(ValueError):
        replace(ScenarioConfig(), **change).validate()


def test_misspelled_config_field_has_actionable_cli_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"beam_width_radd": 0.1}')
    completed = subprocess.run([sys.executable, "-m", "leo_maneuver_analysis", "--config", str(path),
                                "--output", str(tmp_path / "outputs")], cwd=tmp_path,
                               text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "beam_width_radd" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "outputs").exists()
