"""CSV exports and an optional, headless example figure."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from .analysis import AnalysisResult


def _write_csv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _path_rows(result: AnalysisResult):
    for snapshot in result.snapshots:
        for scenario in ("nominal", "maneuvered"):
            metrics = getattr(snapshot, scenario)
            row = asdict(metrics)
            row["path"] = " -> ".join(str(node) for node in metrics.path) if metrics.path is not None else None
            row.update(time_s=snapshot.time_s, scenario=scenario,
                       displacement_m=0.0 if scenario == "nominal" else snapshot.displacement_m,
                       max_path_pointing_angle_rad=0.0 if scenario == "nominal" and metrics.path is not None
                       else snapshot.path_pointing_angle_rad,
                       backend=result.backend, model=result.model)
            yield row


def _link_rows(result: AnalysisResult):
    for snapshot in result.snapshots:
        for scenario in ("nominal", "maneuvered"):
            for metrics in getattr(snapshot, f"{scenario}_links").values():
                yield dict(asdict(metrics), time_s=snapshot.time_s, scenario=scenario)


def _trajectory_rows(result: AnalysisResult):
    sat = result.config.maneuver_satellite_id
    for index, time_s in enumerate(result.times_s):
        row = {"time_s": float(time_s), "satellite_id": sat}
        for scenario in ("nominal", "maneuvered"):
            position = getattr(result, f"{scenario}_positions_m")[index, sat]
            velocity = getattr(result, f"{scenario}_velocities_mps")[index, sat]
            for axis, value in zip("xyz", position):
                row[f"{scenario}_{axis}_m"] = float(value)
            for axis, value in zip("xyz", velocity):
                row[f"{scenario}_v{axis}_mps"] = float(value)
        yield row


def plot_result(result: AnalysisResult, path: Path) -> None:
    """Render only the fixed-path pointing error and capacity proxy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Plotting requires `python -m pip install '.[plot]'`.") from exc

    times = result.times_s / 60.0
    pointing = np.array([snapshot.path_pointing_angle_rad for snapshot in result.snapshots], dtype=float) * 1e6
    nominal = np.array([snapshot.nominal.capacity_proxy_mbps for snapshot in result.snapshots], dtype=float)
    maneuvered = np.array([snapshot.maneuvered.capacity_proxy_mbps for snapshot in result.snapshots], dtype=float)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True, constrained_layout=True)
        axes[0].plot(times, pointing, color="#b65a24", linewidth=2, label="Maximum along fixed path")
        axes[0].axhline(result.config.beam_width_rad * 1e6, color="#666666", linestyle="--",
                       linewidth=1, label="Model beam width")
        axes[0].set_ylabel("Pointing error (µrad)")
        axes[0].legend(loc="upper left", frameon=False)
        axes[1].plot(times, nominal / result.config.nominal_capacity_mbps,
                     color="#286aa6", linewidth=2, label="No maneuver")
        axes[1].plot(times, maneuvered / result.config.nominal_capacity_mbps,
                     color="#b65a24", linewidth=2, label="Prescribed maneuver")
        axes[1].set_ylabel("Path capacity proxy\n(normalized)")
        axes[1].set_xlabel("Time since initial state (min)")
        axes[1].set_ylim(-0.04, 1.08)
        axes[1].legend(loc="lower left", frameon=False)
        for axis in axes:
            axis.axvline(result.config.maneuver_time_s / 60.0, color="#888888", linestyle=":", linewidth=1)
            axis.grid(alpha=0.18)
        if not np.any(np.isfinite(nominal)):
            axes[1].text(0.5, 0.5, "No ISL capacity available for selected route",
                         ha="center", va="center", transform=axes[1].transAxes)
        figure.suptitle("Prescribed maneuver with nominal-trajectory pointing", fontsize=12)
        figure.savefig(path, dpi=160, metadata={"Software": "LEO Maneuver Analysis"})
        plt.close(figure)


def write_results(result: AnalysisResult, output_dir: str | Path, *, plot: bool = False) -> tuple[Path, ...]:
    """Export files after staging them successfully; None becomes a blank CSV cell."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ["scenario.json", "path_metrics.csv", "link_metrics.csv", "maneuver_trajectory.csv"]
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as staging_dir:
        staging = Path(staging_dir)
        (staging / "scenario.json").write_text(json.dumps(result.config.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_csv(staging / "path_metrics.csv", [
            "time_s", "scenario", "status", "path", "propagation_delay_s", "retry_weighted_delay_s",
            "capacity_proxy_mbps", "packet_error_rate", "displacement_m", "max_path_pointing_angle_rad",
            "backend", "model"], _path_rows(result))
        _write_csv(staging / "link_metrics.csv", [
            "time_s", "scenario", "a", "b", "available", "distance_m", "pointing_angle_rad",
            "pointing_gain", "propagation_delay_s", "retry_weighted_delay_s",
            "capacity_proxy_mbps", "packet_error_rate"], _link_rows(result))
        _write_csv(staging / "maneuver_trajectory.csv", ["time_s", "satellite_id"] + [
            f"{scenario}_{quantity}" for scenario in ("nominal", "maneuvered")
            for quantity in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")], _trajectory_rows(result))
        if plot:
            plot_result(result, staging / "maneuver_impact.png")
            names.append("maneuver_impact.png")
        for name in names:
            (staging / name).replace(output / name)
    return tuple(output / name for name in names)
