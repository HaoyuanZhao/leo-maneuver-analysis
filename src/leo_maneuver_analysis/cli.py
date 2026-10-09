"""Command-line entry point for a local JSON scenario."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

from .analysis import run_analysis
from .config import load_config
from .export import write_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a prescribed LEO maneuver and its fixed ISL path.")
    parser.add_argument("--config", type=Path, required=True, help="Local scenario JSON file")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"), help="Directory for CSV outputs")
    parser.add_argument("--backend", choices=("scipy-j2", "basilisk"), help="Override the configured backend")
    parser.add_argument("--plot", action="store_true", help="Also export a PNG figure (requires plot extra)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        config = load_config(args.config)
        if args.backend is not None:
            config = replace(config, backend=args.backend)
        result = run_analysis(config)
        files = write_results(result, args.output, plot=args.plot)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    route = " -> ".join(map(str, result.route)) if result.route is not None else "unreachable"
    final = result.snapshots[-1]
    capacity = final.maneuvered.capacity_proxy_mbps
    print(f"Backend: {result.backend}; model: {result.model}")
    print(f"Satellites: {len(result.satellites)}; snapshots: {len(result.times_s)}")
    print(f"Fixed nominal route: {route}")
    print(f"Final displacement: {final.displacement_m:.3f} m")
    print(f"Final path geometry: {final.maneuvered.status}")
    print(f"Final path capacity proxy: {capacity:.6g} Mbps" if capacity is not None else "Final path capacity proxy: unavailable")
    print(f"Wrote {len(files)} files to {args.output} in {time.perf_counter() - started:.2f} s")
    return 0
