# LEO Maneuver Analysis

Python code for studying how a satellite maneuver affects inter-satellite pointing and communication along a fixed route.

The simulation starts from the same constellation and compares two cases: one without a maneuver and one with a velocity impulse. Both use the same route, and terminal pointing continues to follow the nominal trajectory.

[Run the example](#quick-start) · [Change the scenario](#change-the-scenario) · [Model notes](doc/model.md) · [Tests](#tests)

## Quick start

Requires **Python 3.10+**. From the repository directory, preferably in a virtual environment:

```bash
python -m pip install .
python -m leo_maneuver_analysis --config examples/maneuver_impact.json --output runs/demo
```

The example uses a synthetic constellation; no external orbital data is needed. Results are written to `runs/demo/`:

```text
runs/demo/
├── scenario.json              # Parameters used for the run
├── maneuver_trajectory.csv    # Nominal and maneuvered states of the affected satellite
├── link_metrics.csv           # Geometry, pointing error, and modeled link quality
└── path_metrics.csv           # Route availability and aggregate path metrics
```

For a local plot, install `".[plot]"` instead of `.` and add `--plot` to the run command. This also writes `maneuver_impact.png` to the output directory. `leo-maneuver` is an equivalent installed command.

## Example: a transverse impulse

The [example configuration](examples/maneuver_impact.json) places 36 satellites in three orbital planes. Satellite 3 receives a transverse impulse ten minutes into the simulation.

| Parameter | Value |
| :--- | :--- |
| Constellation | 3 planes × 12 satellites; half-slot phasing between adjacent planes |
| Altitude / inclination | 550 km / 53° |
| Maneuver | 0.2 m/s transverse impulse at 600 s |
| Duration / sampling | 1,800 s / 30 s |
| Source → destination | Satellite 0 → satellite 6 |
| Beam-width parameter | 50 µrad |

At the maneuver time, minimum-propagation-delay routing selects:

```text
0 → 1 → 2 → 3 → 4 → 5 → 6
```

After another 20 minutes, the maneuvered satellite is about **273 m** from its nominal position, and the maximum pointing error along the route is about **74 µrad**. The route remains geometrically connected, but its modeled capacity falls near zero because the terminals do not adjust their pointing for the maneuver.

These are results of the synthetic example, not measured terminal performance. The capacity and delay proxies are defined in the [model notes](doc/model.md#link-and-path-metrics).

## Change the scenario

Edit the JSON file to change the constellation, impulse, beam width, or endpoints. The impulse components are ordered **radial, transverse, normal**, in m/s.

The same analysis is available from Python:

```python
from dataclasses import replace
from leo_maneuver_analysis import load_config, run_analysis

config = load_config("examples/maneuver_impact.json")
config = replace(config, impulse_rtn_mps=(0.0, 0.1, 0.0))
result = run_analysis(config)

print(result.route)
print(result.snapshots[-1].displacement_m)
```

## Model and code

The default propagator uses Earth point-mass gravity plus J2. Links are checked for range and Earth clearance at each sample, but the candidate edges and route stay fixed. The pointing model assumes that terminals track the nominal line of sight without compensating for the maneuver.

This version does not model link reacquisition, adaptive routing, or traffic queues. See [model notes](doc/model.md) for the equations, units, finite-burn API, and optional Basilisk backend.

The main entry point is [`run_analysis`](src/leo_maneuver_analysis/analysis.py). Orbit propagation, topology and routing, and pointing calculations are separated into [`propagation.py`](src/leo_maneuver_analysis/propagation.py), [`network.py`](src/leo_maneuver_analysis/network.py), and [`pointing.py`](src/leo_maneuver_analysis/pointing.py).

## Tests

```bash
python -m pip install ".[dev]"
python -m pytest
```

Tests cover maneuver dynamics, topology constraints, routing, pointing sensitivity, path metrics, configuration errors, and CSV exports. Basilisk tests are skipped when the optional dependency is not installed.

---

**Haoyuan Zhao** · Adapted from satellite-network research code. No code license is included in this repository.
