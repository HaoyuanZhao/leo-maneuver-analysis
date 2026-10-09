# Model notes

[← README](../README.md)

## Scenario and dynamics

The example starts with a synthetic circular constellation, with half-slot phasing between adjacent planes. Satellite IDs index the state arrays. Positions and velocities use a common Earth-centered inertial frame and have shape `(time, satellite, xyz)`.

Units are meters, seconds, m/s, radians, and Mbps. `inclination_deg` is the exception: it is specified in degrees. The time grid includes the maneuver time and final time, even when they are not multiples of `sample_step_s`.

The default `scipy-j2` propagator integrates Earth point-mass gravity plus J2 with constant spacecraft mass. At the configured maneuver time, the RTN impulse is converted to the inertial frame and added to the selected satellite's velocity. Its position is continuous across the impulse. The remaining satellites retain their nominal trajectories.

The propagation API also supports finite burns with constant thrust, fixed inertial direction, and constant mass. Burn and coast segments are integrated separately. The JSON scenario and CLI use an impulse; finite burns are available through `propagate_finite_burn` in [`propagation.py`](../src/leo_maneuver_analysis/propagation.py).

Attitude dynamics, atmospheric drag, and propellant depletion are not modeled. Maneuvers are supplied as inputs rather than inferred from orbital records.

## Topology and route selection

At the maneuver time, the nominal constellation defines the candidate links and route. Candidate links join slot neighbors within each plane and nearest neighbors on adjacent planes. Greedy edge selection prioritizes intra-plane links, then distance and satellite IDs, subject to range, spherical-Earth clearance, and degree limits.

Each satellite has at most two intra-plane links and two cross-plane links, also subject to `isl_degree_cap`. The route minimizes nominal propagation delay. Both the selected edge set and route are held fixed for the rest of the comparison; range and Earth clearance are checked again at every sample.

Ground links, terminal scheduling, link acquisition, and adaptive routing are not included.

## Link and path metrics

At each sample, the pointing error `θ` is the angle between the nominal and actual line-of-sight vectors. Terminals follow the nominal trajectory without correcting for the maneuver.

For beam-width parameter `β` and nominal linear SNR `SNR₀`, [`pointing.py`](../src/leo_maneuver_analysis/pointing.py) uses:

```text
g   = exp(-2(θ / β)²)
SNR = SNR₀ × g²
BER = 0.5 × erfc(sqrt(SNR))
PER = 1 - (1 - BER)^packet_bits
```

Packet errors assume independent bit errors. This is an analytical sensitivity model, not a calibrated optical-terminal model.

For a geometrically available link, the capacity proxy is `nominal_capacity_mbps × max(1 - PER, 1e-12)`. Path capacity is the minimum link capacity, and path packet success is the product of per-hop success probabilities. Shared-link contention, queues, and packet-level simulation are not included.

Geometric propagation delay is distance divided by the speed of light, summed along the path. Retry-weighted delay multiplies each link's propagation delay by `min(max_retransmission_factor, 1 / max(1 - PER, 1e-12))`. The example caps this factor at 20. This quantity is a diagnostic proxy, not application latency or an ARQ timing model. Finite proxy values do not imply usable service when packet error is near one.

Path status distinguishes three cases:

| Status | Meaning |
| :--- | :--- |
| `connected` | Every link on the selected route is geometrically available; modeled capacity may still be near zero. |
| `unavailable` | A selected route exists, but at least one link fails the current geometry checks. |
| `unreachable` | No nominal route was found at the maneuver time. |

Unavailable path metrics are blank in the CSV files, not zero.

## Optional Basilisk backend

Install the optional dependency and select it when running the example:

```bash
python -m pip install ".[basilisk]"
python -m leo_maneuver_analysis --config examples/maneuver_impact.json --output runs/basilisk --backend basilisk
```

This adapter uses Basilisk's spacecraft module with Earth point-mass gravity and constant mass. It does **not** include J2, so its results are not numerically equivalent to the default SciPy backend. Each run records the backend and model. A requested but missing Basilisk installation produces an error rather than falling back to SciPy.

Sample times and finite-burn durations must be multiples of the fixed integration step. With the CLI, `sample_step_s`, `maneuver_time_s`, and `duration_s` must align with `integrator_step_s`. Use a smaller step for finer timing.

The dependency is pinned to `bsk==2.11.1` in [`pyproject.toml`](../pyproject.toml). See the [Basilisk installation guide](https://avslab.github.io/basilisk/Install.html) for platform and installation details.

## Files and configuration

Configuration validation is in [`config.py`](../src/leo_maneuver_analysis/config.py); CSV export and optional plotting are in [`export.py`](../src/leo_maneuver_analysis/export.py). Paths supplied to the CLI are relative to the working directory. Once installed, the package can be called from outside the repository using appropriate paths.

`scenario.json` includes any backend override. Reusing an output directory replaces the named exports after they have been staged successfully. Generated files under `runs/` are ignored by Git.

The [test suite](../tests/) covers RTN geometry, zero-maneuver consistency, impulse continuity, finite burns, topology constraints, weighted routing, disconnection, pointing sensitivity, path aggregation, invalid configurations, and CLI exports. The [CI workflow](../.github/workflows/checks.yml) is configured for Python 3.10 and 3.12 with the default backend.
