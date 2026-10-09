"""Validated parameters in SI units; no external catalogs or runtime globals."""

from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Integral, Real
from pathlib import Path


def positive(value: Real, name: str, *, zero_allowed: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not zero_allowed):
        raise ValueError(f"{name} must be {'non-negative' if zero_allowed else 'positive'}")


def integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class DynamicsConfig:
    backend: str = "scipy-j2"
    integrator_step_seconds: float = 30.0

    def validate(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in {"scipy-j2", "basilisk"}:
            raise ValueError(f"Unsupported propagation backend: {self.backend!r}")
        positive(self.integrator_step_seconds, "integrator_step_seconds")


@dataclass(frozen=True)
class ScenarioConfig:
    backend: str = "scipy-j2"
    num_planes: int = 3
    satellites_per_plane: int = 12
    altitude_m: float = 550_000.0
    inclination_deg: float = 53.0
    duration_s: float = 1_800.0
    sample_step_s: float = 30.0
    integrator_step_s: float = 30.0
    maneuver_time_s: float = 600.0
    maneuver_satellite_id: int = 3
    impulse_rtn_mps: tuple[float, float, float] = (0.0, 0.2, 0.0)
    source_satellite_id: int = 0
    destination_satellite_id: int = 6
    max_isl_distance_m: float = 5_014_000.0
    isl_degree_cap: int = 4
    nominal_capacity_mbps: float = 1_000.0
    beam_width_rad: float = 50e-6
    nominal_snr_linear: float = 100.0
    packet_bits: int = 12_000
    max_retransmission_factor: float = 20.0

    def __post_init__(self) -> None:
        if isinstance(self.impulse_rtn_mps, list):
            object.__setattr__(self, "impulse_rtn_mps", tuple(self.impulse_rtn_mps))

    @property
    def satellite_count(self) -> int:
        return self.num_planes * self.satellites_per_plane

    @property
    def dynamics(self) -> DynamicsConfig:
        return DynamicsConfig(self.backend, self.integrator_step_s)

    def validate(self) -> None:
        self.dynamics.validate()
        integer(self.num_planes, "num_planes", 1)
        integer(self.satellites_per_plane, "satellites_per_plane", 2)
        integer(self.isl_degree_cap, "isl_degree_cap", 1)
        integer(self.packet_bits, "packet_bits", 1)
        for name in ("altitude_m", "duration_s", "sample_step_s", "max_isl_distance_m",
                     "nominal_capacity_mbps", "beam_width_rad", "nominal_snr_linear",
                     "max_retransmission_factor"):
            positive(getattr(self, name), name)
        positive(self.maneuver_time_s, "maneuver_time_s", zero_allowed=True)
        positive(self.inclination_deg, "inclination_deg", zero_allowed=True)
        if self.inclination_deg > 180:
            raise ValueError("inclination_deg must lie in [0, 180]")
        if self.maneuver_time_s > self.duration_s:
            raise ValueError("maneuver_time_s cannot exceed duration_s")
        if self.max_retransmission_factor < 1:
            raise ValueError("max_retransmission_factor must be >= 1")
        for name in ("maneuver_satellite_id", "source_satellite_id", "destination_satellite_id"):
            integer(getattr(self, name), name)
            if getattr(self, name) >= self.satellite_count:
                raise ValueError(f"{name} must be less than satellite_count ({self.satellite_count})")
        if not isinstance(self.impulse_rtn_mps, (tuple, list)) or len(self.impulse_rtn_mps) != 3:
            raise ValueError("impulse_rtn_mps must contain three finite numbers")
        for value in self.impulse_rtn_mps:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("impulse_rtn_mps must contain three finite numbers")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ScenarioConfig:
    """Read a JSON object; omitted fields use defaults and unknown fields fail."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario configuration must be a JSON object")
    unknown = data.keys() - {field.name for field in fields(ScenarioConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    if isinstance(data.get("impulse_rtn_mps"), list):
        data["impulse_rtn_mps"] = tuple(data["impulse_rtn_mps"])
    config = ScenarioConfig(**data)
    config.validate()
    return config
