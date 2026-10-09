"""Circular initial states for a synthetic, single-shell constellation."""

from dataclasses import dataclass
import math

import numpy as np

from .config import ScenarioConfig
from .propagation import EARTH_MU_M3_S2, EARTH_RADIUS_M


@dataclass(frozen=True)
class Satellite:
    id: int
    plane: int
    slot: int


def initial_constellation(config: ScenarioConfig) -> tuple[tuple[Satellite, ...], np.ndarray, np.ndarray]:
    """Return metadata and N x 3 position/velocity arrays in an inertial frame.

    RAAN is evenly spaced over 360 degrees. Adjacent planes have a half-slot
    phase offset. These states describe a synthetic geometry, not a catalog.
    """
    config.validate()
    radius = EARTH_RADIUS_M + config.altitude_m
    speed = math.sqrt(EARTH_MU_M3_S2 / radius)
    inclination = math.radians(config.inclination_deg)
    ci, si = math.cos(inclination), math.sin(inclination)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    satellites, positions, velocities = [], [], []
    for plane in range(config.num_planes):
        raan = plane * 2.0 * math.pi / config.num_planes
        cr, sr = math.cos(raan), math.sin(raan)
        rotation = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]) @ rx
        for slot in range(config.satellites_per_plane):
            phase = 2.0 * math.pi * (slot + 0.5 * plane) / config.satellites_per_plane
            cp, sp = math.cos(phase), math.sin(phase)
            satellites.append(Satellite(len(satellites), plane, slot))
            positions.append(rotation @ np.array([radius * cp, radius * sp, 0.0]))
            velocities.append(rotation @ np.array([-speed * sp, speed * cp, 0.0]))
    return tuple(satellites), np.array(positions), np.array(velocities)
