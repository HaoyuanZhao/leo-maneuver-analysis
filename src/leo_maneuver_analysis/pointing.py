from __future__ import annotations

import math

import numpy as np

from .config import integer, positive


def los_pointing_angle_rad(
    nominal_position_a_m: np.ndarray,
    nominal_position_b_m: np.ndarray,
    effective_position_a_m: np.ndarray,
    effective_position_b_m: np.ndarray,
) -> float:
    for position in (nominal_position_a_m, nominal_position_b_m,
                     effective_position_a_m, effective_position_b_m):
        array = np.asarray(position, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("link positions must be finite 3-vectors")
    nominal_los = np.asarray(nominal_position_b_m, dtype=float) - np.asarray(
        nominal_position_a_m, dtype=float
    )
    effective_los = np.asarray(effective_position_b_m, dtype=float) - np.asarray(
        effective_position_a_m, dtype=float
    )
    denominator = float(np.linalg.norm(nominal_los) * np.linalg.norm(effective_los))
    if denominator <= 0:
        raise ValueError("line-of-sight vectors must be non-zero")
    if np.array_equal(nominal_los, effective_los):
        return 0.0
    cosine = float(np.dot(nominal_los, effective_los)) / denominator
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def pointing_link_quality(
    angle_rad: float,
    *,
    beam_width_rad: float,
    nominal_snr_linear: float,
    packet_bits: int,
    max_retransmission_factor: float,
) -> dict[str, float]:
    positive(angle_rad, "angle_rad", zero_allowed=True)
    positive(beam_width_rad, "beam_width_rad")
    positive(nominal_snr_linear, "nominal_snr_linear")
    integer(packet_bits, "packet_bits", 1)
    positive(max_retransmission_factor, "max_retransmission_factor")
    if max_retransmission_factor < 1:
        raise ValueError("max_retransmission_factor must be >= 1")
    angle = float(angle_rad)
    relative_angle = angle / beam_width_rad
    pointing_gain = float(math.exp(-2.0 * relative_angle ** 2)) if relative_angle < 30 else 0.0
    snr_linear = float(nominal_snr_linear * pointing_gain**2)
    ber = float(0.5 * math.erfc(math.sqrt(max(snr_linear, 0.0))))
    if ber <= 0:
        packet_error_rate = 0.0
    elif ber >= 1:
        packet_error_rate = 1.0
    else:
        packet_error_rate = float(-math.expm1(packet_bits * math.log1p(-ber)))
    packet_success = max(1.0 - packet_error_rate, 1e-12)
    retransmission_factor = float(min(max_retransmission_factor, 1.0 / packet_success))
    return {
        "pointing_angle_rad": angle,
        "pointing_gain": pointing_gain,
        "snr_linear": snr_linear,
        "ber": ber,
        "packet_error_rate": packet_error_rate,
        "throughput_factor": packet_success,
        "retransmission_factor": retransmission_factor,
    }
