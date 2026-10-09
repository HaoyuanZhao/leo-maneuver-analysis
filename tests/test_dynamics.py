import numpy as np
import pytest

from leo_maneuver_analysis.propagation import EARTH_MU_M3_S2, ScipyJ2Propagator, rtn_basis


@pytest.fixture
def state():
    radius = 7_000_000.0
    return np.array([radius, 0.0, 0.0]), np.array([0.0, np.sqrt(EARTH_MU_M3_S2 / radius), 0.0])


def test_rtn_is_right_handed_and_orthonormal(state):
    basis = rtn_basis(*state)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    np.testing.assert_allclose(basis[:, 0], state[0] / np.linalg.norm(state[0]))


def test_impulse_has_continuous_position_and_expected_velocity_jump(state):
    propagator = ScipyJ2Propagator()
    times = np.array([0.0, 60.0, 120.0])
    impulse = np.array([0.0, 1.0, 0.0])
    nominal = propagator.propagate(*state, times)
    moved = propagator.propagate(*state, times, impulse_eci_mps=impulse)
    np.testing.assert_array_equal(nominal.positions_m[0], moved.positions_m[0])
    np.testing.assert_allclose(moved.velocities_mps[0] - nominal.velocities_mps[0], impulse, atol=1e-12)
    assert np.linalg.norm(moved.positions_m[-1] - nominal.positions_m[-1]) > 100.0


def test_zero_duration_burn_equals_coasting(state):
    propagator = ScipyJ2Propagator()
    times = np.array([0.0, 30.0, 60.0])
    coast = propagator.propagate(*state, times)
    finite = propagator.propagate_finite_burn(*state, times, thrust_direction_eci=np.array([0.0, 1.0, 0.0]),
                                             burn_duration_s=0.0, thrust_n=1.0, mass_kg=100.0)
    np.testing.assert_array_equal(coast.positions_m, finite.positions_m)
    np.testing.assert_array_equal(coast.velocities_mps, finite.velocities_mps)


def test_short_finite_burn_has_expected_small_time_displacement(state):
    propagator = ScipyJ2Propagator(max_step_s=0.1)
    times = np.array([0.0, 1.0, 2.0])
    coast = propagator.propagate(*state, times)
    finite = propagator.propagate_finite_burn(*state, times, thrust_direction_eci=np.array([0.0, 1.0, 0.0]),
                                             burn_duration_s=1.0, thrust_n=10.0, mass_kg=100.0)
    # a=0.1 m/s²: 0.05 m during burn, followed by 0.1 m in one second of coasting.
    assert finite.positions_m[-1, 1] - coast.positions_m[-1, 1] == pytest.approx(0.15, abs=5e-4)


@pytest.mark.parametrize("times", [[0.0, 0.0, 30.0], [30.0, 0.0], [-1.0], [float("nan")], []])
def test_invalid_sample_times_fail(state, times):
    with pytest.raises(ValueError, match="sample_times_s"):
        ScipyJ2Propagator().propagate(*state, np.array(times))


def test_degenerate_rtn_and_invalid_step_fail(state):
    with pytest.raises(ValueError, match="non-zero"):
        rtn_basis(np.zeros(3), state[1])
    with pytest.raises(ValueError, match="collinear"):
        rtn_basis(state[0], state[0])
    with pytest.raises(ValueError, match="max_step_s"):
        ScipyJ2Propagator(max_step_s=float("nan"))


@pytest.mark.basilisk
def test_basilisk_circular_orbit_and_impulse(state):
    pytest.importorskip("Basilisk")
    from leo_maneuver_analysis.propagation import BasiliskPropagator
    propagator = BasiliskPropagator(step_s=1.0)
    times = np.array([0.0, 30.0, 60.0])
    nominal = propagator.propagate(*state, times)
    omega = np.sqrt(EARTH_MU_M3_S2 / np.linalg.norm(state[0]) ** 3)
    analytic = np.array([np.cos(omega * times[-1]), np.sin(omega * times[-1]), 0.0]) * np.linalg.norm(state[0])
    np.testing.assert_allclose(nominal.positions_m[-1], analytic, atol=0.01)
    moved = propagator.propagate(*state, times, impulse_eci_mps=np.array([0.0, 0.2, 0.0]))
    np.testing.assert_allclose(moved.velocities_mps[0] - nominal.velocities_mps[0], [0.0, 0.2, 0.0], atol=1e-10)
    assert np.linalg.norm(moved.positions_m[-1] - nominal.positions_m[-1]) > 10.0


@pytest.mark.basilisk
def test_basilisk_finite_burn_and_single_initial_sample(state):
    pytest.importorskip("Basilisk")
    from leo_maneuver_analysis.propagation import BasiliskPropagator
    propagator = BasiliskPropagator(step_s=0.1)
    initial = propagator.propagate(*state, np.array([0.0]))
    np.testing.assert_array_equal(initial.positions_m[0], state[0])
    times = np.array([0.0, 1.0, 2.0])
    coast = propagator.propagate(*state, times)
    burned = propagator.propagate_finite_burn(*state, times, thrust_direction_eci=np.array([0.0, 1.0, 0.0]),
                                              burn_duration_s=1.0, thrust_n=10.0, mass_kg=100.0)
    assert burned.positions_m[-1, 1] - coast.positions_m[-1, 1] == pytest.approx(0.15, abs=0.02)


@pytest.mark.basilisk
def test_basilisk_rejects_off_grid_times_instead_of_clamping_states(state):
    pytest.importorskip("Basilisk")
    from leo_maneuver_analysis.propagation import BasiliskPropagator
    propagator = BasiliskPropagator(step_s=30.0)
    with pytest.raises(ValueError, match="multiples of step_s"):
        propagator.propagate(*state, np.array([0.0, 45.0]))
    with pytest.raises(ValueError, match="burn duration"):
        propagator.propagate_finite_burn(*state, np.array([0.0, 30.0, 60.0]),
                                         thrust_direction_eci=np.ones(3), burn_duration_s=45.0,
                                         thrust_n=1.0, mass_kg=100.0)
