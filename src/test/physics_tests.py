"""
physics_tests.py

Pytest unit tests for physics.py.

Goal: high confidence regression suite with branch/path coverage aligned with the
decision points in physics.py (i.e., cyclomatic-complexity driven test design).

Run (recommended, if you have pytest-cov installed):
    pytest -q --cov=physics --cov-branch --cov-report=term-missing physics_tests.py

Run (without pytest-cov):
    pytest -q physics_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the folder containing physics.py is importable when running pytest from elsewhere.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import physics  # noqa: E402


# -------------------------
# Helpers
# -------------------------
def rot_z(deg: float) -> np.ndarray:
    """Right-handed rotation about Z (yaw). Returns body->world rotation matrix."""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot_y(deg: float) -> np.ndarray:
    """Right-handed rotation about Y (pitch). Returns body->world rotation matrix."""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.copy()
    return v / n


# -------------------------
# get_air_density
# -------------------------
def test_get_air_density_sea_level() -> None:
    # At 0 m: should be exactly sea-level density (no clipping needed).
    assert physics.get_air_density(0.0) == pytest.approx(1.225, rel=0.0, abs=1e-12)


def test_get_air_density_negative_altitude_clips_to_sea_level() -> None:
    # Negative altitude => density formula would exceed sea-level, but function clips to sea_level_density.
    assert physics.get_air_density(-1000.0) == pytest.approx(1.225, rel=0.0, abs=1e-12)


def test_get_air_density_very_high_altitude_temperature_floor() -> None:
    # High altitude makes temperature negative -> floored to 0 -> density becomes 0 (and clipped).
    assert physics.get_air_density(50_000.0) == pytest.approx(0.0, rel=0.0, abs=1e-12)


# -------------------------
# get_angle_of_attack / get_sideslip
# -------------------------
@pytest.mark.parametrize(
    "velocity,R,expected",
    [
        (np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0),  # zero-velocity early return
        (np.array([1.0, 0.0, 0.0]), np.eye(3), 0.0),
        (np.array([1.0, 1.0, 0.0]), np.eye(3), -45.0),
        (np.array([0.0, 1.0, 0.0]), np.eye(3), -90.0),
        # With a rotated body: body forward is world +Y when Rz(+90). World +Y -> body +X => aoa 0.
        (np.array([0.0, 1.0, 0.0]), rot_z(90.0), 0.0),
    ],
)
def test_get_angle_of_attack(velocity: np.ndarray, R: np.ndarray, expected: float) -> None:
    assert physics.get_angle_of_attack(velocity, R) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "velocity,R,expected",
    [
        (np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0),  # zero-velocity early return
        (np.array([1.0, 0.0, 0.0]), np.eye(3), 0.0),
        (np.array([1.0, 0.0, 1.0]), np.eye(3), 45.0),
        (np.array([0.0, 0.0, 1.0]), np.eye(3), 90.0),
        # Rotated body: body forward is world +Y when Rz(+90). World +Z remains +Z in body => sideslip 90.
        (np.array([0.0, 1.0, 1.0]), rot_z(90.0), pytest.approx(45.0, abs=1e-9)),
    ],
)
def test_get_sideslip(velocity: np.ndarray, R: np.ndarray, expected) -> None:
    assert physics.get_sideslip(velocity, R) == expected


# -------------------------
# get_lift_coefficient (branch coverage)
# -------------------------
@pytest.mark.parametrize(
    "aoa,expected_desc,expected",
    [
        # Normal in-range linear region
        (0.0, "linear region", 0.1411764705882353),
        # Clamp to cl_min inside stall region
        (-15.0, "clamped to cl_min", -0.9),
        # At stall_pos -> cl_max
        (15.0, "cl_max at stall_pos", 1.2),
        # a normalization: aoa=170 => a becomes -10 (hits a>90 branch)
        (170.0, "normalized from >90", -0.5647058823529412),
        # a normalization: aoa=-170 => a becomes +10 (hits a<-90 branch)
        (-170.0, "normalized from <-90", 0.8470588235294118),
        # Post-stall decay for a in (stall_pos, 20]
        (18.0, "post-stall quadratic decay (<=20)", 1.056),
        # Deep post-stall for a > 20
        (50.0, "deep post-stall (>20)", 0.4571428571428572),
        # Negative side, moderate (a >= -20) with quadratic recovery
        (-18.0, "negative post-stall (>=-20)", -0.81),
        # Negative deep post-stall (a < -20)
        (-50.0, "negative deep post-stall (<-20)", -0.37142857142857144),
    ],
)
def test_get_lift_coefficient_branches(aoa: float, expected_desc: str, expected: float) -> None:
    # Note: max_lift_coefficient / optimal_lift_aoa are currently unused by the implementation.
    got = physics.get_lift_coefficient(aoa=aoa, max_lift_coefficient=999.0, optimal_lift_aoa=123.0)
    assert got == pytest.approx(expected, abs=1e-9), expected_desc


# -------------------------
# get_drag_coefficient
# -------------------------
def test_get_drag_coefficient_endpoints() -> None:
    min_cd, max_cd = 0.02, 0.12
    assert physics.get_drag_coefficient(0.0, min_cd, max_cd) == pytest.approx(min_cd, abs=1e-12)
    assert physics.get_drag_coefficient(90.0, min_cd, max_cd) == pytest.approx(max_cd, abs=1e-12)


# -------------------------
# Forces
# -------------------------
def test_get_gravity_force() -> None:
    f = physics.get_gravity_force(10.0)
    assert np.allclose(f, np.array([0.0, -98.1, 0.0]))


def test_get_thrust_force_direction_and_scale() -> None:
    R = np.eye(3)
    f = physics.get_thrust_force(R, throttle=0.5, thrust_force=100.0, length=10.0)
    assert np.allclose(f, np.array([50.0, 0.0, 0.0]))

    # Rotate body so forward points world +Y
    R = rot_z(90.0)
    f = physics.get_thrust_force(R, throttle=1.0, thrust_force=20.0, length=1.0)
    assert np.allclose(f, np.array([0.0, 20.0, 0.0]), atol=1e-12)


def test_get_drag_force_zero_velocity_branch() -> None:
    f = physics.get_drag_force(np.zeros(3), air_density=1.225, reference_area=1.0, drag_coefficient=0.1)
    assert np.allclose(f, np.zeros(3))


def test_get_drag_force_opposes_velocity() -> None:
    v = np.array([10.0, 0.0, 0.0])
    rho, area, cd = 1.0, 2.0, 0.5
    f = physics.get_drag_force(v, air_density=rho, reference_area=area, drag_coefficient=cd)
    # Should point opposite to velocity
    assert np.allclose(unit(f), np.array([-1.0, 0.0, 0.0]), atol=1e-12)

    vmag = 10.0
    expected_mag = 0.5 * rho * (vmag**2) * area * cd
    assert float(np.linalg.norm(f)) == pytest.approx(expected_mag, abs=1e-12)


def test_get_lift_force_uses_body_up_axis() -> None:
    v = np.array([10.0, 0.0, 0.0])
    rho, area, cl = 1.0, 2.0, 1.5

    # Identity: body up is +Y
    R = np.eye(3)
    f = physics.get_lift_force(v, reference_area=area, max_lift_coefficient=cl, R=R, air_density=rho)
    assert np.allclose(unit(f), np.array([0.0, 1.0, 0.0]), atol=1e-12)

    # Rotate 90° about Z: body up maps to world -X
    R = rot_z(90.0)
    f = physics.get_lift_force(v, reference_area=area, max_lift_coefficient=cl, R=R, air_density=rho)
    assert np.allclose(unit(f), np.array([-1.0, 0.0, 0.0]), atol=1e-12)


# -------------------------
# Placeholder stubs (currently pass)
# -------------------------
@pytest.mark.parametrize(
    "fn_name",
    [
        "get_lift_force",
        "get_sideforce_force",
        "get_elevator_force",
        "get_aileron_force",
        "get_rudder_force",
    ],
)
def test_placeholder_functions_currently_return_none(fn_name: str) -> None:
    fn = getattr(physics, fn_name)
    assert fn() is None


# -------------------------
# Direction helpers
# -------------------------
def test_direction_helpers_identity() -> None:
    R = np.eye(3)
    assert np.allclose(physics.get_forward_dir(R), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(physics.get_up_dir(R), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(physics.get_right_dir(R), np.array([0.0, 0.0, 1.0]))


def test_direction_helpers_rotated() -> None:
    R = rot_y(90.0)  # forward becomes world -Z (with this convention), right becomes world +X
    assert np.allclose(physics.get_forward_dir(R), np.array([0.0, 0.0, -1.0]), atol=1e-12)
    assert np.allclose(physics.get_right_dir(R), np.array([1.0, 0.0, 0.0]), atol=1e-12)


# -------------------------
# get_lift_dir (branch coverage)
# -------------------------
def test_get_lift_dir_small_velocity_returns_up() -> None:
    R = np.eye(3)
    v = np.array([0.0, 0.0, 0.0])
    d = physics.get_lift_dir(v, R)
    assert np.allclose(d, physics.get_up_dir(R))


def test_get_lift_dir_general_case_projection() -> None:
    # Velocity along +X => airflow along -X; up is +Y => lift_dir should be +Y.
    R = np.eye(3)
    v = np.array([10.0, 0.0, 0.0])
    d = physics.get_lift_dir(v, R)
    assert np.allclose(d, np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert float(np.linalg.norm(d)) == pytest.approx(1.0, abs=1e-12)


def test_get_lift_dir_parallel_up_triggers_cross_with_right() -> None:
    # Choose velocity so airflow aligns with up (+Y) to force lift_norm ~ 0.
    # With identity R: up=(0,1,0), right=(0,0,1), cross(+Y,+Z)=+X.
    R = np.eye(3)
    v = np.array([0.0, -10.0, 0.0])  # airflow = +Y
    d = physics.get_lift_dir(v, R)
    assert np.allclose(d, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert float(np.linalg.norm(d)) == pytest.approx(1.0, abs=1e-12)


def test_get_lift_dir_double_degenerate_falls_back_to_up() -> None:
    # Use a deliberately non-orthonormal "R" so that up_dir == right_dir,
    # and pick velocity so airflow aligns with up/right; then both projection and cross fail.
    R = np.eye(3)
    R[:, 2] = R[:, 1]  # right column == up column => right_dir == up_dir == +Y
    v = np.array([0.0, -1.0, 0.0])  # airflow = +Y, parallel to up & right
    d = physics.get_lift_dir(v, R)
    assert np.allclose(d, physics.get_up_dir(R), atol=1e-12)


# -------------------------
# get_omega
# -------------------------
def test_get_omega_identity_simple_cross() -> None:
    R = np.eye(3)
    force = np.array([0.0, 10.0, 0.0])
    application_point = np.array([1.0, 0.0, 0.0])
    moi = np.array([2.0, 5.0, 10.0])
    omega = physics.get_omega(force, R, application_point, moi)

    # torque = r x F = (1,0,0) x (0,10,0) = (0,0,10); omega = torque/moi = (0,0,1)
    assert np.allclose(omega, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_get_omega_rotated_frame() -> None:
    # Rotate 90° about Z: body X->world Y. We validate the R.T application affects torque.
    R = rot_z(90.0)
    force_world = np.array([10.0, 0.0, 0.0])
    r_world = np.array([0.0, 1.0, 0.0])
    moi = np.array([1.0, 2.0, 4.0])

    # torque_world = r x F = (0,1,0) x (10,0,0) = (0,0,-10)
    torque_world = np.cross(r_world, force_world)
    # torque_body = R.T @ torque_world
    torque_body = R.T @ torque_world
    expected = torque_body / moi

    got = physics.get_omega(force_world, R, r_world, moi)
    assert np.allclose(got, expected, atol=1e-12)