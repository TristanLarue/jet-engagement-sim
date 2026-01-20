import numpy as np
from . import ent_presets

_rng = np.random.default_rng()

def _unit(v):
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)

def _R_from_forward(forward, roll_deg=0.0):
    f = _unit(np.array(forward, dtype=float).reshape(3))
    up0 = np.array([0.0, 1.0, 0.0], dtype=float)
    r = np.cross(f, up0)
    if float(np.linalg.norm(r)) < 1e-6:
        r = np.cross(f, np.array([0.0, 0.0, 1.0], dtype=float))
    r = _unit(r)
    u = _unit(np.cross(r, f))
    roll = float(np.deg2rad(roll_deg))
    if abs(roll) > 1e-12:
        c, s = float(np.cos(roll)), float(np.sin(roll))
        k = f
        def rot(v):
            return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)
        u, r = rot(u), rot(r)
    return np.column_stack((f, u, r))

def _R_from_ypr(yaw_deg, pitch_deg, roll_deg):
    yaw = float(np.deg2rad(yaw_deg))
    pitch = float(np.deg2rad(pitch_deg))
    f = np.array([np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw)], dtype=float)
    return _R_from_forward(f, roll_deg)

def _spawn_jet(xr, zr, ar, sr, yawr, pr, rr, omega_max_dps):
    R = _R_from_ypr(_rng.uniform(*yawr), _rng.uniform(*pr), _rng.uniform(*rr))
    v = (R @ np.array([1.0, 0.0, 0.0], dtype=float)) * float(_rng.uniform(*sr))
    p = np.array([_rng.uniform(*xr), _rng.uniform(*ar), _rng.uniform(*zr)], dtype=float)
    jet = ent_presets.create_Sukoi57(p, v, R)
    if omega_max_dps > 0.0:
        try:
            jet.omega = _rng.uniform(-omega_max_dps, omega_max_dps, size=3).astype(float)
        except Exception:
            pass
    return jet

def _spawn_missile_random(jet):
    """Spawn a missile at a truly random location on the X/Z plane at altitude 50.
    Missile will spawn upright with X+ axis pointing upward and face upward."""
    if jet is None:
        raise ValueError("Jet entity is None")
    
    # Random position on X/Z plane at altitude 50
    # Box size is -20000 to +20000 for both X and Z
    box_size = 20000.0
    pos = np.array([
        _rng.uniform(-box_size, box_size),
        50.0,  # Fixed altitude
        _rng.uniform(-box_size, box_size)
    ], dtype=float)
    
    # Velocity pointing upward (Y direction)
    # Random speed between 80 and 330 m/s
    ms = float(_rng.uniform(80.0, 330.0))
    vel = np.array([0.0, ms, 0.0], dtype=float)
    
    m = ent_presets.create_PAC3(starting_position=pos, starting_velocity=vel, target_entity=jet)
    try:
        # Orientation with X+ axis (forward) pointing upward
        # [0, 1, 0] is the upward direction (Y axis)
        m.orientation = _R_from_forward(np.array([0.0, 1.0, 0.0], dtype=float), 0.0)
    except Exception as e:
        # If orientation fails, missile should still work
        pass
    return m

def spawn_missile_dynamic(jet):
    """Reusable function to spawn a missile dynamically during simulation.
    Missile spawns upright with X+ axis pointing upward on X/Z plane at altitude 50."""
    return _spawn_missile_random(jet)

def create_phase1_scenario():
    """Phase 1: No missiles, basic flight training."""
    jet = _spawn_jet((-6000.0, 6000.0), (-6000.0, 6000.0), (3500.0, 6500.0), (260.0, 360.0), (0.0, 360.0), (-3.0, 3.0), (-5.0, 5.0), 0.0)
    return [jet]

def create_phase2_scenario():
    """Phase 2: 1 missile spawned randomly on X/Z plane at altitude 50."""
    jet = _spawn_jet((-12000.0, 12000.0), (-12000.0, 12000.0), (2500.0, 8000.0), (220.0, 480.0), (0.0, 360.0), (-10.0, 10.0), (-25.0, 25.0), 20.0)
    m1 = _spawn_missile_random(jet)
    return [jet, m1]

def create_phase3_scenario():
    """Phase 3: 3 missiles spawned randomly on X/Z plane at altitude 50."""
    jet = _spawn_jet((-16000.0, 16000.0), (-16000.0, 16000.0), (7000.0, 9000.0), (160.0, 560.0), (0.0, 360.0), (-20.0, 22.0), (-55.0, 55.0), 60.0)
    m1 = _spawn_missile_random(jet)
    m2 = _spawn_missile_random(jet)
    m3 = _spawn_missile_random(jet)
    return [jet, m1, m2, m3]

def create_scenario(phase: int):
    """Create a scenario based on phase (1, 2, or 3)."""
    p = max(1, min(3, int(phase)))  # Clamp to 1-3
    if p == 1:
        return create_phase1_scenario()
    elif p == 2:
        return create_phase2_scenario()
    else:  # p == 3
        return create_phase3_scenario()
