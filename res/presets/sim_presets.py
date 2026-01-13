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

def _spawn_missile(jet, dist_r, lat_r, alt_r, speed_add_r):
    jf = _unit(np.array(getattr(jet, "velocity", np.array([1.0, 0.0, 0.0]))))
    Rj = np.array(getattr(jet, "orientation", np.eye(3)), dtype=float).reshape(3,3)
    jr = _unit(Rj @ np.array([0.0, 0.0, 1.0]))
    ju = _unit(Rj @ np.array([0.0, 1.0, 0.0]))
    dist = float(_rng.uniform(*dist_r))
    lat = float(_rng.uniform(-lat_r, lat_r))
    alt = 50.0
    pos = np.array(getattr(jet, "position", np.zeros(3)), dtype=float) - jf * dist + jr * lat + ju * alt
    pos[1] = 50.0
    jpos = np.array(getattr(jet, "position", np.zeros(3)), dtype=float)
    vdir = _unit(jpos - pos)
    js = float(np.linalg.norm(np.array(getattr(jet, "velocity", np.zeros(3)), dtype=float)))
    ms = float(max(80.0, js + _rng.uniform(*speed_add_r)))
    vel = vdir * ms
    m = ent_presets.create_PAC3(starting_position=pos, starting_velocity=vel, target_entity=jet)
    try:
        m.orientation = _R_from_forward(vel, 0.0)
    except Exception:
        pass
    return m

def create_phase1_scenario():
    jet = _spawn_jet((-6000.0, 6000.0), (-6000.0, 6000.0), (3500.0, 6500.0), (260.0, 360.0), (0.0, 360.0), (-3.0, 3.0), (-5.0, 5.0), 0.0)
    return [jet]

def create_phase2_scenario():
    jet = _spawn_jet((-12000.0, 12000.0), (-12000.0, 12000.0), (2500.0, 8000.0), (220.0, 480.0), (0.0, 360.0), (-10.0, 10.0), (-25.0, 25.0), 20.0)
    return [jet]

def create_phase3_scenario():
    jet = _spawn_jet((-16000.0, 16000.0), (-16000.0, 16000.0), (7000.0, 9000.0), (160.0, 560.0), (0.0, 360.0), (-20.0, 22.0), (-55.0, 55.0), 60.0)
    m1 = _spawn_missile(jet, (12000.0, 20000.0), 3500.0, (0.0, 0.0), (10,15))
    return [jet, m1]

def create_phase4_scenario():
    jet = _spawn_jet((-12000.0, 12000.0), (-12000.0, 12000.0), (2500.0, 8500.0), (240.0, 520.0), (0.0, 360.0), (-8.0, 8.0), (-20.0, 20.0), 15.0)
    m1 = _spawn_missile(jet, (12000.0, 20000.0), 3500.0, (0.0, 0.0), (80.0, 250.0))
    m2 = _spawn_missile(jet, (12000.0, 20000.0), 3500.0, (0.0, 0.0), (80.0, 250.0))
    return [jet, m1, m2]

def create_phase5_scenario():
    jet = _spawn_jet((-18000.0, 18000.0), (-18000.0, 18000.0), (1500.0, 9000.0), (220.0, 580.0), (0.0, 360.0), (-12.0, 12.0), (-35.0, 35.0), 25.0)
    m1 = _spawn_missile(jet, (12000.0, 20000.0), 3500.0, (0.0, 0.0), (80.0, 250.0))
    m2 = _spawn_missile(jet, (12000.0, 20000.0), 3500.0, (0.0, 0.0), (80.0, 250.0))
    m3 = _spawn_missile(jet, (12000.0, 20000.0), 3500.0, (0.0, 0.0), (80.0, 250.0))
    return [jet, m1, m2, m3]

def create_scenario(phase: int):
    p = int(phase)
    if p <= 1:
        return create_phase1_scenario()
    if p == 2:
        return create_phase2_scenario()
    if p == 3:
        return create_phase3_scenario()
    if p == 4:
        return create_phase4_scenario()
    return create_phase5_scenario()
