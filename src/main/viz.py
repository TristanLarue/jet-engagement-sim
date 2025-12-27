def _bar_01(x: float, width: int = 20) -> str:
    x = float(_clamp(x, 0.0, 1.0))
    fill = int(round(x * width))
    bar = "=" * fill + "-" * (width - fill)
    return f"|{bar}|"
import vpython as vp
import numpy as np
from typing import Optional
import physics

# ==========================
# Global variables
# ==========================
scene: Optional[vp.canvas] = None
_static_scene_objects: list = []
_controlled_jet = None
_EPS = 1e-12
_keys_down = set()

# ==========================
# Small math helpers
# ==========================
def _wrap_deg(a: float) -> float:
    x = (a + 180.0) % 360.0 - 180.0
    return 180.0 if abs(x + 180.0) < 1e-9 else x

def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = _safe_norm(v)
    if n < _EPS:
        return fallback
    return v / n

def _np3(x) -> Optional[np.ndarray]:
    if x is None:
        return None

    if hasattr(x, "x") and hasattr(x, "y") and hasattr(x, "z"):
        return np.array([float(x.x), float(x.y), float(x.z)], dtype=float)

    try:
        a = np.array(x, dtype=float).reshape(-1)
        if a.size < 3:
            return None
        return a[:3].astype(float)
    except Exception:
        return None

def _vpvec(x: np.ndarray) -> vp.vector:
    return vp.vector(float(x[0]), float(x[1]), float(x[2]))

def _extract_position(entity) -> Optional[np.ndarray]:
    if hasattr(entity, "position"):
        return _np3(entity.position)
    if hasattr(entity, "p"):
        return _np3(entity.p)
    return None

def _extract_velocity(entity) -> Optional[np.ndarray]:
    for name in ("velocity", "vel", "v"):
        if hasattr(entity, name):
            v = _np3(getattr(entity, name))
            if v is not None:
                return v
    return None

def _extract_orientation(entity) -> np.ndarray:
    if not hasattr(entity, "orientation"):
        return np.eye(3, dtype=float)
    try:
        R = np.asarray(entity.orientation, dtype=float)
        if R.shape != (3, 3):
            return np.eye(3, dtype=float)
        return R
    except Exception:
        return np.eye(3, dtype=float)

def _compute_rpy_deg_from_R(R: np.ndarray) -> tuple[float, float, float]:
    f = R @ np.array([1.0, 0.0, 0.0], dtype=float)
    u = R @ np.array([0.0, 1.0, 0.0], dtype=float)

    f = _safe_unit(f, np.array([1.0, 0.0, 0.0], dtype=float))
    u = _safe_unit(u, np.array([0.0, 1.0, 0.0], dtype=float))

    yaw = np.degrees(np.arctan2(f[2], f[0]))
    pitch = np.degrees(np.arctan2(f[1], np.sqrt(f[0] * f[0] + f[2] * f[2])))

    world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    r_level = np.cross(world_up, f)
    if _safe_norm(r_level) < 1e-9:
        r_level = R @ np.array([0.0, 0.0, 1.0], dtype=float)
    r_level = _safe_unit(r_level, np.array([0.0, 0.0, 1.0], dtype=float))

    u_level = np.cross(f, r_level)
    u_level = _safe_unit(u_level, world_up)

    roll = np.degrees(np.arctan2(np.dot(np.cross(u_level, u), f), np.dot(u_level, u)))

    return _wrap_deg(float(roll)), _wrap_deg(float(pitch)), _wrap_deg(float(yaw))

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _bar_pm1(x: float, width: int = 20) -> str:
    x = float(_clamp(x, -1.0, 1.0))
    half = max(1, width // 2)
    left_fill = int(round(max(0.0, -x) * half))
    right_fill = int(round(max(0.0, x) * half))
    left = "=" * left_fill + "-" * (half - left_fill)
    right = "=" * right_fill + "-" * (half - right_fill)
    return f"{left}|{right}"

# ==========================
# Debug UI stored on scene
# ==========================
def _ensure_debug_ui():
    global scene
    if scene is None:
        return
    if getattr(scene, "debug_ui", None) is not None:
        return

    scene.debug_ui = {
        "panel": vp.wtext(
            text="<pre style='font-size:16px; font-family:monospace; background:#f0f0f0; border:1px solid #ccc; padding:8px;'>"
                 "Roll: 0\nPitch: 0\nYaw: 0\nSpeed: 0\nAoA: 0\nAltitude: 0\n"
                 "PitchIn: 0\nRollIn: 0\nYawIn: 0</pre>"
        )
    }

def update_debug_labels(entity):
    if entity is None:
        return

    _ensure_debug_ui()
    if scene is None or scene.debug_ui is None:
        return

    panel = scene.debug_ui.get("panel", None)
    if panel is None:
        return

    v = _extract_velocity(entity)
    if v is None:
        v = np.zeros(3, dtype=float)

    R = _extract_orientation(entity)

    speed = float(np.linalg.norm(v))
    aoa = float(physics.get_angle_of_attack(v, R))
    roll, pitch, yaw = _compute_rpy_deg_from_R(R)

    altitude = 0.0
    pos = _extract_position(entity)
    if pos is not None:
        altitude = float(pos[1])

    ci = getattr(entity, "control_inputs", None)
    if not isinstance(ci, dict):
        ci = getattr(_controlled_jet, "control_inputs", None) if _controlled_jet is not None else None
    if not isinstance(ci, dict):
        ci = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0}

    pitch_in = float(ci.get("pitch", 0.0))
    roll_in = float(ci.get("roll", 0.0))
    yaw_in = float(ci.get("yaw", 0.0))
    # Use the jet's self.throttle attribute if available, otherwise fallback to control_inputs
    throttle_in = float(getattr(entity, "throttle", ci.get("throttle", 0.0)))

    panel.text = (
        f"<pre style='font-size:16px; font-family:monospace; background:#f0f0f0; border:1px solid #ccc; padding:8px;'>"
        f"Roll:    {roll:.2f}°\n"
        f"Pitch:   {pitch:.2f}°\n"
        f"Yaw:     {yaw:.2f}°\n"
        f"Speed:   {speed:.2f} m/s\n"
        f"AoA:     {aoa:.2f}°\n"
        f"Altitude:{altitude:.2f} m\n"
        f"\n"
        f"PitchIn:   {pitch_in:+.2f} [{_bar_pm1(pitch_in)}]\n"
        f"RollIn:    {roll_in:+.2f} [{_bar_pm1(roll_in)}]\n"
        f"YawIn:     {yaw_in:+.2f} [{_bar_pm1(yaw_in)}]\n"
        f"Throttle:  {throttle_in:.2f}  [{_bar_01(throttle_in, width=20)}]"
        f"</pre>"
    )

# ==========================
# Visualization instance creation
# ==========================
def create_viz_instance(viz_shape: dict, entity):
    _ensure_scene()

    start_pos = viz_shape.get("starting_position", np.zeros(3))
    start_pos_np = _np3(start_pos) if start_pos is not None else np.zeros(3, dtype=float)
    if start_pos_np is None:
        start_pos_np = np.zeros(3, dtype=float)

    radius = float(viz_shape.get("size", 100))

    viz_instance = vp.sphere(
        pos=_vpvec(start_pos_np),
        radius=radius,
        color=viz_shape.get("color", vp.color.gray),
        opacity=float(viz_shape.get("opacity", 1.0)),
        make_trail=viz_shape.get("make_trail", False),
        trail_radius=float(viz_shape.get("trail_radius", 1.0)),
    )

    heading_len = float(viz_shape.get("heading_arrow_length", max(radius * 4.0, 2.0)))
    heading_shaft = float(viz_shape.get("heading_arrow_shaftwidth", max(radius * 0.5, 1.0)))

    vel_scale = float(viz_shape.get("velocity_arrow_scale", 1.0))
    vel_shaft = float(viz_shape.get("velocity_arrow_shaftwidth", max(radius * 0.20, 0.5)))
    vel_max_len = viz_shape.get("velocity_arrow_max_length", None)

    default_dir = np.array([1.0, 0.0, 0.0], dtype=float)

    heading_arrow = vp.arrow(
        pos=viz_instance.pos,
        axis=_vpvec(default_dir * heading_len),
        color=vp.color.red,
        shaftwidth=heading_shaft,
        round=True,
    )

    velocity_arrow = vp.arrow(
        pos=viz_instance.pos,
        axis=_vpvec(default_dir * 0.0),
        color=vp.color.yellow,
        shaftwidth=vel_shaft,
        round=True,
    )

    viz_instance.heading_arrow = heading_arrow
    viz_instance.velocity_arrow = velocity_arrow
    viz_instance._heading_arrow_length = heading_len
    viz_instance._velocity_arrow_scale = vel_scale
    viz_instance._velocity_arrow_max_length = vel_max_len

    if bool(getattr(entity, "manual_control", False)):
        global _controlled_jet
        _controlled_jet = entity

    return viz_instance

# ==========================
# Update instances
# ==========================
def update_instances(entities):
    global _controlled_jet

    debug_target = None
    for e in entities:
        if hasattr(e, "manual_control") and bool(getattr(e, "manual_control")):
            debug_target = e
            break
    if debug_target is None:
        for e in entities:
            if hasattr(e, "orientation") and (_extract_velocity(e) is not None or hasattr(e, "velocity")):
                debug_target = e
                break

    if debug_target is not None and bool(getattr(debug_target, "manual_control", False)):
        _controlled_jet = debug_target

    for entity in entities:
        if not (hasattr(entity, "viz_instance") and entity.viz_instance is not None):
            continue

        viz = entity.viz_instance

        pos_np = _extract_position(entity)
        if pos_np is None:
            continue

        pos_vp = _vpvec(pos_np)
        viz.pos = pos_vp

        if not hasattr(viz, "heading_arrow") or viz.heading_arrow is None:
            r = float(getattr(viz, "radius", 1.0))
            viz._heading_arrow_length = max(r * 4.0, 2.0)
            viz.heading_arrow = vp.arrow(
                pos=viz.pos,
                axis=_vpvec(np.array([1.0, 0.0, 0.0]) * viz._heading_arrow_length),
                color=vp.color.red,
                shaftwidth=max(r * 0.5, 1.0),
                round=True,
            )
        if not hasattr(viz, "velocity_arrow") or viz.velocity_arrow is None:
            r = float(getattr(viz, "radius", 1.0))
            viz._velocity_arrow_scale = getattr(viz, "_velocity_arrow_scale", 1.0)
            viz._velocity_arrow_max_length = getattr(viz, "_velocity_arrow_max_length", None)
            viz.velocity_arrow = vp.arrow(
                pos=viz.pos,
                axis=_vpvec(np.array([1.0, 0.0, 0.0]) * 0.0),
                color=vp.color.yellow,
                shaftwidth=max(r * 0.20, 0.5),
                round=True,
            )

        vel_np = _extract_velocity(entity)
        if vel_np is None:
            vel_np = np.zeros(3, dtype=float)

        speed = float(np.linalg.norm(vel_np))
        vel_dir = _safe_unit(vel_np, np.array([1.0, 0.0, 0.0], dtype=float))

        R = _extract_orientation(entity)
        heading_dir = _safe_unit(R @ np.array([1.0, 0.0, 0.0], dtype=float), vel_dir)

        heading_len = float(getattr(viz, "_heading_arrow_length", max(float(getattr(viz, "radius", 1.0)) * 2.0, 1.0)))
        viz.heading_arrow.pos = pos_vp
        viz.heading_arrow.axis = _vpvec(heading_dir * heading_len)

        vel_scale = float(getattr(viz, "_velocity_arrow_scale", 1.0))
        vel_len = speed * vel_scale

        vel_max_len = getattr(viz, "_velocity_arrow_max_length", None)
        if vel_max_len is not None:
            try:
                vel_len = min(vel_len, float(vel_max_len))
            except Exception:
                pass

        viz.velocity_arrow.pos = pos_vp
        viz.velocity_arrow.axis = _vpvec(vel_dir * vel_len)

    if debug_target is not None:
        p = _extract_position(debug_target)
        if p is not None:
            _focus_camera_on(_vpvec(p))

    update_debug_labels(debug_target)

# ==========================
# Clean visualization
# ==========================
def clean_viz():
    global scene, _static_scene_objects

    if scene is None:
        return

    for obj in list(_static_scene_objects):
        try:
            if hasattr(obj, "visible"):
                obj.visible = False
        except Exception:
            pass
    _static_scene_objects.clear()

    try:
        objects = list(scene.objects)
        for obj in objects:
            try:
                if hasattr(obj, "clear_trail"):
                    obj.clear_trail()
                if hasattr(obj, "make_trail"):
                    obj.make_trail = False
                if hasattr(obj, "visible"):
                    obj.visible = False
            except Exception:
                pass
    except Exception:
        pass

    try:
        if scene is not None:
            scene.delete()
    except Exception:
        pass

    scene = None

# ==========================
# Setup visualization
# ==========================
def setup_viz():
    global scene, _static_scene_objects

    _ensure_scene()

    from simulation import SIMULATION_BOX_SIZE

    ground = vp.box(
        pos=vp.vector(0, 0, 0),
        size=vp.vector(SIMULATION_BOX_SIZE[0], 1, SIMULATION_BOX_SIZE[2]),
        color=vp.color.green,
    )
    _static_scene_objects.append(ground)

    half_x, height, half_z = SIMULATION_BOX_SIZE[0] / 2, SIMULATION_BOX_SIZE[1], SIMULATION_BOX_SIZE[2] / 2

    bottom_edges = [
        vp.curve(pos=[vp.vector(-half_x, 0, -half_z), vp.vector(half_x, 0, -half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, 0, half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(half_x, 0, half_z), vp.vector(-half_x, 0, half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, 0, -half_z)], color=vp.color.black),
    ]

    top_edges = [
        vp.curve(pos=[vp.vector(-half_x, height, -half_z), vp.vector(half_x, height, -half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(half_x, height, -half_z), vp.vector(half_x, height, half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(half_x, height, half_z), vp.vector(-half_x, height, half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(-half_x, height, half_z), vp.vector(-half_x, height, -half_z)], color=vp.color.black),
    ]

    vertical_edges = [
        vp.curve(pos=[vp.vector(-half_x, 0, -half_z), vp.vector(-half_x, height, -half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, height, -half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(half_x, 0, half_z), vp.vector(half_x, height, half_z)], color=vp.color.black),
        vp.curve(pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, height, half_z)], color=vp.color.black),
    ]

    _static_scene_objects.extend(bottom_edges + top_edges + vertical_edges)

    _ensure_debug_ui()

# ==========================
# Internal helper functions
# ==========================
def _ensure_scene():
    global scene

    if scene is None:
        from simulation import SIMULATION_BOX_SIZE, SIMULATION_RESOLUTION

        scene = vp.canvas(
            width=int(SIMULATION_RESOLUTION[0]),
            height=int(SIMULATION_RESOLUTION[1]),
            background=vp.color.white,
            title="Fighter Jet Physics Simulation",
        )

        scene.center = vp.vector(0, SIMULATION_BOX_SIZE[1] / 2, 0)
        scene.debug_ui = None

        scene.bind("keydown", _on_keydown)
        scene.bind("keyup", _on_keyup)

def _apply_inputs():
    global _controlled_jet
    if _controlled_jet is None or not hasattr(_controlled_jet, "control_inputs"):
        return

    w = 1.0 if "w" in _keys_down else 0.0
    s = 1.0 if "s" in _keys_down else 0.0
    pitch = w - s

    d = 1.0 if "d" in _keys_down else 0.0
    a = 1.0 if "a" in _keys_down else 0.0
    roll = d - a

    e = 1.0 if "e" in _keys_down else 0.0
    q = 1.0 if "q" in _keys_down else 0.0
    yaw = e - q

    shift = 1.0 if "shift" in _keys_down else 0.0
    throttle = shift

    _controlled_jet.control_inputs["pitch"] = pitch
    _controlled_jet.control_inputs["roll"] = roll
    _controlled_jet.control_inputs["yaw"] = yaw
    _controlled_jet.throttle = throttle

def _on_keydown(evt):
    key = evt.key.lower()
    if key == "esc":
        # Add simulation end trigger here
        return
    # Detect shift key
    if evt.shift:
        _keys_down.add("shift")
    if key in ("w", "a", "s", "d", "q", "e"):
        _keys_down.add(key)
        _apply_inputs()
    elif key == "shift":
        _keys_down.add("shift")
        _apply_inputs()

def _on_keyup(evt):
    key = evt.key.lower()
    if evt.shift:
        _keys_down.discard("shift")
    if key in ("w", "a", "s", "d", "q", "e"):
        _keys_down.discard(key)
        _apply_inputs()
    elif key == "shift":
        _keys_down.discard("shift")
        _apply_inputs()

def _focus_camera_on(pos_vp: vp.vector):
    global scene
    if scene is None:
        return
    alpha = 0.20
    scene.center = scene.center * (1 - alpha) + pos_vp * alpha
