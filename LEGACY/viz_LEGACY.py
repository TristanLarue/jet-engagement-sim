import vpython as vp
from typing import Tuple
import numpy as np
import physics

# ==========================
# Arrow visual configuration
# ==========================
MIN_ARROW_WIDTH = 150.0   # meters: shaft (thickness) for ALL arrows
VEL_ARROW_LEN   = 1000.0  # meters: length for velocity arrow
FORCE_ARROW_LEN = 750.0   # meters: length for force arrows

# ==========================
# Global storage
# ==========================
scene: vp.canvas | None = None

# Static scene objects (ground, box edges, etc.)
_static_scene_objects: list = []

# Per-entity visual components
viz_instances: dict[int, vp.sphere] = {}
ui_panels: dict[int, vp.wtext] = {}

velocity_arrows: dict[int, vp.arrow] = {}
thrust_arrows: dict[int, vp.arrow] = {}
lift_arrows: dict[int, vp.arrow] = {}
drag_arrows: dict[int, vp.arrow] = {}
gravity_arrows: dict[int, vp.arrow] = {}

joystick_canvases: dict[int, dict] = {}

# ==========================
# Manual control (keyboard)
# ==========================

MANUAL_INPUT_MAG = 1.0  # full deflection when key held
GLOBAL_MANUAL_PITCH: float = 0.0
GLOBAL_MANUAL_ROLL: float = 0.0
_manual_keys_down: set[str] = set()


def _recompute_manual_inputs():
    global GLOBAL_MANUAL_PITCH, GLOBAL_MANUAL_ROLL

    pitch = 0.0
    if "up" in _manual_keys_down:
        pitch += MANUAL_INPUT_MAG
    if "down" in _manual_keys_down:
        pitch -= MANUAL_INPUT_MAG

    roll = 0.0
    if "right" in _manual_keys_down:
        roll += MANUAL_INPUT_MAG
    if "left" in _manual_keys_down:
        roll -= MANUAL_INPUT_MAG

    GLOBAL_MANUAL_PITCH = float(np.clip(pitch, -1.0, 1.0))
    GLOBAL_MANUAL_ROLL = float(np.clip(roll, -1.0, 1.0))


def _on_keydown(evt):
    key = evt.key
    if key in ("up", "down", "left", "right"):
        _manual_keys_down.add(key)
        _recompute_manual_inputs()


def _on_keyup(evt):
    key = evt.key
    if key in ("up", "down", "left", "right"):
        if key in _manual_keys_down:
            _manual_keys_down.remove(key)
            _recompute_manual_inputs()


# ==========================
# Window / scene setup
# ==========================

def _ensure_window() -> vp.canvas:
    global scene
    if scene is None:
        from config import BOX_SIZE

        scene = vp.canvas(width=1000, height=600, background=vp.color.white)
        scene.center = vp.vector(0, BOX_SIZE[1] / 2, 0)

        scene.bind("keydown", _on_keydown)
        scene.bind("keyup", _on_keyup)

    return scene


def _build_static_scene():
    from config import BOX_SIZE

    _ensure_window()

    for obj in list(_static_scene_objects):
        try:
            if hasattr(obj, "visible"):
                obj.visible = False
        except Exception:
            pass
    _static_scene_objects.clear()

    half_x, height, half_z = BOX_SIZE[0] / 2, BOX_SIZE[1], BOX_SIZE[2] / 2

    ground = vp.box(
        pos=vp.vector(0, 0, 0),
        size=vp.vector(BOX_SIZE[0], 1, BOX_SIZE[2]),
        color=vp.color.green,
    )
    _static_scene_objects.append(ground)

    # Bottom 4 edges
    c1 = vp.curve(
        pos=[vp.vector(-half_x, 0, -half_z), vp.vector(half_x, 0, -half_z)],
        color=vp.color.black,
    )
    c2 = vp.curve(
        pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, 0, half_z)],
        color=vp.color.black,
    )
    c3 = vp.curve(
        pos=[vp.vector(half_x, 0, half_z), vp.vector(-half_x, 0, half_z)],
        color=vp.color.black,
    )
    c4 = vp.curve(
        pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, 0, -half_z)],
        color=vp.color.black,
    )

    # Top 4 edges
    c5 = vp.curve(
        pos=[vp.vector(-half_x, height, -half_z), vp.vector(half_x, height, -half_z)],
        color=vp.color.black,
    )
    c6 = vp.curve(
        pos=[vp.vector(half_x, height, -half_z), vp.vector(half_x, height, half_z)],
        color=vp.color.black,
    )
    c7 = vp.curve(
        pos=[vp.vector(half_x, height, half_z), vp.vector(-half_x, height, half_z)],
        color=vp.color.black,
    )
    c8 = vp.curve(
        pos=[vp.vector(-half_x, height, half_z), vp.vector(-half_x, height, -half_z)],
        color=vp.color.black,
    )

    # Vertical 4 edges
    c9 = vp.curve(
        pos=[vp.vector(-half_x, 0, -half_z), vp.vector(-half_x, height, -half_z)],
        color=vp.color.black,
    )
    c10 = vp.curve(
        pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, height, -half_z)],
        color=vp.color.black,
    )
    c11 = vp.curve(
        pos=[vp.vector(half_x, 0, half_z), vp.vector(half_x, height, half_z)],
        color=vp.color.black,
    )
    c12 = vp.curve(
        pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, height, half_z)],
        color=vp.color.black,
    )

    _static_scene_objects.extend([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])


def initialize_viz() -> vp.canvas:
    s = _ensure_window()
    _build_static_scene()
    return s


# ==========================
# Entity instance creation
# ==========================

def create_instance(
    shape: str = "missile",
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: float = 100,
    opacity: float = 1.0,
    make_trail: bool = False,
    trail_radius: float = 0.05,
):
    _ensure_window()
    pos = vp.vector(*position)

    if shape == "jet":
        color = vp.color.red
    elif shape == "missile":
        color = vp.color.blue
    else:
        color = vp.color.white

    obj = vp.sphere(
        pos=pos,
        radius=size * 0.001,
        color=color,
        opacity=opacity,
        make_trail=make_trail,
        trail_radius=trail_radius,
    )

    obj_id = id(obj)
    viz_instances[obj_id] = obj

    shaft = MIN_ARROW_WIDTH

    vel_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(1, 0, 0),
        color=vp.color.yellow,
        shaftwidth=shaft,
    )
    velocity_arrows[obj_id] = vel_arrow

    thrust_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(1, 0, 0),
        color=vp.color.red,
        shaftwidth=shaft,
        opacity=0.7,
    )
    thrust_arrows[obj_id] = thrust_arrow

    lift_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(0, 1, 0),
        color=vp.color.green,
        shaftwidth=shaft,
        opacity=0.7,
    )
    lift_arrows[obj_id] = lift_arrow

    drag_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(-1, 0, 0),
        color=vp.color.white,
        shaftwidth=shaft,
        opacity=0.7,
    )
    drag_arrows[obj_id] = drag_arrow

    gravity_color = vp.vector(0.6, 0.0, 0.9)
    gravity_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(0, -1, 0),
        color=gravity_color,
        shaftwidth=shaft,
        opacity=0.7,
    )
    gravity_arrows[obj_id] = gravity_arrow

    ui_label = vp.wtext(text="")
    ui_panels[obj_id] = ui_label

    if shape == "jet":
        joystick_canvas = vp.graph(
            width=200,
            height=200,
            xmin=-1.2,
            xmax=1.2,
            ymin=-1.2,
            ymax=1.2,
            xticks=False,
            yticks=False,
            background=vp.color.gray(0.95),
        )

        boundary = vp.gcurve(color=vp.color.black, graph=joystick_canvas)
        boundary.plot(pos=[(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)])

        cross_h = vp.gcurve(color=vp.color.gray(0.7), graph=joystick_canvas)
        cross_h.plot(pos=[(-1, 0), (1, 0)])
        cross_v = vp.gcurve(color=vp.color.gray(0.7), graph=joystick_canvas)
        cross_v.plot(pos=[(0, -1), (0, 1)])

        input_dot = vp.gdots(color=vp.color.green, size=10, graph=joystick_canvas)
        input_dot.plot(pos=(0, 0))

        orient_dot = vp.gdots(color=vp.color.blue, size=8, graph=joystick_canvas)
        orient_dot.plot(pos=(0, 0))

        joystick_canvases[obj_id] = {
            "canvas": joystick_canvas,
            "input": input_dot,
            "orient": orient_dot,
            "boundary": boundary,
            "cross_h": cross_h,
            "cross_v": cross_v,
        }

    return obj


# ==========================
# Update helpers
# ==========================

def _set_arrow_from_vector(arrow: vp.arrow, pos: vp.vector, vec: np.ndarray, length: float):
    arrow.pos = pos
    mag = np.linalg.norm(vec)
    if mag > 0.0:
        direction = vec / mag
        arrow.axis = vp.vector(*direction) * length
    else:
        arrow.axis = vp.vector(0, 0, 0)


def update_instance(entity):
    pos_vec = vp.vector(*tuple(entity.p))
    entity.viz_instance.pos = pos_vec

    R = entity.orientation
    arrow_id = id(entity.viz_instance)

    v = entity.v
    speed = np.linalg.norm(v)

    thrust_force = physics.get_thrust_force(entity.throttle, entity.thrust_force, R)
    lift_force = physics.get_lift_force(
        v, entity.reference_area, entity.max_lift_coefficient, R
    )
    drag_force = physics.get_drag_force(
        v,
        entity.reference_area,
        entity.min_drag_coefficient,
        entity.max_drag_coefficient,
        R,
    )
    gravity_vec = physics.get_gravity_acc()

    if arrow_id in velocity_arrows:
        vel_arrow = velocity_arrows[arrow_id]
        vel_arrow.pos = pos_vec
        if speed > 0.0:
            vel_dir = v / speed
            vel_arrow.axis = vp.vector(*vel_dir) * VEL_ARROW_LEN
        else:
            vel_arrow.axis = vp.vector(0, 0, 0)

    if arrow_id in thrust_arrows:
        _set_arrow_from_vector(thrust_arrows[arrow_id], pos_vec, thrust_force, FORCE_ARROW_LEN)

    if arrow_id in lift_arrows:
        _set_arrow_from_vector(lift_arrows[arrow_id], pos_vec, lift_force, FORCE_ARROW_LEN)

    if arrow_id in drag_arrows:
        _set_arrow_from_vector(drag_arrows[arrow_id], pos_vec, drag_force, FORCE_ARROW_LEN)

    if arrow_id in gravity_arrows:
        _set_arrow_from_vector(gravity_arrows[arrow_id], pos_vec, gravity_vec, FORCE_ARROW_LEN)

    panel_id = arrow_id
    if panel_id in ui_panels:
        ui_label = ui_panels[panel_id]
        aoa = physics.get_aoa(entity.v, R)
        speed = np.linalg.norm(entity.v)

        if entity.shape == "jet":
            # Extract Euler angles for display
            roll, pitch, yaw = physics.extract_euler_angles(entity.orientation)
            reward = getattr(entity, "ai_reward", None)
            reward_str = f"Reward: {reward:.3f}" if reward is not None else "Reward: N/A"
            throttle = getattr(entity, "throttle", None)
            throttle_str = f"Throttle: {throttle:.2f}" if throttle is not None else "Throttle: N/A"
            ui_label.text = (
                f"<span style='font-size:16px'>"
                f"<b style='color:red'>JET</b><br>"
                f"AoA: {aoa:.1f}°<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Pitch: {pitch:.1f}°<br>"
                f"Roll: {roll:.1f}°<br>"
                f"Yaw: {yaw:.1f}°<br>"
                f"{reward_str}<br>"
                f"{throttle_str}<br>"
                f"</span>"
            )
        elif entity.shape == "missile":
            if getattr(entity, "target", None) is not None:
                dist = np.linalg.norm(entity.target.p - entity.p)
                eta = dist / speed if speed > 0.0 else float("inf")
            else:
                dist = float("nan")
                eta = float("nan")
            ui_label.text = (
                f"<span style='font-size:16px'>"
                f"<b style='color:blue'>MISSILE</b><br>"
                f"AoA: {aoa:.1f}°<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Distance: {dist:.1f} m<br>"
                f"ETA: {eta:.2f} s<br>"
                f"</span>"
            )

    if entity.shape == "jet" and getattr(entity, "manual_control", False):
        entity.pitch_input = GLOBAL_MANUAL_PITCH
        entity.roll_input = GLOBAL_MANUAL_ROLL

    if entity.shape == "jet":
        j_id = id(entity.viz_instance)
        if j_id in joystick_canvases:
            js = joystick_canvases[j_id]

            roll_in = float(np.clip(getattr(entity, "roll_input", 0.0), -1.0, 1.0))
            pitch_in = float(np.clip(getattr(entity, "pitch_input", 0.0), -1.0, 1.0))

            js["input"].data = []
            js["input"].plot(pos=(roll_in, pitch_in))

            # Extract current orientation for display
            roll, pitch, yaw = physics.extract_euler_angles(entity.orientation)
            max_ang = 180.0
            roll_norm = float(np.clip((roll - 180) / max_ang, -1.0, 1.0))
            pitch_norm = float(np.clip((pitch - 180) / max_ang, -1.0, 1.0))

            js["orient"].data = []
            js["orient"].plot(pos=(roll_norm, pitch_norm))

    return


# ==========================
# Cleanup / reset
# ==========================

def cleanup_viz(rebuild: bool = True):
    """
    Delete ALL visual components (3D objects + UI panels + joystick graphs),
    keep the window open, and optionally rebuild the base scene.
    """
    global GLOBAL_MANUAL_PITCH, GLOBAL_MANUAL_ROLL

    _ensure_window()

    # 1) Hide / clear tracked per-entity objects
    for obj in list(viz_instances.values()):
        try:
            if hasattr(obj, "clear_trail"):
                obj.clear_trail()
            if hasattr(obj, "make_trail"):
                obj.make_trail = False
            if hasattr(obj, "visible"):
                obj.visible = False
        except Exception:
            pass
    viz_instances.clear()

    for d in (velocity_arrows, thrust_arrows, lift_arrows, drag_arrows, gravity_arrows):
        for arr in list(d.values()):
            try:
                if hasattr(arr, "visible"):
                    arr.visible = False
            except Exception:
                pass
        d.clear()

    # 2) Clear UI labels
    for lbl in list(ui_panels.values()):
        try:
            lbl.text = ""
            if hasattr(lbl, "visible"):
                lbl.visible = False
        except Exception:
            pass
    ui_panels.clear()

    # 3) Clear joystick canvases (graphs + curves + dots)
    for js in list(joystick_canvases.values()):
        for key in ("input", "orient", "boundary", "cross_h", "cross_v", "canvas"):
            obj = js.get(key)
            if obj is None:
                continue
            try:
                if hasattr(obj, "delete"):
                    obj.delete()
                elif hasattr(obj, "visible"):
                    obj.visible = False
            except Exception:
                pass
        js.clear()
    joystick_canvases.clear()

    # 4) Hide static scene objects
    for obj in list(_static_scene_objects):
        try:
            if hasattr(obj, "visible"):
                obj.visible = False
        except Exception:
            pass
    _static_scene_objects.clear()

    # 5) Safety net: hide any remaining 3D objects in the scene
    try:
        objs = list(vp.scene.objects)
    except Exception:
        objs = []

    for obj in objs:
        try:
            if hasattr(obj, "clear_trail"):
                obj.clear_trail()
            if hasattr(obj, "make_trail"):
                obj.make_trail = False
            if hasattr(obj, "visible"):
                obj.visible = False
        except Exception:
            continue

    # 6) Reset manual input state
    GLOBAL_MANUAL_PITCH = 0.0
    GLOBAL_MANUAL_ROLL = 0.0
    _manual_keys_down.clear()

    # 7) Rebuild base scene if requested
    if rebuild:
        _build_static_scene()


def viz_cleanup():
    cleanup_viz(rebuild=True)


def close_viz():
    global scene
    cleanup_viz(rebuild=False)
    try:
        if scene is not None:
            scene.delete()
    except Exception:
        pass
    scene = None
