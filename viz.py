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

# Global storage
ui_panels = {}

velocity_arrows = {}
thrust_arrows = {}
lift_arrows = {}
drag_arrows = {}
gravity_arrows = {}

joystick_canvases = {}

# ==========================
# Manual control (keyboard)
# ==========================
MANUAL_INPUT_MAG = 1.0  # full deflection when key held

manual_roll_input: float = 0.0
manual_pitch_input: float = 0.0
_manual_keys_down: set[str] = set()


def _recompute_manual_inputs():
    """Recompute manual pitch/roll from currently pressed arrow keys."""
    global manual_pitch_input, manual_roll_input

    pitch = 0.0
    # Up arrow: nose up (positive pitch_input)
    if "up" in _manual_keys_down:
        pitch += MANUAL_INPUT_MAG
    if "down" in _manual_keys_down:
        pitch -= MANUAL_INPUT_MAG

    roll = 0.0
    if "right" in _manual_keys_down:
        roll += MANUAL_INPUT_MAG
    if "left" in _manual_keys_down:
        roll -= MANUAL_INPUT_MAG

    manual_pitch_input = float(np.clip(pitch, -1.0, 1.0))
    manual_roll_input = float(np.clip(roll, -1.0, 1.0))


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


def create_instance(
    shape: str = "missile",
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: float = 100,
    opacity: float = 1.0,
    make_trail: bool = False,
    trail_radius: float = 0.05,
):
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

    shaft = MIN_ARROW_WIDTH

    # === Velocity direction (YELLOW) ===
    vel_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(1, 0, 0),
        color=vp.color.yellow,
        shaftwidth=shaft,
    )
    velocity_arrows[id(obj)] = vel_arrow

    # === Thrust force (RED) ===
    thrust_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(1, 0, 0),
        color=vp.color.red,
        shaftwidth=shaft,
        opacity=0.7,
    )
    thrust_arrows[id(obj)] = thrust_arrow

    # === Lift force (GREEN) ===
    lift_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(0, 1, 0),
        color=vp.color.green,
        shaftwidth=shaft,
        opacity=0.7,
    )
    lift_arrows[id(obj)] = lift_arrow

    # === Drag force (WHITE) ===
    drag_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(-1, 0, 0),
        color=vp.color.white,
        shaftwidth=shaft,
        opacity=0.7,
    )
    drag_arrows[id(obj)] = drag_arrow

    # === Gravity direction (PURPLE) ===
    gravity_color = vp.vector(0.6, 0.0, 0.9)
    gravity_arrow = vp.arrow(
        pos=pos,
        axis=vp.vector(0, -1, 0),
        color=gravity_color,
        shaftwidth=shaft,
        opacity=0.7,
    )
    gravity_arrows[id(obj)] = gravity_arrow

    # Text UI panel
    ui_label = vp.wtext(text="")
    ui_panels[id(obj)] = ui_label

    # Joystick panel ONLY for jet
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

        # Green = input (roll_input, pitch_input)
        input_dot = vp.gdots(color=vp.color.green, size=10, graph=joystick_canvas)
        input_dot.plot(pos=(0, 0))

        # Blue = current orientation (roll, pitch)
        orient_dot = vp.gdots(color=vp.color.blue, size=8, graph=joystick_canvas)
        orient_dot.plot(pos=(0, 0))

        joystick_canvases[id(obj)] = {
            "canvas": joystick_canvas,
            "input": input_dot,
            "orient": orient_dot,
        }

    return obj


def initialize_viz():
    from config import BOX_SIZE

    scene = vp.canvas(width=1080, height=1080, background=vp.color.white)
    ground = vp.box(
        pos=vp.vector(0, 0, 0),
        size=vp.vector(BOX_SIZE[0], 1, BOX_SIZE[2]),
        color=vp.color.green,
    )

    half_x, height, half_z = BOX_SIZE[0] / 2, BOX_SIZE[1], BOX_SIZE[2] / 2

    # Bottom 4 edges
    vp.curve(
        pos=[vp.vector(-half_x, 0, -half_z), vp.vector(half_x, 0, -half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, 0, half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(half_x, 0, half_z), vp.vector(-half_x, 0, half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, 0, -half_z)],
        color=vp.color.black,
    )

    # Top 4 edges
    vp.curve(
        pos=[vp.vector(-half_x, height, -half_z), vp.vector(half_x, height, -half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(half_x, height, -half_z), vp.vector(half_x, height, half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(half_x, height, half_z), vp.vector(-half_x, height, half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(-half_x, height, half_z), vp.vector(-half_x, height, -half_z)],
        color=vp.color.black,
    )

    # Vertical 4 edges
    vp.curve(
        pos=[vp.vector(-half_x, 0, -half_z), vp.vector(-half_x, height, -half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, height, -half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(half_x, 0, half_z), vp.vector(half_x, height, half_z)],
        color=vp.color.black,
    )
    vp.curve(
        pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, height, half_z)],
        color=vp.color.black,
    )

    scene.center = vp.vector(0, height / 2, 0)

    # Bind keyboard handlers for manual control
    scene.bind("keydown", _on_keydown)
    scene.bind("keyup", _on_keyup)

    return scene


def _set_arrow_from_vector(arrow: vp.arrow, pos: vp.vector, vec: np.ndarray, length: float):
    """Set arrow at position 'pos' pointing along 'vec' with fixed length."""
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

    R = physics.get_rotation_matrix(entity.roll, entity.pitch, entity.yaw)
    arrow_id = id(entity.viz_instance)

    v = entity.v
    speed = np.linalg.norm(v)

    # Forces in world space
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
    gravity_vec = physics.get_gravity_acc()  # direction is enough

    # === Velocity arrow (YELLOW) ===
    if arrow_id in velocity_arrows:
        vel_arrow = velocity_arrows[arrow_id]
        vel_arrow.pos = pos_vec
        if speed > 0.0:
            vel_dir = v / speed
            vel_arrow.axis = vp.vector(*vel_dir) * VEL_ARROW_LEN
        else:
            vel_arrow.axis = vp.vector(0, 0, 0)

    # === Thrust arrow (RED) ===
    if arrow_id in thrust_arrows:
        _set_arrow_from_vector(thrust_arrows[arrow_id], pos_vec, thrust_force, FORCE_ARROW_LEN)

    # === Lift arrow (GREEN) ===
    if arrow_id in lift_arrows:
        _set_arrow_from_vector(lift_arrows[arrow_id], pos_vec, lift_force, FORCE_ARROW_LEN)

    # === Drag arrow (WHITE) ===
    if arrow_id in drag_arrows:
        _set_arrow_from_vector(drag_arrows[arrow_id], pos_vec, drag_force, FORCE_ARROW_LEN)

    # === Gravity arrow (PURPLE) ===
    if arrow_id in gravity_arrows:
        _set_arrow_from_vector(gravity_arrows[arrow_id], pos_vec, gravity_vec, FORCE_ARROW_LEN)

    # === Text UI panel ===
    panel_id = arrow_id
    if panel_id in ui_panels:
        ui_label = ui_panels[panel_id]
        aoa = physics.get_aoa(entity.v, R)
        speed = np.linalg.norm(entity.v)

        if entity.shape == "jet":
            reward = getattr(entity, "ai_reward", None)
            reward_str = f"Reward: {reward:.3f}" if reward is not None else "Reward: N/A"
            throttle = getattr(entity, "throttle", None)
            throttle_str = f"Throttle: {throttle:.2f}" if throttle is not None else "Throttle: N/A"
            ui_label.text = (
                f"<span style='font-size:16px'>"
                f"<b style='color:red'>JET</b><br>"
                f"AoA: {aoa:.1f}°<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Pitch: {entity.pitch:.1f}°<br>"
                f"Roll: {entity.roll:.1f}°<br>"
                f"Yaw: {entity.yaw:.1f}°<br>"
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

    # === Manual control override for jet ===
    if entity.shape == "jet" and getattr(entity, "manual_control", False):
        # Override pitch/roll inputs with keyboard-based manual inputs
        entity.pitch_input = manual_pitch_input
        entity.roll_input = manual_roll_input

    # === Joystick update ONLY for jet ===
    if entity.shape == "jet":
        j_id = id(entity.viz_instance)
        if j_id in joystick_canvases:
            js = joystick_canvases[j_id]

            roll_in = float(np.clip(getattr(entity, "roll_input", 0.0), -1.0, 1.0))
            pitch_in = float(np.clip(getattr(entity, "pitch_input", 0.0), -1.0, 1.0))

            js["input"].data = []
            js["input"].plot(pos=(roll_in, pitch_in))

            max_ang = 180.0
            roll_norm = float(np.clip((entity.roll - 180) / max_ang, -1.0, 1.0))
            pitch_norm = float(np.clip((entity.pitch - 180) / max_ang, -1.0, 1.0))

            js["orient"].data = []
            js["orient"].plot(pos=(roll_norm, pitch_norm))

    return


def cleanup_viz():
    global ui_panels, velocity_arrows, thrust_arrows, lift_arrows, drag_arrows, gravity_arrows, joystick_canvases
    global manual_roll_input, manual_pitch_input, _manual_keys_down

    # Safely hide and clear all VPython objects in the active scene
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

    # Hide and clear all arrow dictionaries
    for d in (velocity_arrows, thrust_arrows, lift_arrows, drag_arrows, gravity_arrows):
        for arr in list(d.values()):
            try:
                if hasattr(arr, "visible"):
                    arr.visible = False
            except Exception:
                pass
        d.clear()

    # Clear UI labels
    for lbl in list(ui_panels.values()):
        try:
            lbl.text = ""
            if hasattr(lbl, "visible"):
                lbl.visible = False
        except Exception:
            pass
    ui_panels.clear()

    # Clear joystick canvases
    for js in list(joystick_canvases.values()):
        try:
            js["input"].data = []
            js["orient"].data = []
            if hasattr(js["canvas"], "visible"):
                js["canvas"].visible = False
        except Exception:
            pass
    joystick_canvases.clear()

    # Reset manual input state
    manual_roll_input = 0.0
    manual_pitch_input = 0.0
    _manual_keys_down.clear()


def close_viz():
    cleanup_viz()
    try:
        vp.scene.delete()
    except Exception:
        pass
