from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import importlib.util
import numpy as np
import vpython as vp
MODELS_SCALE = 50

@dataclass
class Viz_instance:
    kind: str
    viz_shape: Dict[str, Any]


@dataclass
class Entity_visual:
    body: Any
    kind: str
    length: float
    heading: Any | None = None
    # Jet-only: same model loaded into the forces canvas
    body_forces: Any | None = None
    # Trails created via vp.attach_trail (curves) so mesh compounds work
    trails: list[Any] = field(default_factory=list)
_main_canvas: Optional[vp.canvas] = None
_forces_canvas: Optional[vp.canvas] = None
_canvases_created: bool = False

_entity_visuals: Dict[int, Entity_visual] = {}
_static_objects: list[Any] = []
_force_objects: list[Any] = []
_last_jet_instance: Any = None  # Global to track last jet instance created
_missile_lines: Dict[int, Any] = {}  # Store lines connecting jet to missiles
_exploded_missiles: set[int] = set()  # Track which missiles have exploded
_explosion_spheres: Dict[int, Any] = {}  # Store permanent orange spheres for explosions

FOCUS_ENTITY_ID: int = 0  # Global variable for camera focus entity ID

_main_status: Any | None = None
_forces_status: Any | None = None
_main_info: Any | None = None
_jet_static: Any | None = None
_jet_dynamic: Any | None = None
_reward_display: Any | None = None
_forces_jet_display: Any | None = None  # Jet object in forces canvas
_jet_intercepted_label: Any | None = None  # Label for "JET INTERCEPTED" message

# Force arrows for visualization
_force_arrows: Dict[str, Any] = {}  # Store force arrows by name

_sim_config: Dict[str, Any] = {}
_tick: int = 0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_ROOT = _REPO_ROOT / "res" / "viz_models"
_LOADER_MODULES: Dict[str, Any] = {}


def _vpvec(x: Any) -> vp.vector:
    if isinstance(x, vp.vector):
        return x
    a = np.array(x, dtype=float).reshape(3)
    return vp.vector(float(a[0]), float(a[1]), float(a[2]))


def _coerce_color(c: Any) -> vp.vector:
    """Accept vp.vector or (r,g,b) sequences. Falls back to white."""
    if c is None:
        return vp.color.white
    if isinstance(c, vp.vector):
        return c
    try:
        a = np.asarray(c, dtype=float).reshape(-1)
        if a.size >= 3:
            return vp.vector(float(a[0]), float(a[1]), float(a[2]))
    except Exception:
        pass
    return vp.color.white


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return fallback
    return v / n


def _get_viz_id(entity: Any) -> int:
    vid = getattr(entity, "viz_id", None)
    if isinstance(vid, int):
        return vid
    try:
        new_id = int(id(entity))
        setattr(entity, "viz_id", new_id)
        return new_id
    except Exception:
        return int(id(entity))


def _get_kind(entity: Any) -> str:
    k = getattr(entity, "shape", None)
    if isinstance(k, str) and k:
        return k
    name = entity.__class__.__name__.lower()
    if "missile" in name:
        return "missile"
    if "jet" in name:
        return "jet"
    return "entity"


def _get_viz_shape(entity: Any) -> Dict[str, Any]:
    vs = getattr(entity, "viz_shape", None)
    if isinstance(vs, dict):
        return vs

    vi = getattr(entity, "viz_instance", None)
    if isinstance(vi, Viz_instance):
        return vi.viz_shape
    vs2 = getattr(vi, "viz_shape", None)
    if isinstance(vs2, dict):
        return vs2

    return {}


def _extract_pos(entity: Any) -> Optional[np.ndarray]:
    p = getattr(entity, "position", None)
    if p is None:
        return None
    a = np.array(p, dtype=float).reshape(3)
    return a


def _extract_R(entity: Any) -> Optional[np.ndarray]:
    R = getattr(entity, "orientation", None)
    if R is None:
        return None
    return np.array(R, dtype=float).reshape(3, 3)


def _extract_vel(entity: Any) -> Optional[np.ndarray]:
    v = getattr(entity, "velocity", None)
    if v is None:
        return None
    a = np.array(v, dtype=float).reshape(3)
    return a


def _fmt_sim_info() -> str:
    duration_s = _sim_config.get("duration_s", "N/A")
    tick_rate = _sim_config.get("tick_rate", "N/A")
    sim_speed = _sim_config.get("sim_speed", "N/A")
    box_size = _sim_config.get("box_size", "N/A")
    resolution = _sim_config.get("resolution", "N/A")

    time_remaining = "N/A"
    if isinstance(duration_s, (int, float)) and isinstance(tick_rate, (int, float)) and tick_rate > 0:
        t = float(_tick) / float(tick_rate)
        time_remaining = f"{max(0.0, float(duration_s) - t):.2f}"

    return (
        "SIMULATION\n"
        f"SIMULATION_DURATION: {duration_s}\n"
        f"SIMULATION_TICKRATE: {tick_rate}\n"
        f"SIMULATION_SPEED: {sim_speed}\n"
        f"SIMULATION_BOX_SIZE: {box_size}\n"
        f"SIMULATION_RESOLUTION: {resolution}\n"
        "\n"
        f"Current Tick: {_tick}\n"
        f"Time remaining: {time_remaining}"
    )


def _static_text(entity: Any = None) -> str:
    if entity is None:
        return (
            "Configuration\n"
            "Mass: N/A\n"
            "Reference area: N/A\n"
            "Cd min/max: N/A\n"
            "Cl max: N/A\n"
            "I_roll: N/A\n"
            "I_pitch: N/A\n"
            "I_yaw: N/A\n"
            "Optimal lift AoA / incidence: N/A\n"
            "Wingspan: N/A\n"
            "Length: N/A\n"
            "Thrust force: N/A"
        )
    
    # Extract real entity data
    mass = getattr(entity, 'mass', 'N/A')
    ref_area = getattr(entity, 'reference_area', 'N/A')
    cd_min = getattr(entity, 'min_drag_coefficient', 'N/A')
    cd_max = getattr(entity, 'max_drag_coefficient', 'N/A')
    cl_max = getattr(entity, 'max_lift_coefficient', 'N/A')
    moment_i = getattr(entity, 'moment_of_inertia', None)
    i_roll = moment_i[0] if moment_i is not None and len(moment_i) > 0 else 'N/A'
    i_pitch = moment_i[1] if moment_i is not None and len(moment_i) > 1 else 'N/A'
    i_yaw = moment_i[2] if moment_i is not None and len(moment_i) > 2 else 'N/A'
    opt_aoa = getattr(entity, 'optimal_lift_aoa', 'N/A')
    wingspan = getattr(entity, 'wingspan', 'N/A')
    length = getattr(entity, 'length', 'N/A')
    thrust = getattr(entity, 'thrust_force', 'N/A')
    
    return (
        "Configuration\n"
        f"Mass: {mass:.1f} kg\n"
        f"Reference area: {ref_area:.2f} m²\n"
        f"Cd min/max: {cd_min:.3f}/{cd_max:.3f}\n"
        f"Cl max: {cl_max:.2f}\n"
        f"I_roll: {i_roll:.0f} kg·m²\n"
        f"I_pitch: {i_pitch:.0f} kg·m²\n"
        f"I_yaw: {i_yaw:.0f} kg·m²\n"
        f"Optimal lift AoA: {opt_aoa:.1f}°\n"
        f"Wingspan: {wingspan:.2f} m\n"
        f"Length: {length:.1f} m\n"
        f"Thrust force: {thrust:.0f} N"
    )


def _dynamic_text(entity: Any = None) -> str:
    if entity is None:
        return (
            "Live Data\n"
            "Bearing: N/A\n"
            "Altitude: N/A\n"
            "Speed |V|: N/A\n"
            "Omega |ω|: N/A\n"
            "AoA: N/A\n"
            "Sideslip: N/A\n"
            "Cd: N/A\n"
            "Cl: N/A\n"
            "Closest missile dist: N/A\n"
            "Status: N/A"
        )
    
    # Extract real dynamic entity data
    pos = getattr(entity, 'position', None)
    vel = getattr(entity, 'velocity', None)
    omega = getattr(entity, 'omega', None)
    
    altitude = pos[1] if pos is not None else None
    speed = float(np.linalg.norm(vel)) if vel is not None else None
    omega_mag = float(np.linalg.norm(omega)) if omega is not None else None
    
    # Calculate bearing from velocity if available
    bearing = None
    if vel is not None:
        bearing = float(np.degrees(np.arctan2(vel[0], vel[2])))
    
    # Calculate angle of attack, sideslip, Cd, and Cl using physics functions
    aoa = None
    sideslip = None
    cd = None
    cl = None
    cl_side = None
    air_density = None
    if hasattr(entity, 'orientation') and vel is not None:
        try:
            import physics
            aoa_raw = physics.get_angle_of_attack(vel, entity.orientation)
            aoa = float(aoa_raw)  # Try without conversion first
            sideslip_raw = physics.get_sideslip(vel, entity.orientation)
            sideslip = float(sideslip_raw)
            cd = physics.get_drag_coefficient(aoa, entity.min_drag_coefficient, entity.max_drag_coefficient)
            cl = physics.get_lift_coefficient(aoa, entity.max_lift_coefficient, entity.optimal_lift_aoa, -2.0)
            cl_side = physics.get_lift_coefficient(sideslip, entity.max_lift_coefficient, entity.optimal_lift_aoa, 0.0)
            if altitude is not None:
                air_density = physics.get_air_density(altitude)
        except Exception:
            pass
    
    # Format the display text
    bearing_str = f"{bearing:.1f}°" if bearing is not None else "N/A"
    altitude_str = f"{altitude:.1f} m" if altitude is not None else "N/A"
    air_density_str = f"{air_density:.4f} kg/m³" if air_density is not None else "N/A"
    speed_str = f"{speed:.1f} m/s" if speed is not None else "N/A"
    omega_str = f"{omega_mag:.3f} rad/s" if omega_mag is not None else "N/A"
    aoa_str = f"{aoa:.1f}°" if aoa is not None else "N/A"
    sideslip_str = f"{sideslip:.1f}°" if sideslip is not None else "N/A"
    cd_str = f"{cd:.3f}" if cd is not None else "N/A"
    cl_str = f"{cl:.3f}" if cl is not None else "N/A"
    cl_side_str = f"{cl_side:.3f}" if cl_side is not None else "N/A"
    
    return (
        "Live Data\n"
        f"Bearing: {bearing_str}\n"
        f"Altitude: {altitude_str}\n"
        f"Air density: {air_density_str}\n"
        f"Speed |V|: {speed_str}\n"
        f"Omega |ω|: {omega_str}\n"
        f"AoA: {aoa_str}\n"
        f"Sideslip: {sideslip_str}\n"
        f"Cd: {cd_str}\n"
        f"Cl: {cl_str}\n"
        f"Cl_side: {cl_side_str}\n"
        "Closest missile dist: N/A\n"
        "Status: Active"
    )


def _make_hud_label(canvas: vp.canvas, text: str, pos_px: vp.vector, *, height: int = 16, align: str = "right") -> Any:
    lbl = vp.label(
        canvas=canvas,
        pos=pos_px,
        text=text,
        height=height,
        font="monospace",
        color=vp.color.white,
        opacity=0.0,
        box=False,
        line=False,
        align=align,
    )
    lbl.pixel_pos = True
    return lbl


def _make_box_label(canvas: vp.canvas, text: str, pos_px: vp.vector, *, height: int = 18, align: str = "center") -> Any:
    lbl = vp.label(
        canvas=canvas,
        pos=pos_px,
        text=text,
        height=height,
        border=8,
        box=True,
        line=False,
        opacity=0.35,
        background=vp.color.black,
        font="monospace",
        color=vp.color.white,
        align=align,
    )
    lbl.pixel_pos = True
    return lbl


def _place_hud() -> None:
    if _main_canvas is not None:
        w = int(getattr(_main_canvas, "width", 1280))
        h = int(getattr(_main_canvas, "height", 900))
        if _main_status is not None:
            _main_status.pos = vp.vector(w / 2, h - 22, 0)
        if _main_info is not None:
            _main_info.pos = vp.vector(12, h - 24, 0)
        if _jet_intercepted_label is not None:
            _jet_intercepted_label.pos = vp.vector(w / 2, h / 2, 0)

    if _forces_canvas is not None:
        w = int(getattr(_forces_canvas, "width", 640))
        h = int(getattr(_forces_canvas, "height", 900))
        if _forces_status is not None:
            _forces_status.pos = vp.vector(w / 2, h - 22, 0)
        if _jet_static is not None:
            _jet_static.pos = vp.vector(w - 12, h - 24, 0)
        if _jet_dynamic is not None:
            _jet_dynamic.pos = vp.vector(w - 12, 320, 0)


def initialize_viz(sim_config: Optional[Dict[str, Any]] = None) -> None:
    global _main_canvas, _forces_canvas, _canvases_created
    global _main_status, _forces_status, _main_info, _jet_static, _jet_dynamic, _reward_display
    global _sim_config

    if isinstance(sim_config, dict):
        _sim_config = dict(sim_config)

    if not _canvases_created:
        main_w, forces_w, h = 1250, 670, 1000

        _main_canvas = vp.scene
        _main_canvas.align = "left"
        _main_canvas.width = main_w
        _main_canvas.height = h
        _main_canvas.resizable = False
        _main_canvas.title = (
            "<style>body{background:#000;margin:0;overflow:hidden}</style>"
            "<span style='font-family:system-ui;font-weight:850;font-size:24px;letter-spacing:.4px'>"
            "Aerodynamics Physics Engine</span> "
            "<span style='font-family:system-ui;font-size:13px;opacity:.75'>by Tristan Larue</span><br>"
        )
        _main_canvas.caption = ""
        _main_canvas.background = vp.color.black
        _main_canvas.autoscale = False
        _main_canvas.range = 2600
        _main_canvas.forward = vp.vector(-1, -0.22, -1)
        _main_canvas.up = vp.vector(0, 1, 0)
        _main_canvas.lights = []
        vp.distant_light(canvas=_main_canvas, direction=vp.vector(-1, -1, -1), color=vp.color.gray(0.9))
        vp.distant_light(canvas=_main_canvas, direction=vp.vector(1, 1, 1), color=vp.color.gray(0.6))

        _forces_canvas = vp.canvas(
            align="left",
            width=forces_w,
            height=h,
            title="",
            caption="",
            background=vp.color.black,
            resizable=False,
        )
        _forces_canvas.autoscale = False
        _forces_canvas.range = 1200
        _forces_canvas.forward = vp.vector(-1, -0.4, -1)
        _forces_canvas.up = vp.vector(0, 1, 0)
        _forces_canvas.lights = []
        vp.distant_light(canvas=_forces_canvas, direction=vp.vector(-1, -1, -1), color=vp.color.gray(0.9))
        vp.distant_light(canvas=_forces_canvas, direction=vp.vector(1, 1, 1), color=vp.color.gray(0.6))

        _canvases_created = True

    assert _main_canvas is not None
    assert _forces_canvas is not None

    _main_canvas.select()
    
    # Import simulation to get box size
    import simulation
    box_size = simulation.SIMULATION_BOX_SIZE
    ground_size_x = float(box_size[0] * 2)  # Full width of simulation box
    ground_size_z = float(box_size[2] * 2)  # Full depth of simulation box
    
    _static_objects.append(vp.box(canvas=_main_canvas, pos=vp.vector(0, 0, 0), size=vp.vector(ground_size_x, 1, ground_size_z), color=vp.color.gray(0.12)))
    
    # Add simulation boundary box lines
    half_x = float(box_size[0])
    half_y = float(box_size[1])
    half_z = float(box_size[2])
    
    # Bottom rectangle
    _static_objects.append(vp.curve(canvas=_main_canvas, pos=[vp.vector(-half_x, 0, -half_z), vp.vector(half_x, 0, -half_z), vp.vector(half_x, 0, half_z), vp.vector(-half_x, 0, half_z), vp.vector(-half_x, 0, -half_z)], color=vp.color.white, radius=10))
    
    # Top rectangle
    _static_objects.append(vp.curve(canvas=_main_canvas, pos=[vp.vector(-half_x, half_y*2, -half_z), vp.vector(half_x, half_y*2, -half_z), vp.vector(half_x, half_y*2, half_z), vp.vector(-half_x, half_y*2, half_z), vp.vector(-half_x, half_y*2, -half_z)], color=vp.color.white, radius=10))
    
    # Vertical edges
    _static_objects.append(vp.curve(canvas=_main_canvas, pos=[vp.vector(-half_x, 0, -half_z), vp.vector(-half_x, half_y*2, -half_z)], color=vp.color.white, radius=10))
    _static_objects.append(vp.curve(canvas=_main_canvas, pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, half_y*2, -half_z)], color=vp.color.white, radius=10))
    _static_objects.append(vp.curve(canvas=_main_canvas, pos=[vp.vector(half_x, 0, half_z), vp.vector(half_x, half_y*2, half_z)], color=vp.color.white, radius=10))
    _static_objects.append(vp.curve(canvas=_main_canvas, pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, half_y*2, half_z)], color=vp.color.white, radius=10))

    _forces_canvas.select()
    _force_objects.append(vp.arrow(canvas=_forces_canvas, pos=vp.vector(0, 0, 0), axis=vp.vector(400, 0, 0), shaftwidth=8.0, color=vp.color.white))    # X axis - white
    _force_objects.append(vp.arrow(canvas=_forces_canvas, pos=vp.vector(0, 0, 0), axis=vp.vector(0, 400, 0), shaftwidth=8.0, color=vp.color.white))  # Y axis - white  
    _force_objects.append(vp.arrow(canvas=_forces_canvas, pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 400), shaftwidth=8.0, color=vp.color.white))   # Z axis - white
    
    # Create jet object at origin in forces canvas
    global _forces_jet_display, _force_arrows
    _forces_jet_display = _load_model_from_folder("Su57", canvas=_forces_canvas, color=vp.color.white, opacity=0.2)
    if _forces_jet_display is not None:
        _scale_loaded_model(_forces_jet_display, MODELS_SCALE)
    else:
        _forces_jet_display = vp.sphere(canvas=_forces_canvas, pos=vp.vector(0, 0, 0), radius=MODELS_SCALE, color=vp.color.white)
    _force_objects.append(_forces_jet_display)
    
    # Create force arrows with different colors and amplifiers
    force_config = {
        "gravity": {"color": vp.color.purple, "amplifier": 0.8},
        "thrust": {"color": vp.color.orange, "amplifier": 1.5},
        "drag": {"color": vp.color.red, "amplifier": 1.33},
        "lift": {"color": vp.color.cyan, "amplifier": 0.13},
        "sideforce": {"color": vp.color.magenta, "amplifier": 3.0},
        "elevator": {"color": vp.color.yellow, "amplifier": 0.27},
        "aileron_left": {"color": vp.color.green, "amplifier": 0.5},
        "aileron_right": {"color": vp.color.blue, "amplifier": 0.5},
        "rudder": {"color": vp.color.white, "amplifier": 0.3}
    }
    
    _force_arrows = {}
    for force_name, config in force_config.items():
        arrow = vp.arrow(
            canvas=_forces_canvas,
            pos=vp.vector(0, 0, 0),
            axis=vp.vector(0, 0, 0),
            shaftwidth=20.0,
            color=config["color"],
            visible=False  # Start invisible
        )
        label = vp.label(
            canvas=_forces_canvas,
            pos=vp.vector(0, 0, 0),
            text=force_name.replace('_', ' ').title(),
            color=config["color"],
            height=12,
            box=False,
            line=False,
            visible=False
        )
        _force_arrows[force_name] = {"arrow": arrow, "label": label, "amplifier": config["amplifier"]}
        _force_objects.append(arrow)
        _force_objects.append(label)

    _main_status = _make_box_label(_main_canvas, "MAIN SIMULATION\nEntities: N/A\nTick: N/A", vp.vector(_main_canvas.width / 2, _main_canvas.height - 22, 0), align="center")
    _forces_status = _make_box_label(_forces_canvas, "JET FORCES VIEW\nTelemetry: N/A\nForces: N/A", vp.vector(_forces_canvas.width / 2, _forces_canvas.height - 22, 0), align="center")
    _reward_display = _make_box_label(_main_canvas, "AI Current Reward:\n0.0", vp.vector(_main_canvas.width - 150, 80, 0), align="right")

    _main_info = _make_box_label(_main_canvas, _fmt_sim_info(), vp.vector(12, _main_canvas.height - 24, 0), height=16, align="left")

    _jet_static = _make_hud_label(_forces_canvas, _static_text(), vp.vector(_forces_canvas.width - 12, _forces_canvas.height - 24, 0), height=16, align="right")
    _jet_dynamic = _make_hud_label(_forces_canvas, _dynamic_text(), vp.vector(_forces_canvas.width - 12, 320, 0), height=16, align="right")

    _place_hud()


def _iter_objects(obj: Any) -> Sequence[Any]:
    if obj is None:
        return ()
    if isinstance(obj, (list, tuple)):
        return obj
    return (obj,)


def cleanup_viz() -> None:
    global _tick, _main_status, _forces_status, _main_info, _jet_static, _jet_dynamic, _reward_display, _last_jet_instance, _forces_jet_display, _force_arrows, _missile_lines, _exploded_missiles, _explosion_spheres, _jet_intercepted_label

    for ev in list(_entity_visuals.values()):
        for obj in _iter_objects(ev.body):
            try:
                if hasattr(obj, "clear_trail"):
                    obj.clear_trail()
            except Exception:
                pass
            try:
                obj.visible = False
            except Exception:
                pass
        if ev.heading is not None:
            try:
                ev.heading.visible = False
            except Exception:
                pass
    _entity_visuals.clear()

    # Clean up missile lines
    for line in list(_missile_lines.values()):
        try:
            line.visible = False
        except Exception:
            pass
    _missile_lines.clear()

    # Clean up explosion spheres
    for sphere in list(_explosion_spheres.values()):
        try:
            sphere.visible = False
        except Exception:
            pass
    _explosion_spheres.clear()
    _exploded_missiles.clear()

    for obj in list(_static_objects) + list(_force_objects):
        try:
            obj.visible = False
        except Exception:
            pass
    _static_objects.clear()
    _force_objects.clear()

    for lbl in [_main_status, _forces_status, _main_info, _jet_static, _jet_dynamic, _reward_display, _jet_intercepted_label]:
        if lbl is None:
            continue
        try:
            lbl.visible = False
        except Exception:
            pass

    _main_status = None
    _forces_status = None
    _main_info = None
    _jet_static = None
    _jet_dynamic = None
    _reward_display = None
    _last_jet_instance = None
    _forces_jet_display = None
    _force_arrows = {}
    _jet_intercepted_label = None

    _tick = 0

def _scale_loaded_model(obj: Any, s: float) -> None:
    if obj is None or abs(s - 1.0) < 1e-9:
        return
    for o in _iter_objects(obj):
        if hasattr(o, "size"):
            try:
                o.size = o.size * float(s)
            except Exception:
                pass

def _load_model_from_folder(compound_shape: str, *, canvas: vp.canvas, color: Any, opacity: float) -> Any | None:
    folder = _MODELS_ROOT / compound_shape
    loader_path = folder / "loader.py"
    if not loader_path.exists():
        return None

    mod = _LOADER_MODULES.get(compound_shape)
    if mod is None:
        name = f"_viz_model_{compound_shape}_{abs(hash(str(loader_path)))}"
        spec = importlib.util.spec_from_file_location(name, str(loader_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LOADER_MODULES[compound_shape] = mod

    fn = None
    for candidate in ("load_mesh", "load_model", "create_model", "build_model", "build", "make_model", "load"):
        f = getattr(mod, candidate, None)
        if callable(f):
            fn = f
            break
    if fn is None:
        return None

    for kwargs in (
        {"canvas": canvas, "color": color, "opacity": opacity},
        {"canvas": canvas, "color": color},
        {"canvas": canvas},
        {"scene": canvas, "color": color, "opacity": opacity},
        {"scene": canvas},
        {},
    ):
        try:
            return fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None
    return None


def _apply_trail(ev: Entity_visual, trail_radius: float, trail_retain: int = 800, trail_color: Optional[vp.vector] = None) -> None:
    """Attach a VPython trail that works for mesh compounds (uses attach_trail)."""
    # Clear old trails
    for t in list(ev.trails):
        try:
            if hasattr(t, 'clear'):
                t.clear()
        except Exception:
            pass
        try:
            t.visible = False
        except Exception:
            pass
    ev.trails.clear()

    # Attach to the first object only (avoids multiple duplicated trails for multi-part meshes)
    objs = list(_iter_objects(ev.body))
    if not objs:
        return
    o = objs[0]
    try:
        # Create trail with configurable parameters
        trail_kwargs = {
            'radius': float(trail_radius),
            'retain': int(trail_retain)
        }
        # Add color if specified (not all VPython versions support trail color)
        if trail_color is not None:
            try:
                trail_kwargs['color'] = trail_color
            except Exception:
                pass  # Color not supported, use default
        
        tr = vp.attach_trail(o, **trail_kwargs)
        ev.trails.append(tr)
    except Exception:
        pass

def _create_entity_visual(entity: Any) -> Entity_visual:
    global _last_jet_instance
    assert _main_canvas is not None

    vs = _get_viz_shape(entity)
    kind = _get_kind(entity)
    compound_shape = str(vs.get("compound_shape", kind))

    col = vs.get("color", None)
    color = _coerce_color(col)
    opacity = float(vs.get("opacity", 1.0))

    make_trail = bool(vs.get("make_trail", False))
    trail_radius = float(vs.get("trail_radius", 25.0))
    trail_retain = int(vs.get("trail_retain", 800))
    trail_color = vs.get("trail_color", None)
    if trail_color is None:
        trail_color = color  # Use entity color as default
    else:
        trail_color = _coerce_color(trail_color)
    length = float(vs.get("length", 200.0))

    _main_canvas.select()
    body = _load_model_from_folder(compound_shape, canvas=_main_canvas, color=color, opacity=opacity)


    # 500x visualization scale for imported mesh models
    # Reduce Jet scale by 50%
    scale_factor = MODELS_SCALE * 0.5 if kind == "jet" else MODELS_SCALE
    if body is not None:
        _scale_loaded_model(body, scale_factor)
    if body is None:
        body = vp.sphere(canvas=_main_canvas, pos=vp.vector(0, 0, 0), radius=scale_factor, color=color, opacity=opacity)

    # Trails: attach_trail works for mesh compounds

    # Also load the jet model in the forces canvas
    body_forces = None
    if kind == "jet" and _forces_canvas is not None:
        try:
            _forces_canvas.select()
        except Exception:
            pass
        body_forces = _load_model_from_folder(compound_shape, canvas=_forces_canvas, color=color, opacity=opacity)
        if body_forces is not None:
            _scale_loaded_model(body_forces, scale_factor)  # Use same 50% scale for jet
        if body_forces is None:
            body_forces = vp.sphere(canvas=_forces_canvas, pos=vp.vector(0, 0, 0), radius=scale_factor, color=color, opacity=opacity)

    # No heading arrow created

    ev = Entity_visual(body=body, kind=kind, length=length, heading=None, body_forces=body_forces)
    if make_trail:
        # Set all trails to permanent retention (use very large number)
        permanent_retain = 999999
        _apply_trail(ev, trail_radius, permanent_retain, trail_color)
    
    # Track last jet instance for camera centering
    if kind == "jet":
        _last_jet_instance = entity
    
    return ev


def _set_pose(obj: Any, center: vp.vector, axis_dir: np.ndarray, up_dir: np.ndarray, length: float) -> None:
    if isinstance(obj, vp.cylinder):
        axis_v = _vpvec(axis_dir * float(length))
        obj.axis = axis_v
        obj.pos = center - axis_v * 0.5
        if hasattr(obj, "up"):
            obj.up = _vpvec(up_dir)
        return

    try:
        obj.pos = center
    except Exception:
        pass
    if hasattr(obj, "axis"):
        try:
            obj.axis = _vpvec(axis_dir)
        except Exception:
            pass
    if hasattr(obj, "up"):
        try:
            obj.up = _vpvec(up_dir)
        except Exception:
            pass


def _update_force_arrows(entity: Any) -> None:
    """Update force arrows based on entity's current forces in body frame."""
    if not _force_arrows or entity is None:
        return
        
    try:
        import physics
        
        # Calculate all forces (similar to entity's tick method)
        air_density = physics.get_air_density(entity.position[1])
        aoa = physics.get_angle_of_attack(entity.velocity, entity.orientation)
        sideslip = physics.get_sideslip(entity.velocity, entity.orientation)
        cd = physics.get_drag_coefficient(aoa, entity.min_drag_coefficient, entity.max_drag_coefficient)
        cl = physics.get_lift_coefficient(aoa, entity.max_lift_coefficient, entity.optimal_lift_aoa, -2.0)
        side_cl = physics.get_lift_coefficient(sideslip, entity.max_lift_coefficient, entity.optimal_lift_aoa, 0.0)
        side_surface_area = entity.reference_area * 0.1  # Approximate area of vertical stabilizer
        
        # Calculate forces in world frame, then convert to body frame for display
        R_inv = entity.orientation.T  # Transpose for world->body conversion
        
        forces_data = {
            "gravity": {
                "force": physics.get_gravity_force(entity.mass),
                "offset": np.array([0.0, 0.0, 0.0])
            },
            "thrust": {
                "force": physics.get_thrust_force(entity.orientation, entity.throttle, entity.thrust_force, entity.length),
                "offset": np.array([-12.0 * entity.length, 0.0, 0.0])
            },
            "drag": {
                "force": physics.get_drag_force(entity.velocity, air_density, entity.reference_area, cd),
                "offset": np.array([-4.8 * entity.length, 0.0, 0.0])
            },
            "lift": {
                "force": physics.get_lift_force(entity.velocity, entity.reference_area, cl, entity.orientation, air_density),
                "offset": np.array([-4.8 * entity.length, 0.0, 0.0])
            },
            "sideforce": {
                "force": physics.get_sideforce_force(entity.velocity, side_surface_area, side_cl, entity.orientation, air_density),
                "offset": np.array([-10.8 * entity.length, 0.0, 0.0])
            },
            "elevator": {
                "force": physics.get_elevator_force(entity.velocity, air_density, entity.reference_area, entity.max_lift_coefficient, entity.orientation, entity.control_inputs["pitch"], entity.optimal_lift_aoa, 0.30, 1.40),
                "offset": np.array([-10.8 * entity.length, 0.0, 0.0])
            },
            "rudder": {
                "force": physics.get_rudder_force(entity.velocity, air_density, entity.reference_area, entity.max_lift_coefficient, entity.orientation, entity.control_inputs["yaw"], entity.optimal_lift_aoa, 0.03, 0.30),
                "offset": np.array([-10.8 * entity.length, 0.0, 0.0])
            }
        }
        
        # Handle aileron forces (returns single force, opposite aileron is negated)
        aileron_force = physics.get_aileron_force(entity.velocity, air_density, entity.reference_area, entity.max_lift_coefficient, entity.orientation, entity.control_inputs["roll"], entity.optimal_lift_aoa, 0.04, 0.6)
        forces_data["aileron_left"] = {
            "force": aileron_force,
            "offset": np.array([-1.2 * entity.length, 0.0, 12.0 * entity.wingspan])
        }
        forces_data["aileron_right"] = {
            "force": -aileron_force, 
            "offset": np.array([-1.2 * entity.length, 0.0, -12.0 * entity.wingspan])
        }
        
        # Update each force arrow
        for force_name, force_data in forces_data.items():
            if force_name not in _force_arrows:
                continue
                
            force_world = force_data["force"]
            offset_body = force_data["offset"]
            
            # Convert force from world frame to body frame for display
            force_body = R_inv @ force_world
            force_magnitude = float(np.linalg.norm(force_body))
            
            arrow_info = _force_arrows[force_name]
            arrow = arrow_info["arrow"]
            label = arrow_info["label"]
            amplifier = arrow_info["amplifier"]
            
            if force_magnitude > 1e-6:  # Only show if force is significant
                # Calculate arrow length using square root of magnitude with amplifier
                arrow_length = np.sqrt(force_magnitude) * amplifier
                force_direction = force_body / force_magnitude
                
                # Position arrow at force application point (base of arrow)
                arrow.pos = _vpvec(offset_body)
                # Arrow points in direction of force
                arrow.axis = _vpvec(force_direction * arrow_length)
                arrow.visible = True
                
                # Position label at arrow tip
                tip_position = offset_body + (force_direction * arrow_length)
                label.pos = _vpvec(tip_position)
                label.visible = True
            else:
                arrow.visible = False
                label.visible = False
                
    except Exception as e:
        # Hide all arrows and labels if calculation fails
        for arrow_info in _force_arrows.values():
            arrow_info["arrow"].visible = False
            arrow_info["label"].visible = False


def _update_entity_visual(ev: Entity_visual, entity: Any) -> None:
    p = _extract_pos(entity)
    if p is None:
        return
    center = _vpvec(p)

    R = _extract_R(entity)
    if R is None:
        for obj in _iter_objects(ev.body):
            try:
                obj.pos = center
            except Exception:
                pass
        for obj in _iter_objects(ev.body_forces):
            try:
                obj.pos = center
            except Exception:
                pass
        return

    axis_dir = _safe_unit(R @ np.array([1.0, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))
    up_dir = _safe_unit(R @ np.array([0.0, 1.0, 0.0], dtype=float), np.array([0.0, 1.0, 0.0], dtype=float))

    for obj in _iter_objects(ev.body):
        _set_pose(obj, center, axis_dir, up_dir, ev.length)

    # Mirror jet pose in forces canvas
    for obj in _iter_objects(ev.body_forces):
        _set_pose(obj, center, axis_dir, up_dir, ev.length)


def update_instances(entities: Sequence[Any]) -> None:
    global _tick, _jet_intercepted_label

    if not _canvases_created:
        try:
            import simulation as sim

            initialize_viz(
                {
                    "duration_s": getattr(sim, "SIMULATION_DURATION", "N/A"),
                    "tick_rate": getattr(sim, "SIMULATION_TICKRATE", "N/A"),
                    "sim_speed": getattr(sim, "SIMULATION_SPEED", "N/A"),
                    "box_size": getattr(sim, "SIMULATION_BOX_SIZE", "N/A"),
                    "resolution": getattr(sim, "SIMULATION_RESOLUTION", "N/A"),
                }
            )
        except Exception:
            initialize_viz({})

    if _main_status is None or not _static_objects or not _force_objects:
        initialize_viz(_sim_config)

    assert _main_canvas is not None

    _tick += 1

    seen: set[int] = set()
    focus_entity: Any = None  # Track the entity with FOCUS_ENTITY_ID for forces display
    jet_entity: Any = None
    jet_vid: int = -1
    jet_alive: bool = True

    # Find jet entity first (regardless of alive status)
    for e in entities:
        if e.__class__.__name__.lower() == "jet":
            jet_entity = e
            jet_vid = _get_viz_id(e)
            jet_alive = bool(getattr(e, "alive", True))
            break

    for e in entities:
        vid = _get_viz_id(e)
        seen.add(vid)
        
        # Track focus entity for forces display
        if vid == FOCUS_ENTITY_ID:
            focus_entity = e

        ev = _entity_visuals.get(vid)
        if ev is None:
            ev = _create_entity_visual(e)
            _entity_visuals[vid] = ev
        
        # Check if this is a missile entity
        is_missile = (ev.kind == "missile" or 
                     e.__class__.__name__.lower() == "missile" or
                     "missile" in e.__class__.__name__.lower())
        
        if is_missile:
            # Get alive status - be very explicit about checking
            # Default to True if attribute doesn't exist (assume alive)
            is_alive = bool(getattr(e, "alive", True))
            
            # Ensure we're working on the main canvas (left canvas) for missiles
            # Missiles only exist on main canvas, not forces canvas
            if _main_canvas is not None:
                _main_canvas.select()
            
            # Keep exploded missiles invisible
            if vid in _exploded_missiles:
                # Already exploded - ensure it stays invisible on main canvas
                def _keep_invisible(obj):
                    """Recursively keep opacity at 0 and visibility False."""
                    if obj is None:
                        return
                    try:
                        if hasattr(obj, "opacity"):
                            obj.opacity = 0.0
                        if hasattr(obj, "visible"):
                            obj.visible = False
                    except Exception:
                        pass
                    if isinstance(obj, (list, tuple)):
                        for sub_obj in obj:
                            _keep_invisible(sub_obj)
                
                # Only modify body on main canvas (missiles don't have body_forces)
                if ev.body is not None:
                    for obj in _iter_objects(ev.body):
                        _keep_invisible(obj)
            elif not is_alive:
                # Missile just exploded - set opacity to 0 and create explosion sphere
                if vid not in _exploded_missiles:
                    _exploded_missiles.add(vid)
                    missile_pos = _extract_pos(e)
                    
                    # Set missile opacity to 0 and visibility to False to make it disappear
                    # Handle both single objects and compound models (lists/tuples)
                    # Only modify objects on main canvas
                    def _set_invisible(obj):
                        """Recursively set opacity and visibility on object and nested objects."""
                        if obj is None:
                            return
                        try:
                            if hasattr(obj, "opacity"):
                                obj.opacity = 0.0
                            if hasattr(obj, "visible"):
                                obj.visible = False
                        except Exception:
                            pass
                        # If it's a compound (list/tuple), recurse
                        if isinstance(obj, (list, tuple)):
                            for sub_obj in obj:
                                _set_invisible(sub_obj)
                    
                    # Only modify body on main canvas (missiles don't have body_forces)
                    if ev.body is not None:
                        for obj in _iter_objects(ev.body):
                            _set_invisible(obj)
                    
                    # Remove missile line if it exists (lines are on main canvas)
                    if vid in _missile_lines:
                        try:
                            _missile_lines[vid].visible = False
                            del _missile_lines[vid]
                        except Exception:
                            pass
                    
                    # Create permanent orange sphere at explosion location on main canvas
                    # Adjust position by subtracting velocity to get position from previous tick
                    if missile_pos is not None and _main_canvas is not None:
                        try:
                            # Get missile velocity and tick rate
                            missile_vel = _extract_vel(e)
                            tick_rate = _sim_config.get("tick_rate", None)
                            
                            # Fallback: try to get tick rate from simulation module
                            if tick_rate is None or tick_rate == "N/A":
                                try:
                                    import simulation as sim
                                    tick_rate = getattr(sim, "SIMULATION_TICKRATE", None)
                                except Exception:
                                    pass
                            
                            # Calculate adjusted position: subtract one tick's worth of movement
                            explosion_pos = np.array(missile_pos, dtype=float)
                            if missile_vel is not None and tick_rate is not None:
                                try:
                                    tick_rate_float = float(tick_rate)
                                    if tick_rate_float > 0:
                                        dt = 1.0 / tick_rate_float
                                        # Subtract velocity * dt to get position from previous tick
                                        explosion_pos = explosion_pos - (missile_vel * dt)
                                except (ValueError, TypeError, ZeroDivisionError):
                                    # If calculation fails, use current position
                                    pass
                            
                            # Ensure we're on the main canvas (left canvas)
                            _main_canvas.select()
                            
                            # Create the explosion sphere on main canvas at adjusted position
                            explosion_sphere = vp.sphere(
                                canvas=_main_canvas,
                                pos=_vpvec(explosion_pos),
                                radius=30.0,  # Size of about 30 as requested
                                color=vp.color.orange,
                                opacity=1.0
                            )
                            # Explicitly set visibility
                            explosion_sphere.visible = True
                            
                            # Store references
                            _explosion_spheres[vid] = explosion_sphere
                            _static_objects.append(explosion_sphere)
                        except Exception:
                            # If sphere creation fails, continue anyway
                            pass
            elif is_alive:
                # Missile is still alive - update visuals based on distance to jet
                # Ensure we're on main canvas (missiles only exist on main canvas)
                if _main_canvas is not None:
                    _main_canvas.select()
                
                # Make sure it's visible on main canvas
                if ev.body is not None:
                    for obj in _iter_objects(ev.body):
                        try:
                            if hasattr(obj, "opacity"):
                                obj.opacity = 1.0
                            if hasattr(obj, "visible"):
                                obj.visible = True
                        except Exception:
                            pass
                
                if jet_entity is not None:
                    missile_pos = _extract_pos(e)
                    jet_pos = _extract_pos(jet_entity)
                    if missile_pos is not None and jet_pos is not None:
                        import physics
                        distance = float(np.linalg.norm(missile_pos - jet_pos))
                        kill_range = 30.0  # explosion_radius from missile entity
                        in_kill_range = distance < kill_range
                        
                        for obj in _iter_objects(ev.body):
                            try:
                                if hasattr(obj, "radius"):
                                    if in_kill_range:
                                        obj.radius = MODELS_SCALE * 1.5  # Large orange sphere
                                    else:
                                        obj.radius = MODELS_SCALE * 0.3  # Small white sphere
                                if hasattr(obj, "color"):
                                    if in_kill_range:
                                        obj.color = vp.color.orange
                                    else:
                                        obj.color = vp.color.white
                            except Exception:
                                pass
        
        # Skip visual updates for exploded missiles to prevent them from becoming visible again
        if ev.kind != "missile" or vid not in _exploded_missiles:
            _update_entity_visual(ev, e)
        
        # Center camera on entity with FOCUS_ENTITY_ID
        if vid == FOCUS_ENTITY_ID:
            p = _extract_pos(e)
            if p is not None:
                _main_canvas.center = vp.vector(float(p[0]), float(p[1]), float(p[2]))

    # Update lines connecting jet to all active missiles - update positions every tick
    # Use cylinder objects instead of curves for reliable position updates in VPython
    if jet_entity is not None and jet_vid >= 0:
        active_missile_vids = set()
        
        # Iterate through entities and update/create lines immediately with fresh positions each tick
        for e in entities:
            if e.__class__.__name__.lower() == "missile" and bool(getattr(e, "alive", True)):
                missile_vid = _get_viz_id(e)
                active_missile_vids.add(missile_vid)
                
                # Get fresh positions right before updating line (ensures positions are current each tick)
                fresh_jet_pos = _extract_pos(jet_entity)
                fresh_missile_pos = _extract_pos(e)
                
                if fresh_jet_pos is not None and fresh_missile_pos is not None:
                    # Calculate direction vector from jet to missile
                    direction = fresh_missile_pos - fresh_jet_pos
                    distance = float(np.linalg.norm(direction))
                    
                    if distance > 1e-6:
                        if missile_vid not in _missile_lines:
                            # Create new cylinder line with current positions
                            # VPython cylinders: pos is start point, axis is the vector to end point
                            try:
                                line = vp.cylinder(
                                    canvas=_main_canvas,
                                    pos=_vpvec(fresh_jet_pos),
                                    axis=_vpvec(direction),
                                    radius=5.0,
                                    color=vp.color.white,
                                    opacity=0.4
                                )
                                _missile_lines[missile_vid] = line
                            except Exception:
                                pass
                        else:
                            # Update existing cylinder line with fresh positions from current tick
                            # VPython cylinders: pos is start point, axis is the vector to end point
                            try:
                                line = _missile_lines[missile_vid]
                                line.pos = _vpvec(fresh_jet_pos)
                                line.axis = _vpvec(direction)
                                line.opacity = 0.4
                                line.visible = True
                            except Exception:
                                pass
        
        # Remove lines for missiles that no longer exist or are not alive
        for mid in list(_missile_lines.keys()):
            if mid not in active_missile_vids:
                try:
                    _missile_lines[mid].visible = False
                    del _missile_lines[mid]
                except Exception:
                    pass

    for vid in list(_entity_visuals.keys()):
        if vid in seen:
            continue
        ev = _entity_visuals.pop(vid, None)
        if ev is None:
            continue
        for obj in _iter_objects(ev.body):
            try:
                if hasattr(obj, "clear_trail"):
                    obj.clear_trail()
            except Exception:
                pass
            try:
                obj.visible = False
            except Exception:
                pass
        if ev.heading is not None:
            try:
                ev.heading.visible = False
            except Exception:
                pass
        # Remove missile line if entity is removed
        if vid in _missile_lines:
            try:
                _missile_lines[vid].visible = False
                del _missile_lines[vid]
            except Exception:
                pass

    if _main_status is not None:
        _main_status.text = f"MAIN SIMULATION\nEntities: {len(seen)}\nTick: {_tick}"

    if _forces_status is not None:
        status_text = f"JET FORCES VIEW\nTracking Entity: {FOCUS_ENTITY_ID}\nTelemetry: {'Active' if focus_entity else 'No Data'}"
        _forces_status.text = status_text

    # Update forces canvas jet orientation and text displays
    if focus_entity is not None and _forces_jet_display is not None:
        # Keep jet at 0,0,0 in body frame (no orientation updates)
        # The jet represents the body reference frame, not world frame
        
        # Update force arrows with current entity forces
        _update_force_arrows(focus_entity)
        
        # Update text displays with real data
        if _jet_static is not None:
            _jet_static.text = _static_text(focus_entity)
        if _jet_dynamic is not None:
            _jet_dynamic.text = _dynamic_text(focus_entity)
    else:
        # No focus entity, show N/A data and hide force arrows
        for arrow_info in _force_arrows.values():
            arrow_info["arrow"].visible = False
        if _jet_static is not None:
            _jet_static.text = _static_text()
        if _jet_dynamic is not None:
            _jet_dynamic.text = _dynamic_text()

    if _main_info is not None:
        _main_info.text = _fmt_sim_info()

    # Update reward display
    if _reward_display is not None:
        current_reward = 0.0
        for ent in entities:
            if hasattr(ent, 'current_reward'):
                current_reward = float(getattr(ent, 'current_reward', 0.0))
                break
        _reward_display.text = f"AI Current Reward:\n{current_reward:.3f}"

    # Show/hide "JET INTERCEPTED" message on main canvas
    if jet_entity is not None and not jet_alive:
        # Jet is intercepted - show red box in center of main canvas
        if _jet_intercepted_label is None:
            # Create the label if it doesn't exist
            w = int(getattr(_main_canvas, "width", 1280))
            h = int(getattr(_main_canvas, "height", 900))
            _jet_intercepted_label = vp.label(
                canvas=_main_canvas,
                pos=vp.vector(w / 2, h / 2, 0),
                text="JET INTERCEPTED",
                height=36,
                border=12,
                box=True,
                line=False,
                opacity=0.9,
                background=vp.color.red,
                font="monospace",
                color=vp.color.white,
                align="center",
            )
            _jet_intercepted_label.pixel_pos = True
        else:
            # Label exists, make sure it's visible
            _jet_intercepted_label.visible = True
    else:
        # Jet is alive or doesn't exist - hide the label
        if _jet_intercepted_label is not None:
            _jet_intercepted_label.visible = False

    _place_hud()


def setup_viz() -> None:
    try:
        import simulation as sim

        initialize_viz(
            {
                "duration_s": getattr(sim, "SIMULATION_DURATION", "N/A"),
                "tick_rate": getattr(sim, "SIMULATION_TICKRATE", "N/A"),
                "sim_speed": getattr(sim, "SIMULATION_SPEED", "N/A"),
                "box_size": getattr(sim, "SIMULATION_BOX_SIZE", "N/A"),
                "resolution": getattr(sim, "SIMULATION_RESOLUTION", "N/A"),
            }
        )
    except Exception:
        initialize_viz({})


def clean_viz() -> None:
    cleanup_viz()


def create_viz_instance(viz_shape: Dict[str, Any], entity: Any) -> Viz_instance:
    kind = str(viz_shape.get("compound_shape", _get_kind(entity)))
    return Viz_instance(kind=kind, viz_shape=dict(viz_shape))


# Runtime trail control functions
def set_entity_trail(entity_id: int, enabled: bool, radius: float = 25.0, retain: int = 800, color: Optional[vp.vector] = None) -> bool:
    """Enable or disable trails for a specific entity at runtime.
    
    Args:
        entity_id: The viz_id of the entity
        enabled: Whether to enable or disable the trail
        radius: Trail radius (used when enabling)
        retain: Number of trail points to retain (used when enabling)
        color: Trail color (used when enabling, None for entity color)
    
    Returns:
        True if operation succeeded, False otherwise
    """
    ev = _entity_visuals.get(entity_id)
    if ev is None:
        return False
    
    if enabled and not ev.trails:
        # Enable trail
        _apply_trail(ev, radius, retain, color)
        return True
    elif not enabled and ev.trails:
        # Disable trail
        _clear_entity_trails(ev)
        return True
    
    return True  # Already in desired state


def clear_entity_trail(entity_id: int) -> bool:
    """Clear trails for a specific entity.
    
    Args:
        entity_id: The viz_id of the entity
    
    Returns:
        True if operation succeeded, False otherwise
    """
    ev = _entity_visuals.get(entity_id)
    if ev is None:
        return False
    
    _clear_entity_trails(ev)
    return True


def _clear_entity_trails(ev: Entity_visual) -> None:
    """Internal function to clear trails from an Entity_visual."""
    for t in list(ev.trails):
        try:
            if hasattr(t, 'clear'):
                t.clear()
        except Exception:
            pass
        try:
            t.visible = False
        except Exception:
            pass
    ev.trails.clear()


def get_entity_trail_status(entity_id: int) -> Dict[str, Any]:
    """Get trail status information for an entity.
    
    Args:
        entity_id: The viz_id of the entity
    
    Returns:
        Dictionary with trail status information
    """
    ev = _entity_visuals.get(entity_id)
    if ev is None:
        return {"exists": False, "has_trail": False, "trail_count": 0}
    
    return {
        "exists": True,
        "has_trail": len(ev.trails) > 0,
        "trail_count": len(ev.trails),
        "kind": ev.kind
    }


def clear_all_trails() -> int:
    """Clear all trails from all entities.
    
    Returns:
        Number of entities that had trails cleared
    """
    cleared_count = 0
    for ev in _entity_visuals.values():
        if ev.trails:
            _clear_entity_trails(ev)
            cleared_count += 1
    return cleared_count


def get_trail_statistics() -> Dict[str, Any]:
    """Get statistics about trails in the visualization.
    
    Returns:
        Dictionary with trail statistics
    """
    total_entities = len(_entity_visuals)
    entities_with_trails = sum(1 for ev in _entity_visuals.values() if ev.trails)
    total_trail_objects = sum(len(ev.trails) for ev in _entity_visuals.values())
    
    trails_by_kind: Dict[str, int] = {}
    for ev in _entity_visuals.values():
        if ev.trails:
            trails_by_kind[ev.kind] = trails_by_kind.get(ev.kind, 0) + 1
    
    return {
        "total_entities": total_entities,
        "entities_with_trails": entities_with_trails,
        "total_trail_objects": total_trail_objects,
        "trails_by_kind": trails_by_kind
    }
