import numpy as np
from config import AIR_DENSITY, TICK_RATE
import math
'''=====================================================================
8888888b.  888    888 Y88b   d88P  .d8888b. 8888888 .d8888b.   .d8888b.  
888   Y88b 888    888  Y88b d88P  d88P  Y88b  888  d88P  Y88b d88P  Y88b 
888    888 888    888   Y88o88P   Y88b.       888  888    888 Y88b.      
888   d88P 8888888888    Y888P     "Y888b.    888  888         "Y888b.   
8888888P"  888    888     888         "Y88b.  888  888            "Y88b. 
888        888    888     888           "888  888  888    888       "888 
888        888    888     888     Y88b  d88P  888  Y88b  d88P Y88b  d88P 
888        888    888     888      "Y8888P" 8888888 "Y8888P"   "Y8888P"
====================================================================='''
# What we're all really here for: PHYSICS CALCULATIONS
# Recreating real aerodynamics as a student is WAY out of scope so here are some...
# GROUND RULES FOR PHYSICS SIMPLIFICATIONS:
# 1. The pilot, no matter the amount of G-forces he can encounter, is a superhuman and shall not die.
# 2. The jets and missiles have been made from materials unbeknownst to mankind and are indestructible.
# 3. No full rigid-body rotational dynamics (I use a diagonal inertia, no ω×Iω coupling, no gyroscopic effects).
# 4. No induced drag model (all drag is a simple Cd vs AoA simplification).
# 5. No real airfoil / wing planform (Cl–AoA and stall are hand-tuned curves, not thin airfoil theory).
# 6. No compressibility or Mach effects (no transonic drag rise, no supersonic behavior).
# 7. No standard atmosphere model (air density, temperature, and speed of sound are fixed).
# 8. No aeroelastic or structural effects (no G-limits, no wing bending, no flutter, no structural failure).
# 9. No control-surface aerodynamics or hinge moments (inputs map straight to ideal torques / angular response).
# 10. No strict energy conservation checks (simple integrator, numerical errors are tolerated).
# 11. No wind, turbulence, or gusts (air is perfectly still and uniform).
# 12. No fuel burn or moving mass / CG (mass and inertia are constant for each entity).
# 13. No ground effect, terrain collision, or detailed takeoff/landing physics.
# 14. No separate wings/tail/fins bodies (all aerodynamics are lumped into a single force + torque at a fixed CP).

# TODO: Implement an atmosphere model for wind, air density variation with altitude and temperature effects.
# TODO: ADD MACH EFFECTS WITH THE ATMOSPHERE MODEL!!!!
# ===========================================================================================================================

# Calculation of the rotation matrix for this tick
def get_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Convert to radians
    roll_rad  = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad   = np.radians(yaw)

    cr, sr = np.cos(roll_rad),  np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad),   np.sin(yaw_rad)

    # Roll: rotation around X (nose)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr]
    ])

    # Yaw: rotation around Y (up)
    Ry = np.array([
        [ cy, 0.0,  sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0,  cy]
    ])

    # Pitch: rotation around Z (right wing)
    Rz = np.array([
        [ cp, -sp, 0.0],
        [ sp,  cp, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Body → world: apply roll last, pitch in the middle, yaw first
    return Ry @ Rz @ Rx

# ANGLE OF ATTACK: Comparing the nose heading to the velocity vector
# Calculated only on the pitch axis for simplicity
def get_aoa(v: np.ndarray, R: np.ndarray) -> float:
    if np.linalg.norm(v) == 0.0:
        return 0.0 #Value doesn't matter
    # R: body -> world
    v_body = R.T @ v #Apply the velocity to the body's POV
    return -np.degrees(np.arctan2(v_body[1], v_body[0]))


# EXTREMELY SIMPLIFIED LIFT COEFFICIENT CALCULATION ONLY BASED ON AOA (LINEAR)
# Could be heavily improved in the future:
def get_cl(v: np.ndarray, R: np.ndarray, max_cl: float)-> float: 
    angle = get_aoa(v, R)
    if angle < -2.0:
        return 0.0
    elif angle <= 15.0:
        return (angle+2.0) / 17.0 * max_cl
    elif angle < 20.0:
        return max_cl * (1.0 - (angle - 15.0) / 5.0)
    else:
        return 0.0

# EXTREMELY SIMPLIFIED DRAG COEFFICIENT CALCULATION ONLY BASED ON AOA (QUADRATIC)
# Could be heavily improved in the future:
def get_cd(v: np.ndarray, R: np.ndarray, min_cd: float, max_cd: float) -> float:
    angle = get_aoa(v, R)
    range = (max_cd-min_cd)
    # Drag is explosive with AoA (quadratic)
    cd = range * ((1/8100) * angle**2) + min_cd
    return cd
    


# ==================== #
# =====| FORCES |===== #
# ==================== #

# Lift force, perpendicular to velocity and in "up" direction of plane
def get_lift_force(v: np.ndarray, reference_area: float, max_cl: float, R: np.ndarray) -> np.ndarray:
    lift_mag = 0.5 * AIR_DENSITY * np.linalg.norm(v)**2 * get_cl(v, R, max_cl) * reference_area
    if lift_mag == 0.0:
        return np.array([0.0, 0.0, 0.0])
    return lift_mag * get_lift_dir(v, R)

# Thrust force, in direction of the nose
def get_thrust_force(throttle: float, thrust_force: float, R: np.ndarray) -> np.ndarray:
    thrust_dir = get_forward_dir(R)
    return throttle * thrust_force * thrust_dir

# Drag force, opposite to velocity
def get_drag_force(v: np.ndarray, reference_area: float, min_cd: float, max_cd: float, R: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v)
    if v_mag == 0.0:
        return np.array([0.0, 0.0, 0.0])
    drag_mag = 0.5 * AIR_DENSITY * v_mag**2 * get_cd(v, R, min_cd, max_cd) * reference_area
    return drag_mag * -(v / v_mag)


# =========================== #
# =====| ACCELERATIONS |===== #
# =========================== #

# I dont think I have to explain gravity
def get_gravity_acc() -> np.ndarray:
    return np.array([0.0, -9.81, 0.0]) / TICK_RATE

# Lift acceleration, perpendicular to velocity and in "up" direction of plane
# Simplified due to the lift coefficient function
def get_lift_acc(v: np.ndarray, mass: float, reference_area: float, max_cl: float, R: np.ndarray) -> np.ndarray:
    lift_force = get_lift_force(v, reference_area, max_cl, R)
    return (lift_force / mass) / TICK_RATE
    
#Thrust acceleration, in direction of the nose
def get_thrust_acc(throttle: float, thrust: float, mass: float, R: np.ndarray) -> np.ndarray:
    thrust_force = get_thrust_force(throttle, thrust, R)
    return (thrust_force / mass) / TICK_RATE # type: ignore

# Drag deceleration, opposite to velocity
# Simplified due to the drag coefficient function
def get_drag_acc(v: np.ndarray, mass: float, reference_area: float, min_cd: float, max_cd: float, R: np.ndarray) -> np.ndarray:
    drag_force = get_drag_force(v, reference_area, min_cd, max_cd, R)
    return (drag_force / mass) / TICK_RATE





# =============================== #
# =====| DIRECTION VECTORS |===== #
# =============================== #
def get_forward_dir(R: np.ndarray) -> np.ndarray: #Returns the "forward" unit vector of the plane
    return R @ np.array([1.0, 0.0, 0.0])

def get_up_dir(R: np.ndarray) -> np.ndarray: #Returns the "up" unit vector of the plane
    return R @ np.array([0.0, 1.0, 0.0])

def get_right_dir(R: np.ndarray) -> np.ndarray: #Returns the "right" unit vector of the plane (kinda useful on edge cases)
    return R @ np.array([0.0, 0.0, 1.0])

def get_lift_dir(v: np.ndarray, R: np.ndarray) -> np.ndarray: #Returns the "up" unit vector of the plane perpendicular to velocity/airflow
    v_mag = np.linalg.norm(v)
    if v_mag == 0.0: #Useless since no V = no lift
        return get_up_dir(R)
    
    flow_dir = -v / v_mag #-vel_unit
    up_body = get_up_dir(R)           # body-up in world space
    # Now projecting up_body onto plane perpendicular to velocity
    l = up_body - np.dot(up_body, flow_dir) * flow_dir
    l_mag = np.linalg.norm(l)

    if l_mag == 0.0:
        #almost impossible case, but why not
        right_world = get_right_dir(R)
        l = np.cross(flow_dir, right_world)
        l_mag = np.linalg.norm(l)
        if l_mag == 0.0:
            return up_body  # absolute fallback
    return l / l_mag


# ==================== #
# =====| MOMENT |===== #
# ==================== #

def get_moment(force_world: np.ndarray, R: np.ndarray, distance: float, mass: float) -> tuple[float, float]:
    force_body = R.T @ force_world
    tau_body = np.cross(np.array([-distance, 0.0, 0.0]), force_body)
    # Approximate moment of inertia for a cylinder/aircraft: I ≈ (1/12) * m * L^2
    # Using a reasonable length scale based on distance to CP
    moment_of_inertia = mass * (abs(distance) * 10) ** 2  # Simplified inertia estimate
    
    # Convert torque to angular acceleration (degrees/s²) and integrate over timestep
    pitch_angular_acc = np.degrees(tau_body[2] / moment_of_inertia)
    yaw_angular_acc = np.degrees(tau_body[1] / moment_of_inertia)
    
    # Return angular velocity change for this tick as Python floats
    pitch_vel_change = float(pitch_angular_acc / TICK_RATE)
    yaw_vel_change = float(yaw_angular_acc / TICK_RATE)
    
    return pitch_vel_change, yaw_vel_change

def get_next_rotation(roll: float, pitch: float, yaw: float, roll_v: float, pitch_v: float, yaw_v: float) -> tuple[float, float, float]:
    new_pitch = (pitch + pitch_v / TICK_RATE) % 360
    new_yaw = (yaw + yaw_v / TICK_RATE) % 360
    new_roll = (roll + roll_v / TICK_RATE) % 360
    return new_roll, new_pitch, new_yaw

# ====================== #
# =====| POSITION |===== #
# ====================== #
def get_next_pos(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    return p + (v / TICK_RATE)