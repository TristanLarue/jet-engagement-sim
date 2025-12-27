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

# Matrix utility functions
def get_delta_rotation_matrix(omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Create small rotation matrix from body angular velocity."""
    # Convert deg/s to rad/s and scale by dt
    omega_rad = np.radians(omega_body) * dt
    
    # Small angle approximation for delta rotation matrix
    # R ≈ I + [omega]× (skew-symmetric matrix)
    wx, wy, wz = omega_rad
    return np.array([
        [1.0, -wz,  wy],
        [ wz, 1.0, -wx],
        [-wy,  wx, 1.0]
    ])

def orthonormalize_matrix(R: np.ndarray) -> np.ndarray:
    """Orthonormalize rotation matrix to prevent drift."""
    # Gram-Schmidt process
    x = R[:, 0]  # First column
    y = R[:, 1]  # Second column
    
    # Orthogonalize
    x = x / np.linalg.norm(x)
    y = y - np.dot(y, x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)  # Third column
    
    return np.column_stack([x, y, z])

def extract_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw from rotation matrix for display."""
    # Extract Euler angles (ZYX order)
    pitch = np.degrees(np.arcsin(-R[2, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw

# ANGLE OF ATTACK: Comparing the nose heading to the velocity vector
# Calculated only on the pitch axis for simplicity
def get_aoa(v: np.ndarray, R: np.ndarray) -> float:
    if np.linalg.norm(v) == 0.0:
        return 0.0 #Value doesn't matter
    # R: body -> world
    v_body = R.T @ v #Apply the velocity to the body's POV
    return -np.degrees(np.arctan2(v_body[1], v_body[0]))

# DYNAMIC PRESSURE: Fundamental aerodynamic parameter
def get_dynamic_pressure(v: np.ndarray) -> float:
    v_mag = np.linalg.norm(v)
    return 0.5 * AIR_DENSITY * v_mag**2


# LIFT COEFFICIENT: Based on angle of attack with configurable optimal point
def get_cl(aoa: float, max_cl: float, optimal_aoa: float = 15.0) -> float:
    stall_aoa = optimal_aoa * 1.2  # Stall occurs 20% beyond optimal
    
    if aoa >= 0:  # Positive AoA (nose up)
        if aoa <= optimal_aoa:
            return (aoa / optimal_aoa) * max_cl  # Linear rise to peak
        elif aoa <= stall_aoa:
            # Linear drop from max to zero over 20% range
            return max_cl * (1.0 - (aoa - optimal_aoa) / (stall_aoa - optimal_aoa))
        else:
            return 0.0  # Complete stall
    else:  # Negative AoA (inverted flight)
        aoa = abs(aoa)
        inverted_optimal = optimal_aoa * 1.2  # Inverted optimal is wider
        inverted_stall = optimal_aoa * 1.7   # Inverted stall even wider
        
        if aoa <= inverted_optimal:
            return -(aoa / inverted_optimal) * max_cl * 0.75  # Inverted efficiency
        elif aoa <= inverted_stall:
            return -max_cl * 0.75 * (1.0 - (aoa - inverted_optimal) / (inverted_stall - inverted_optimal))
        else:
            return 0.0  # Deep inverted stall

# DRAG COEFFICIENT: Quadratic increase with angle of attack
def get_cd(aoa: float, min_cd: float, max_cd: float) -> float:
    range_cd = (max_cd - min_cd)
    # Drag increases quadratically with AoA
    cd = range_cd * ((1/8100) * aoa**2) + min_cd
    return cd
    


# ==================== #
# =====| FORCES |===== #
# ==================== #

# Lift force: Calculate from entity properties
def get_lift_force(entity) -> np.ndarray:
    aoa = get_aoa(entity.v, entity.orientation)
    dynamic_pressure = get_dynamic_pressure(entity.v)
    cl = get_cl(aoa, entity.max_lift_coefficient, optimal_aoa=15.0)
    lift_dir = get_lift_dir(entity.v, entity.orientation)
    lift_mag = dynamic_pressure * cl * entity.reference_area
    return lift_mag * lift_dir

# Thrust force: Simple directional force
def get_thrust_force(entity) -> np.ndarray:
    forward_dir = get_forward_dir(entity.orientation)
    return entity.throttle * entity.thrust_force * forward_dir

# Drag force: Calculate from entity properties
def get_drag_force(entity) -> np.ndarray:
    aoa = get_aoa(entity.v, entity.orientation)
    dynamic_pressure = get_dynamic_pressure(entity.v)
    cd = get_cd(aoa, entity.min_drag_coefficient, entity.max_drag_coefficient)
    drag_dir = -entity.v / max(np.linalg.norm(entity.v), 1e-6)  # Opposite to velocity
    drag_mag = dynamic_pressure * cd * entity.reference_area
    return drag_mag * drag_dir

# Gravity force: Simple constant force
def get_gravity_force(mass: float) -> np.ndarray:
    return np.array([0.0, -9.81 * mass, 0.0])

# Control surface force: Aerodynamic control force with limits
def get_control_force(input_value: float, dynamic_pressure: float, max_force: float, effectiveness: float) -> float:
    if abs(input_value) < 0.01:
        return 0.0
    aero_force = dynamic_pressure * 1.0 * abs(input_value) * effectiveness
    return min(aero_force, max_force) * np.sign(input_value)

def get_elevator_force(entity, pitch_input: float) -> np.ndarray:
    """Calculate elevator control surface force vector."""
    aoa = get_aoa(entity.v, entity.orientation)
    dynamic_pressure = get_dynamic_pressure(entity.v)
    control_effectiveness = get_control_surface_weight(aoa)
    
    elevator_force_mag = get_control_force(pitch_input, dynamic_pressure, 50000.0, control_effectiveness)
    return entity.orientation @ np.array([0.0, elevator_force_mag, 0.0])

def get_left_aileron_force(entity, roll_input: float) -> np.ndarray:
    """Calculate left aileron control surface force vector."""
    aoa = get_aoa(entity.v, entity.orientation)
    dynamic_pressure = get_dynamic_pressure(entity.v)
    control_effectiveness = get_control_surface_weight(aoa)
    
    aileron_force_mag = get_control_force(roll_input, dynamic_pressure, 25000.0, control_effectiveness)
    return entity.orientation @ np.array([0.0, aileron_force_mag, 0.0])

def get_right_aileron_force(entity, roll_input: float) -> np.ndarray:
    """Calculate right aileron control surface force vector (differential)."""
    aoa = get_aoa(entity.v, entity.orientation)
    dynamic_pressure = get_dynamic_pressure(entity.v)
    control_effectiveness = get_control_surface_weight(aoa)
    
    aileron_force_mag = get_control_force(roll_input, dynamic_pressure, 25000.0, control_effectiveness)
    return -entity.orientation @ np.array([0.0, aileron_force_mag, 0.0])  # Opposite force for differential control








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


# ========================= #
# =====| ORIENTATION |===== #
# ========================= #

def get_moment(force_world: np.ndarray, R: np.ndarray, cp_diff: np.ndarray) -> np.ndarray:
    # Transform force_world to body space
    force_body = R.T @ force_world
    # Calculate moment in body space (right hand rule applies for rotation direction)
    moment_body = np.cross(cp_diff, force_body)
    return moment_body


def get_force_torque(force_world: np.ndarray, R: np.ndarray, cp_diff: np.ndarray, mass: float, length: float) -> tuple[float, float]:
    moment_body = get_moment(force_world, R, cp_diff)
    pitch_torque = moment_body[2]
    yaw_torque = moment_body[1]
    # Uniform rod: I = (1/12) * m * L²
    I = (1.0 / 12.0) * mass * (length ** 2)
    
    pitch_accel = np.degrees(pitch_torque / I)
    yaw_accel = np.degrees(yaw_torque / I)
    
    return pitch_accel, yaw_accel

def apply_force_at_offset(force_vector: np.ndarray, offset_position: np.ndarray, mass: float, moment_of_inertia: float) -> tuple[np.ndarray, np.ndarray]:
    # Linear acceleration from Newton's second law: F = ma
    linear_acceleration = force_vector / mass
    
    # Calculate moment (torque) using cross product: τ = r × F
    moment_body = np.cross(offset_position, force_vector)
    
    # Angular acceleration: α = τ / I (convert to degrees/s²)
    angular_acceleration = np.degrees(moment_body / moment_of_inertia)
    
    return linear_acceleration, angular_acceleration

def get_control_surface_weight(aoa: float) -> float:
    if abs(aoa) < 20.0:
        return 1.0
    return 1 - min((abs(aoa)-20) / 20, 1)  # Normalize to [0, 1] over a 40 degree range

# ========================== #
# =====| MAIN PHYSICS |===== #
# ========================== #

# Combined calculation functions
def combine_translation_forces(forces: list[np.ndarray]) -> np.ndarray:
    """Combine all force vectors into total force."""
    return sum(forces)

def combine_torques(forces_and_offsets: list[tuple[np.ndarray, np.ndarray]], entity) -> np.ndarray:
    """Combine forces and their offsets into total torque vector."""
    total_torque = np.zeros(3)
    
    for force_vector, offset_pos in forces_and_offsets:
        # Transform force to body coordinates for moment calculation
        force_body = entity.orientation.T @ force_vector
        # Calculate moment: τ = r × F
        moment_body = np.cross(offset_pos, force_body)
        total_torque += moment_body
    
    return total_torque

def calculate_accelerations(total_force: np.ndarray, total_torque: np.ndarray, entity) -> tuple[np.ndarray, np.ndarray]:
    """Convert forces and torques to accelerations."""
    # Linear acceleration: a = F/m
    linear_acceleration = total_force / entity.mass
    
    # Angular acceleration: α = τ/I (convert to degrees/s²)
    angular_acceleration = np.degrees(total_torque / entity.moment_of_inertia)
    
    # Apply time scaling
    return (linear_acceleration / TICK_RATE, angular_acceleration / TICK_RATE)

def calculate_entity_physics(entity, control_inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete physics calculation for an entity using modular force functions.
    
    Args:
        entity: Entity with physical properties (mass, orientation, velocity, etc.)
        control_inputs: Dict with 'pitch', 'roll', 'throttle' values
    
    Returns:
        tuple: (velocity_acceleration, omega_delta) ready to apply to entity
    """
    # Control surface forces with their offsets
    forces_and_offsets = [
        (get_thrust_force(entity), np.array([0.0, 0.0, 0.0])),
        (get_lift_force(entity), entity.cp_diff),
        (get_drag_force(entity), entity.cp_diff),
        (get_gravity_force(entity.mass), np.array([0.0, 0.0, 0.0])),
    ]
    
    # Add control surface forces if entity has pitch/roll capability
    if hasattr(entity, 'pitch_input'):
        elevator_force = get_elevator_force(entity, control_inputs.get('pitch', 0.0))
        elevator_pos = np.array([-entity.length * 0.4, 0.0, 0.0])
        forces_and_offsets.append((elevator_force, elevator_pos))
        
        roll_input = control_inputs.get('roll', 0.0)
        wing_span = entity.length * 0.4
        
        left_aileron_force = get_left_aileron_force(entity, roll_input)
        right_aileron_force = get_right_aileron_force(entity, roll_input)
        
        left_aileron_pos = np.array([0.0, 0.0, -wing_span])
        right_aileron_pos = np.array([0.0, 0.0, wing_span])
        
        forces_and_offsets.append((left_aileron_force, left_aileron_pos))
        forces_and_offsets.append((right_aileron_force, right_aileron_pos))
    
    # Calculate total force and torque
    total_force = combine_translation_forces([f for f, _ in forces_and_offsets])
    total_torque = combine_torques(forces_and_offsets, entity)
    
    # Convert to accelerations
    return calculate_accelerations(total_force, total_torque, entity)


# Rotation matrix system - no more individual angle updates needed

# ====================== #
# =====| POSITION |===== #
# ====================== #
def get_next_pos(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    return p + (v / TICK_RATE)