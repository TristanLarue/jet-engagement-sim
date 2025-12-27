import numpy as np
'''
ALL POSITIONS,VELOCITIES,ACCELERATIONS & FORCES ARE IN WORLD-SPACE

FORCES ACTING ON PARTICLES:
- Thrust
    In direction of the particle's forward axis.
    Magnitude linear to throttle
    Applied at the back of the particle (technically useless, but fun)
    *TORQUE ABSENT*

- Gravity
    Constant downward acceleration of 9.81 m/s²
    Must be transformed into a force
    Applied at the center of mass
    *TORQUE ABSENT*
    
- Drag
    In the opposite direction of velocity.
    Magnitude based on "0.5 * ad * v_mag**2 * reference_area * cd * drag_direction"
    Where ad is a dynamic air density based on altitude
    Where v_mag is the magnitude of the velocity vector
    Where cd is a simplified drag coefficient based on angle of attack
    IMPORTANT: reference_area & cd are based on the fuselage WITHOUT control surfaces (these are calculated separately)
    *TORQUE PRESENT*

- Lift
    Perpendicular to both velocity and right axis. ("Up" direction of velocity based on jet's roll)
    Magnitude based on "0.5 * ad * v_mag**2 * reference_area * cl * lift_direction"
    Where ad is a dynamic air density based on altitude
    Where v_mag is the magnitude of the velocity vector
    Where cl is a simplified lift coefficient based on angle of attack
    *TORQUE PRESENT*

- Sideforce
    Perpendicular to both velocity and up axis.
    *lift-based force* that creates a yaw restoring motion by the vertical stabilizer
    *TORQUE PRESENT*

- Control_Elevators
    *lift-based force* that creates a pitch motion
    Applied at the tail of the aircraft
    *TORQUE PRESENT*

- Control_Ailerons
    *lift-based force* that creates a roll motion
    Applied at the wings of the aircraft
    *TORQUE PRESENT*

- Control_Rudder
    *lift-based force* that creates a yaw motion
    Applied at the vertical stabilizer of the aircraft
    *TORQUE PRESENT*
'''

def get_air_density(altitude: float) -> float:
    '''Returns the air density based on altitude'''
    '''Simplified with a barometric formula'''
    sea_level_density = 1.225
    temperature_lapse_rate = 0.0065
    sea_level_temp = 288.15 #K
    temperature = sea_level_temp - (temperature_lapse_rate * altitude)
    if temperature < 0:
        temperature = 0
    density = sea_level_density * (temperature / sea_level_temp) ** 5.255
    return np.clip(density, 0.0, sea_level_density)

def get_angle_of_attack(velocity: np.ndarray, R: np.ndarray) -> float:
    '''Returns the aoa angle in degrees'''
    '''Between -180 and 180 degrees'''
    if np.linalg.norm(velocity) < 1e-12:
        return 0.0 #Value doesn't matter
    v_body = R.T @ velocity #Apply the velocity to the body's POV
    return -np.degrees(np.arctan2(v_body[1], v_body[0])) #Angle between forward and "up" axis

def get_sideslip(velocity: np.ndarray, R: np.ndarray) -> float:
    '''Returns the sideslip angle in degrees'''
    '''Between -180 and 180 degrees'''
    if np.linalg.norm(velocity) < 1e-12:
        return 0.0 #Value doesn't matter
    v_body = R.T @ velocity #Apply the velocity to the body's POV
    return np.degrees(np.arctan2(v_body[2], v_body[0])) #Angle between forward and "right" axis

def get_lift_coefficient(aoa: float, max_lift_coefficient: float, optimal_lift_aoa: float,zero_lift_aoa: float) -> float:
    a = ((aoa + 180.0) % 360.0) - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    stall_pos = float(optimal_lift_aoa)
    stall_neg = -stall_pos
    cl_max = float(max_lift_coefficient)
    cl_min = -0.75 * cl_max

    denom = (stall_pos - zero_lift_aoa)
    slope = (cl_max / denom) if abs(denom) > 1e-12 else 0.0

    if stall_neg <= a <= stall_pos:
        return max(cl_min, min(cl_max, slope * (a - zero_lift_aoa)))

    if a > stall_pos:
        return (cl_max - 0.016 * (a - stall_pos) ** 2) if a <= (stall_pos + 5.0) else max(0.0, 0.8 * cl_max * (1.0 - (a - (stall_pos + 5.0)) / 70.0))

    return (cl_min + 0.01 * (a - stall_neg) ** 2) if a >= (stall_neg - 5.0) else min(0.0, -0.65 * abs(cl_min) * (1.0 - (abs(a) - abs(stall_neg - 5.0)) / 70.0))

def get_drag_coefficient(aoa: float, min_drag_coefficient: float, max_drag_coefficient: float) -> float:
    range_cd = (max_drag_coefficient - min_drag_coefficient)
    # Drag increases quadratically with AoA
    cd = range_cd * ((1/8100) * aoa**2) + min_drag_coefficient
    return cd

def get_lift_force(velocity: np.ndarray, reference_area: float, max_lift_coefficient: float, R: np.ndarray, air_density: float) -> np.ndarray:
    velocity_magnitude = np.linalg.norm(velocity)
    lift_magnitude = 0.5 * air_density * velocity_magnitude**2 * reference_area * max_lift_coefficient
    lift_force = lift_magnitude * get_lift_dir(velocity, R)  # Lift acts in the body "up" direction
    return lift_force

def get_gravity_force(mass: float):
    gravity_force = np.array([0.0, -9.81 * mass, 0.0])
    return gravity_force

def get_thrust_force(R:np.ndarray, throttle: float, thrust_force: float, length: float):
    return thrust_force * throttle * get_forward_dir(R)

def get_drag_force(velocity: np.ndarray, air_density: float, reference_area: float, drag_coefficient: float) -> np.ndarray:
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude < 1e-12:
        return np.zeros(3)
    drag_direction = -velocity / velocity_magnitude
    drag_magnitude = 0.5 * air_density * velocity_magnitude**2 * reference_area * drag_coefficient
    return drag_magnitude * drag_direction
 
def get_fuselage_lift_force(velocity: np.ndarray, reference_area: float, R: np.ndarray, air_density: float, lift_coefficient: float) -> np.ndarray:
    return get_lift_force(velocity, reference_area, lift_coefficient, R, air_density)

def get_sideforce_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, vertical_stabilizer_area_multiplier: float = 0.06, max_sideslip_degrees: float = 25.0) -> np.ndarray:
    velocity_magnitude = float(np.linalg.norm(velocity))
    if velocity_magnitude < 1e-12: return np.zeros(3)

    airflow_direction = -velocity / velocity_magnitude
    right_direction = get_right_dir(R)

    sideforce_direction = right_direction - np.dot(right_direction, airflow_direction) * airflow_direction
    sideforce_direction_norm = float(np.linalg.norm(sideforce_direction))
    sideforce_direction = (sideforce_direction / sideforce_direction_norm) if sideforce_direction_norm > 1e-12 else right_direction

    sideslip = float(get_sideslip(velocity, R))
    sideslip = float(np.clip(sideslip, -max_sideslip_degrees, max_sideslip_degrees))

    sideforce_coefficient = -(sideslip / max_sideslip_degrees) * float(max_lift_coefficient)
    sideforce_magnitude = 0.5 * float(air_density) * velocity_magnitude * velocity_magnitude * (float(reference_area) * float(vertical_stabilizer_area_multiplier)) * sideforce_coefficient
    return sideforce_magnitude * sideforce_direction


def get_dynamic_pressure(velocity: np.ndarray, air_density: float):
    velocity_magnitude = float(np.linalg.norm(velocity))
    if velocity_magnitude < 1e-12: return 0.0, 0.0
    return float(0.5 * float(air_density) * velocity_magnitude * velocity_magnitude), velocity_magnitude


def get_control_effectiveness(velocity: np.ndarray, R: np.ndarray, optimal_lift_aoa: float, stall_fade_degrees: float = 25.0, minimum_effectiveness: float = 0.2) -> float:
    angle_of_attack = abs(float(get_angle_of_attack(velocity, R)))
    if angle_of_attack <= float(optimal_lift_aoa): return 1.0
    if float(stall_fade_degrees) < 1e-12: return float(minimum_effectiveness)
    return float(max(float(minimum_effectiveness), 1.0 - (angle_of_attack - float(optimal_lift_aoa)) / float(stall_fade_degrees)))


def get_control_force_magnitude(dynamic_pressure: float, reference_area: float, surface_area_multiplier: float, max_lift_coefficient: float, lift_coefficient_multiplier: float, control_input: float, control_effectiveness: float) -> float:
    return float(dynamic_pressure) * float(reference_area) * float(surface_area_multiplier) * float(max_lift_coefficient) * float(lift_coefficient_multiplier) * float(control_input) * float(control_effectiveness)

def get_elevator_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, pitch_input: float, optimal_lift_aoa: float, elevator_surface_area_multiplier: float = 0.18, elevator_lift_coefficient_multiplier: float = 1.0) -> np.ndarray:
    if abs(float(pitch_input)) < 1e-12: return np.zeros(3)
    dynamic_pressure, velocity_magnitude = get_dynamic_pressure(velocity, air_density)
    if velocity_magnitude < 1e-12: return np.zeros(3)
    return get_control_force_magnitude(dynamic_pressure, reference_area, elevator_surface_area_multiplier, max_lift_coefficient, elevator_lift_coefficient_multiplier, pitch_input, get_control_effectiveness(velocity, R, optimal_lift_aoa)) * get_lift_dir(velocity, R)

def get_aileron_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, roll_input: float, optimal_lift_aoa: float, aileron_surface_area_multiplier: float = 0.06, aileron_lift_coefficient_multiplier: float = 0.8):
    if abs(float(roll_input)) < 1e-12: return np.zeros(3), np.zeros(3)
    dynamic_pressure, velocity_magnitude = get_dynamic_pressure(velocity, air_density)
    if velocity_magnitude < 1e-12: return np.zeros(3), np.zeros(3)
    aileron_force_vector = get_control_force_magnitude(dynamic_pressure, reference_area, aileron_surface_area_multiplier, max_lift_coefficient, aileron_lift_coefficient_multiplier, roll_input, get_control_effectiveness(velocity, R, optimal_lift_aoa)) * get_lift_dir(velocity, R)
    return aileron_force_vector, -aileron_force_vector

def get_rudder_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, yaw_input: float, optimal_lift_aoa: float, rudder_surface_area_multiplier: float = 0.10, rudder_lift_coefficient_multiplier: float = 0.7) -> np.ndarray:
    if abs(float(yaw_input)) < 1e-12: return np.zeros(3)
    dynamic_pressure, velocity_magnitude = get_dynamic_pressure(velocity, air_density)
    if velocity_magnitude < 1e-12: return np.zeros(3)
    airflow_direction = -velocity / velocity_magnitude
    right_dir = get_right_dir(R)
    rudder_force_direction = right_dir - np.dot(right_dir, airflow_direction) * airflow_direction
    rudder_force_direction_magnitude = float(np.linalg.norm(rudder_force_direction))
    if rudder_force_direction_magnitude < 1e-12:
        rudder_force_direction = np.cross(airflow_direction, get_up_dir(R))
        rudder_force_direction_magnitude = float(np.linalg.norm(rudder_force_direction))
        if rudder_force_direction_magnitude < 1e-12: return np.zeros(3)
    return get_control_force_magnitude(dynamic_pressure, reference_area, rudder_surface_area_multiplier, max_lift_coefficient, rudder_lift_coefficient_multiplier, yaw_input, get_control_effectiveness(velocity, R, optimal_lift_aoa)) * (rudder_force_direction / rudder_force_direction_magnitude)



def get_forward_dir(R: np.ndarray) -> np.ndarray: #Returns the "forward" unit vector of the plane
    return R @ np.array([1.0, 0.0, 0.0])

def get_up_dir(R: np.ndarray) -> np.ndarray: #Returns the "up" unit vector of the plane
    return R @ np.array([0.0, 1.0, 0.0])

def get_right_dir(R: np.ndarray) -> np.ndarray: #Returns the "right" unit vector of the plane (kinda useful on edge cases)
    return R @ np.array([0.0, 0.0, 1.0])

def get_lift_dir(velocity: np.ndarray, R: np.ndarray) -> np.ndarray:
    up_direction = get_up_dir(R)
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude < 1e-12:
        return up_direction
    airflow_direction = -velocity / velocity_magnitude  # unit airflow direction
    lift = up_direction - np.dot(up_direction, airflow_direction) * airflow_direction  # projection of up onto plane ⟂ flow
    lift_norm = np.linalg.norm(lift)
    if lift_norm < 1e-12:
        # up is (almost) parallel to flow -> pick any stable perpendicular direction
        right = get_right_dir(R)
        lift = np.cross(airflow_direction, right)
        lift_norm = np.linalg.norm(lift)
        if lift_norm < 1e-12:
            return up_direction  # absolute fallback
    return lift / lift_norm

def get_omega(force: np.ndarray, R: np.ndarray, application_point: np.ndarray, moment_of_inertia: np.ndarray) -> np.ndarray:
    F_body = R.T @ force
    torque_body = np.cross(application_point, F_body)
    alpha_body = torque_body / moment_of_inertia
    return np.degrees(alpha_body)

def align_roll_to_velocity(R: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    velocity_magnitude = float(np.linalg.norm(velocity))
    if velocity_magnitude < 1e-12: return R

    forward_direction = get_forward_dir(R)
    forward_magnitude = float(np.linalg.norm(forward_direction))
    if forward_magnitude < 1e-12: return R
    forward_direction = forward_direction / forward_magnitude

    velocity_direction = velocity / velocity_magnitude
    lateral_direction = velocity_direction - np.dot(velocity_direction, forward_direction) * forward_direction
    lateral_magnitude = float(np.linalg.norm(lateral_direction))
    if lateral_magnitude < 1e-12: return R

    up_direction = (-lateral_direction) / lateral_magnitude
    right_direction = np.cross(forward_direction, up_direction)
    right_magnitude = float(np.linalg.norm(right_direction))
    if right_magnitude < 1e-12: return R
    right_direction = right_direction / right_magnitude

    return np.column_stack((forward_direction, np.cross(right_direction, forward_direction), right_direction))


def integrate_velocity(velocity: np.ndarray, acceleration: np.ndarray, dt: float) -> np.ndarray:
    return velocity + acceleration * dt

def integrate_position(position: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    return position + velocity * dt


def integrate_orientation(R: np.ndarray, omega_deg_s: np.ndarray, dt: float) -> np.ndarray:
    w = np.radians(omega_deg_s)# rad/s in BODY axes
    th = float(np.linalg.norm(w) * dt)# rotation angle in rad
    if th < 1e-12:
        return R
    axis = w / np.linalg.norm(w)
    K = np.array([[0.0, -axis[2],  axis[1]],
                  [axis[2],  0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype=float)
    dR = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
    return R @ dR  # right-multiply because omega is in BODY frame
