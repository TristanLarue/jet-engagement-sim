import numpy as np
from config import AIR_DENSITY, TICK_RATE

def snap_orientation_to_velocity(ent):
    """Align entity orientation to match its velocity direction"""
    v_mag = np.linalg.norm(ent.v)
    if v_mag > 0:
        v_norm = ent.v / v_mag
        # Yaw: horizontal angle (XZ plane)
        ent.yaw = np.degrees(np.arctan2(v_norm[0], v_norm[2]))
        # Pitch: vertical angle
        horizontal_mag = np.sqrt(v_norm[0]**2 + v_norm[2]**2)
        ent.pitch = np.degrees(np.arctan2(v_norm[1], horizontal_mag))
        ent.roll = 0.0
    else:
        ent.pitch = 0.0
        ent.yaw = 0.0
        ent.roll = 0.0

def get_rotation_matrix(ent):
    roll_rad = np.radians(ent.roll)
    pitch_rad = np.radians(ent.pitch)
    yaw_rad = np.radians(ent.yaw)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])
    return Rz @ Ry @ Rx

def get_thrust_dir(ent): #Returns the "forward" unit vector of the plane
    return get_rotation_matrix(ent) @ np.array([1.0, 0.0, 0.0])

def get_aoa(ent):
    v_mag = np.linalg.norm(ent.v)
    if v_mag == 0.0:
        return 0.0
    # Velocity in body coordinates
    R = get_rotation_matrix(ent) # !!! multiply a body vector with R to get a world vector !!!
    v_body = R.T @ ent.v # Litterally the opposite of R, Makes a world vector become a body vector when multiplied
    forward_c = v_body[0] # forward component of wind
    up_c = v_body[1] # "up" component
    return np.arctan2(up_c, forward_c)

def get_current_drag_coefficient(ent): #Returns a simplified real time drag coefficient based on simplified AoA (exponential)
    aoa = get_aoa(ent)
    aoa_deg = np.degrees(aoa)
    aoa_offset = abs(aoa_deg + 2.0) #aoa_deg - -2.0deg to align with wing angle offset
    aoa_ratio = np.clip(aoa_offset / 90.0, 0.0, 1.0)
    return ent.min_drag_coefficient + (ent.max_drag_coefficient - ent.min_drag_coefficient) * (aoa_ratio ** 2)

def get_current_lift_coefficient(ent): #Returns a simplified real time lift coefficient based on simplified AoA (linear)
    # linear from -2°(0_lift) to 15°(max_lift) to 20°(0_lift)
    aoa_deg = np.degrees(get_aoa(ent))
    if aoa_deg <= -2.0:
        return 0.0
    if aoa_deg <= 15.0:
        return ent.max_lift_coefficient * ((aoa_deg + 2.0) / 17.0)
    if aoa_deg < 20.0: #stall region// lift breaks past a critical aoa angle (20 in our simulation)
        lift_decay = (aoa_deg - 15.0) / 5.0  # 0 at 15°, 1 at 20°
        return ent.max_lift_coefficient * (1.0 - lift_decay)
    return 0.0 #TOP GUN THROTTLE SPLIT SCENE BABYYYYYYYYYYYYY, a big part of this simulation and why I made it was to see if the AI could actually find crazy maneuvers like this line



def apply_angular_velocity(ent):
    velocity_magnitude = np.linalg.norm(ent.v)
    # Pitch control with g-limit
    if velocity_magnitude > 0:
        max_pitch_rate = (ent.g_limit * 9.81 / velocity_magnitude) * (180.0 / np.pi)
        pitch_rate = ent.pitch_input * min(max_pitch_rate, ent.max_angular_rate)
    else:
        pitch_rate = ent.pitch_input * ent.max_angular_rate
    ent.pitch += pitch_rate / TICK_RATE
    
    # Roll control
    max_roll_rate = 360.0
    roll_rate = ent.roll_input * max_roll_rate
    ent.roll += roll_rate / TICK_RATE
    
    # Normalize angles to [-180, 180]
    ent.pitch = ((ent.pitch + 180.0) % 360.0) - 180.0
    ent.roll = ((ent.roll + 180.0) % 360.0) - 180.0

def apply_restoring_torque(ent):
    v_mag = np.linalg.norm(ent.v)
    if v_mag == 0: 
        return
    d = ent.v / v_mag
    yaw_target = np.degrees(np.arctan2(d[2], d[0]))
    h = np.hypot(d[0], d[2])
    pitch_target = 0.0 if h == 0 else np.degrees(np.arctan2(d[1], h))

    yaw_error = (yaw_target - ent.yaw + 180.0) % 360.0 - 180.0
    pitch_error = (pitch_target - ent.pitch + 180.0) % 360.0 - 180.0

    max_rate = min(ent.max_angular_rate, 0.0003 * v_mag * v_mag)  # deg/s
    step = max_rate / TICK_RATE
    if step <= 0.0:
        return

    if abs(yaw_error) <= step:
        ent.yaw = yaw_target
    else:
        ent.yaw += step if yaw_error > 0.0 else -step

    if abs(pitch_error) <= step:
        ent.pitch = pitch_target
    else:
        ent.pitch += step if pitch_error > 0.0 else -step

    ent.pitch = ((ent.pitch + 180.0) % 360.0) - 180.0
    ent.yaw   = ((ent.yaw   + 180.0) % 360.0) - 180.0
    ent.roll  = ((ent.roll  + 180.0) % 360.0) - 180.0

def apply_thrust(ent):
    thrust_dir = get_thrust_dir(ent)
    thrust_acc = ent.throttle * ent.thrust * thrust_dir
    ent.v += thrust_acc / TICK_RATE

def apply_gravity(ent):
    ent.v += np.array([0.0, -9.81, 0.0]) / TICK_RATE

def apply_lift(ent):
    velocity_magnitude = np.linalg.norm(ent.v)
    if velocity_magnitude == 0:
        return
    
    # Body "up" in world space (same as before)
    lift_dir = get_rotation_matrix(ent) @ np.array([0.0, 1.0, 0.0])
    
    # Flow direction (air hitting the aircraft) = opposite of velocity
    velocity_unit = ent.v / velocity_magnitude
    flow_dir = -velocity_unit

    # Project body-up onto plane perpendicular to the flow
    lift_dir = lift_dir - np.dot(lift_dir, flow_dir) * flow_dir
    lift_dir_norm = np.linalg.norm(lift_dir)
    if lift_dir_norm == 0.0:
        return
    lift_dir /= lift_dir_norm

    # Lift coefficient from your AoA model
    lift_coefficient = get_current_lift_coefficient(ent)
    if lift_coefficient == 0.0:
        return
    
    # Lift magnitude
    lift_force = 0.5 * AIR_DENSITY * velocity_magnitude**2 * lift_coefficient * ent.reference_area
    lift_acc_magnitude = lift_force / ent.mass

    # Final lift acceleration
    lift_acc = lift_acc_magnitude * lift_dir
    
    # No G-limit here: just apply it directly
    ent.v += lift_acc / TICK_RATE

def apply_drag(ent):
    velocity_magnitude = np.linalg.norm(ent.v)
    if velocity_magnitude == 0:
        return
    
    # Calculate drag coefficient
    drag_coefficient = get_current_drag_coefficient(ent)
    
    # Calculate drag acceleration
    drag_force = 0.5 * AIR_DENSITY * velocity_magnitude**2 * drag_coefficient * ent.reference_area
    drag_acc_magnitude = drag_force / ent.mass
    velocity_unit = ent.v / velocity_magnitude
    drag_acc = drag_acc_magnitude * -velocity_unit
    
    ent.v += drag_acc / TICK_RATE

def apply_velocity(ent):
    ent.p += ent.v / TICK_RATE