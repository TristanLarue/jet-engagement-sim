import numpy as np
from config import AIR_DENSITY, TICK_RATE

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

def get_aoa(ent):#The angle of difference between the fuselage and the velocity (from -90 to +90 degrees)
    velocity_magnitude = np.linalg.norm(ent.v)
    if velocity_magnitude == 0:
        return 0.0
    velocity_unit = ent.v / velocity_magnitude
    cos_angle = np.clip(np.dot(get_thrust_dir(ent), velocity_unit), -1.0, 1.0)
    return np.arccos(cos_angle) 

def get_current_drag_coefficient(ent): #Returns a simplified real time drag coefficient based on simplified AoA (exponential)
    aoa = get_aoa(ent)
    aoa_deg = np.degrees(aoa)
    aoa_offset = abs(aoa_deg + 2.0) #aoa_deg - -2.0deg to align with wing angle offset
    aoa_ratio = np.clip(aoa_offset / 90.0, 0.0, 1.0)
    return ent.min_drag_coefficient + (ent.max_drag_coefficient - ent.min_drag_coefficient) * (aoa_ratio ** 2)

def get_current_lift_coefficient(ent): #Returns a simplified real time lift coefficient based on simplified AoA (linear)
    aoa = get_aoa(ent)
    aoa_deg = np.degrees(aoa)
    if aoa_deg <= -2.0:
        return 0.0
    elif aoa_deg <= 15.0:
        return ent.max_lift_coefficient * ((aoa_deg + 2.0) / 17.0)
    else:
        return 0.0

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
    velocity_magnitude = np.linalg.norm(ent.v)
    if velocity_magnitude == 0:
        return
    
    stability_constant = 0.00001
    
    # Get forward direction from orientation
    forward_dir = get_thrust_dir(ent)
    velocity_dir = ent.v / velocity_magnitude
    
    # YAW: Compare horizontal components (ignore Y axis)
    forward_horizontal = np.array([forward_dir[0], 0.0, forward_dir[2]])
    velocity_horizontal = np.array([velocity_dir[0], 0.0, velocity_dir[2]])
    
    forward_h_mag = np.linalg.norm(forward_horizontal)
    velocity_h_mag = np.linalg.norm(velocity_horizontal)
    
    if forward_h_mag > 0 and velocity_h_mag > 0:
        forward_horizontal = forward_horizontal / forward_h_mag
        velocity_horizontal = velocity_horizontal / velocity_h_mag
        
        # Cross product to determine sign (left/right)
        cross_yaw = forward_horizontal[0] * velocity_horizontal[2] - forward_horizontal[2] * velocity_horizontal[0]
        dot_yaw = np.clip(np.dot(forward_horizontal, velocity_horizontal), -1.0, 1.0)
        yaw_error = np.degrees(np.arccos(dot_yaw))
        if cross_yaw < 0:
            yaw_error = -yaw_error
        
        restoring_yaw_rate = stability_constant * velocity_magnitude**2 * yaw_error
        ent.yaw += restoring_yaw_rate / TICK_RATE
    
    # PITCH: Compare vertical angle (side view, XY plane for forward dir)
    forward_vertical = np.array([np.linalg.norm([forward_dir[0], forward_dir[2]]), forward_dir[1]])
    velocity_vertical = np.array([np.linalg.norm([velocity_dir[0], velocity_dir[2]]), velocity_dir[1]])
    
    forward_v_mag = np.linalg.norm(forward_vertical)
    velocity_v_mag = np.linalg.norm(velocity_vertical)
    
    if forward_v_mag > 0 and velocity_v_mag > 0:
        forward_vertical = forward_vertical / forward_v_mag
        velocity_vertical = velocity_vertical / velocity_v_mag
        
        # Cross product to determine sign (up/down)
        cross_pitch = forward_vertical[0] * velocity_vertical[1] - forward_vertical[1] * velocity_vertical[0]
        dot_pitch = np.clip(np.dot(forward_vertical, velocity_vertical), -1.0, 1.0)
        pitch_error = np.degrees(np.arccos(dot_pitch))
        if cross_pitch < 0:
            pitch_error = -pitch_error
        
        restoring_pitch_rate = stability_constant * velocity_magnitude**2 * pitch_error
        ent.pitch += restoring_pitch_rate / TICK_RATE
    
    # Normalize angles to [-180, 180]
    ent.pitch = ((ent.pitch + 180.0) % 360.0) - 180.0
    ent.yaw = ((ent.yaw + 180.0) % 360.0) - 180.0
    ent.roll = ((ent.roll + 180.0) % 360.0) - 180.0

def apply_thrust(ent):
    thrust_dir = get_thrust_dir(ent)
    thrust_acc = ent.throttle * ent.thrust * thrust_dir
    ent.v += thrust_acc / TICK_RATE

def apply_gravity(ent):
    ent.v += np.array([0.0, -9.81, 0.0]) / TICK_RATE

def apply_lift_with_g_limit(ent):
    velocity_magnitude = np.linalg.norm(ent.v)
    if velocity_magnitude == 0:
        return
    
    # Calculate lift direction (up vector)
    lift_dir = get_rotation_matrix(ent) @ np.array([0.0, 1.0, 0.0])
    
    # Calculate lift coefficient
    lift_coefficient = get_current_lift_coefficient(ent)
    
    # Calculate lift acceleration
    lift_force = 0.5 * AIR_DENSITY * velocity_magnitude**2 * lift_coefficient * ent.reference_area
    lift_acc_magnitude = lift_force / ent.mass
    lift_acc = lift_acc_magnitude * lift_dir
    
    # Apply g-limit
    lift_magnitude = np.linalg.norm(lift_acc)
    max_lift_acc = ent.g_limit * 9.81
    if lift_magnitude > max_lift_acc:
        lift_acc = (lift_acc / lift_magnitude) * max_lift_acc
    
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
