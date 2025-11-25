import numpy as np
import physics

def jet_float_think(entities: list, jet_entity):
    """
    Simple stabilization logic for the jet:
    - Keep pitch above 5 degrees to stay afloat
    - Correct roll if it exceeds 5 degrees on either side
    """
    # Pitch control: if below 14 degrees, pitch up
    if jet_entity.pitch < 14.0:
        jet_entity.pitch_input = 0.5  # Pitch up
    elif jet_entity.pitch > 15.0:
        jet_entity.pitch_input = -0.3  # Pitch down slightly to avoid over-correction
    else:
        jet_entity.pitch_input = 0.0  # Hold steady
    
    # Roll correction: keep wings level
    if jet_entity.roll > 5.0:
        # Rolling right, correct by rolling left
        jet_entity.roll_input = -0.5
    elif jet_entity.roll < -5.0:
        # Rolling left, correct by rolling right
        jet_entity.roll_input = 0.5
    else:
        jet_entity.roll_input = 0.0  # Wings level

def missile_direct_attack_think(entities: list, missile_entity):
    """
    Predictive guidance for missile:
    - Calculate where the target will be based on current position + velocity * (distance/missile_speed)
    - Point the missile toward the predicted intercept point
    """
    if not hasattr(missile_entity, 'target') or missile_entity.target is None:
        return
    
    target = missile_entity.target
    
    # Calculate relative position
    relative_pos = target.p - missile_entity.p
    distance = np.linalg.norm(relative_pos)
    
    if distance == 0.0:
        return
    
    # Calculate missile speed
    missile_speed = np.linalg.norm(missile_entity.v)
    
    if missile_speed == 0.0:
        missile_speed = 1.0  # Avoid division by zero
    
    # Predict target position
    time_to_intercept = distance / missile_speed
    predicted_target_pos = target.p + target.v * time_to_intercept
    
    # Calculate vector to predicted intercept point
    intercept_vector = predicted_target_pos - missile_entity.p
    intercept_distance = np.linalg.norm(intercept_vector)
    
    if intercept_distance == 0.0:
        return
    
    intercept_direction = intercept_vector / intercept_distance
    
    # Get missile's current forward direction
    R = physics.get_rotation_matrix(missile_entity.roll, missile_entity.pitch, missile_entity.yaw)
    forward_dir = physics.get_forward_dir(R)
    
    # Calculate error between current heading and desired heading
    # Project onto pitch and yaw corrections
    up_dir = physics.get_up_dir(R)
    right_dir = physics.get_right_dir(R)
    
    # Pitch error: dot product with up direction
    pitch_error = np.dot(intercept_direction, up_dir)
    # Yaw error: dot product with right direction  
    yaw_error = np.dot(intercept_direction, right_dir)
    
    # Set control inputs proportional to error
    missile_entity.pitch_input = np.clip(pitch_error * 2.0, -1.0, 1.0)
    # Note: yaw_input is commented out in entity class, but we can still try to use roll for turns
    # For simplicity, we'll use roll to help with horizontal corrections
    missile_entity.roll_input = np.clip(yaw_error * 2.0, -1.0, 1.0)
