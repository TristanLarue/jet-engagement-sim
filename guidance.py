import numpy as np
import physics

def jet_ai_think(entities: list, jet_entity):
    target_pitch = 0.0 #Target level flight
    target_speed = 250.0 #Target cruising speed (m/s)
    target_roll = 0.0 #Keep wings level
    
    current_speed = np.linalg.norm(jet_entity.v)
    
    #Pitch stabilization with proportional control
    pitch_error = target_pitch - jet_entity.pitch
    # Normalize error to [-180, 180]
    while pitch_error > 180.0:
        pitch_error -= 360.0
    while pitch_error < -180.0:
        pitch_error += 360.0
    jet_entity.pitch_input = np.clip(pitch_error * 0.05, -1.0, 1.0)
    
    #Roll stabilization
    roll_error = target_roll - jet_entity.roll
    # Normalize error to [-180, 180]
    while roll_error > 180.0:
        roll_error -= 360.0
    while roll_error < -180.0:
        roll_error += 360.0
    jet_entity.roll_input = np.clip(roll_error * 0.02, -1.0, 1.0)
    
    #Speed control via throttle
    speed_error = target_speed - current_speed
    base_throttle = 0.7
    jet_entity.throttle = np.clip(base_throttle + speed_error * 0.002, 0.1, 1.0)
    

def missile_direct_attack_think(entities: list, missile_entity):
    target_entity = missile_entity.target
    if target_entity:
        direction_to_target = target_entity.p - missile_entity.p
        distance = np.linalg.norm(direction_to_target)
        
        if distance > 0:
            target_dir = direction_to_target / distance
            
            # Calculate desired pitch (vertical angle)
            desired_pitch = np.degrees(np.arctan2(target_dir[1], np.sqrt(target_dir[0]**2 + target_dir[2]**2)))
            pitch_error = desired_pitch - missile_entity.pitch
            
            # Calculate desired heading in horizontal plane
            desired_yaw = np.degrees(np.arctan2(target_dir[0], target_dir[2]))
            yaw_error = desired_yaw - missile_entity.yaw
            
            # Normalize yaw error to [-180, 180]
            while yaw_error > 180.0:
                yaw_error -= 360.0
            while yaw_error < -180.0:
                yaw_error += 360.0
            
            # Use roll to turn towards target (bank-to-turn)
            missile_entity.pitch_input = np.clip(pitch_error / 30.0, -1.0, 1.0)
            missile_entity.roll_input = np.clip(yaw_error / 90.0, -1.0, 1.0)
            missile_entity.yaw_input = 0.0
