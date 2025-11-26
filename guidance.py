import numpy as np
import math
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

def missile_direct_attack_think(self, entities: list):
    ''' This think step will adjust the missile's angular velocity to guide it towards its target'''
    target = None
    for e in entities:
        if e is not self and e.shape == "jet":
            target = e
            break
    if target is None:
        return
    d = target.p - self.p
    dist = np.linalg.norm(d)
    if dist == 0.0:
        return
    dir_world = d / dist
    R = physics.get_rotation_matrix(self.roll, self.pitch, self.yaw)
    dir_body = R.T @ dir_world  # x=fwd, y=up, z=right
    pitch_err = math.degrees(math.atan2(dir_body[1], dir_body[0]))
    yaw_err   = math.degrees(math.atan2(dir_body[2], dir_body[0]))
    max_rate = self.max_roll_velocity
    desired_pitch_v = max(-max_rate, min(max_rate, pitch_err * 5.0))
    desired_yaw_v   = max(-max_rate, min(max_rate, yaw_err   * 5.0))
    self.pitch_v += (desired_pitch_v - self.pitch_v) * 0.5
    self.yaw_v   += (desired_yaw_v   - self.yaw_v)   * 0.5