import numpy as np
import math
import physics
from deeplearning import jet_ai_step



def jet_ai_think(jet_entity, entities: list):
    """Single entry point: call AI, then apply outputs to the jet."""
    pitch_input, roll_input, throttle = jet_ai_step(entities, jet_entity)
    jet_entity.pitch_input = pitch_input
    jet_entity.roll_input = roll_input
    jet_entity.throttle = throttle

def jet_float_think(entities: list, jet_entity):
    """
    Simple stabilization logic for the jet:
    - Keep pitch above 5 degrees to stay afloat
    - Correct roll if it exceeds 5 degrees on either side
    """
    
    # Extract current Euler angles from orientation matrix
    roll, pitch, yaw = physics.extract_euler_angles(jet_entity.orientation)
    
    # Pitch control: if below 14 degrees, pitch up
    if pitch < 14.0:
        jet_entity.pitch_input = 0.5  # Pitch up
    elif pitch > 15.0:
        jet_entity.pitch_input = -0.3  # Pitch down slightly to avoid over-correction
    else:
        jet_entity.pitch_input = 0.0  # Hold steady
    
    # Roll correction: keep wings level
    if roll > 5.0:
        # Rolling right, correct by rolling left
        jet_entity.roll_input = -0.5
    elif roll < -5.0:
        # Rolling left, correct by rolling right
        jet_entity.roll_input = 0.5
    else:
        jet_entity.roll_input = 0.0  # Wings level

def jet_manual_control(jet_entity, entities=None):
    """
    Modular manual control guidance for jet.
    Reads pitch/roll from viz.py global state and applies to jet_entity.
    """
    try:
        import viz
        jet_entity.pitch_input = viz.GLOBAL_MANUAL_PITCH
        jet_entity.roll_input = viz.GLOBAL_MANUAL_ROLL
    except Exception as e:
        # Fallback: zero input if viz.py not loaded
        jet_entity.pitch_input = 0.0
        jet_entity.roll_input = 0.0

def missile_direct_attack_think(self, entities: list):
    target = None
    for e in entities:
        if e is not self and e.shape == "jet":
            target = e
            break
    if target is None:
        return

    p_m = self.p
    v_m = self.v
    p_j = target.p
    v_j = target.v

    speed_m = np.linalg.norm(v_m)
    if speed_m == 0.0:
        return

    d = p_j - p_m
    dist = np.linalg.norm(d)
    if dist == 0.0:
        return

    # rough time to go, then refine it a few times
    t = dist / speed_m
    for _ in range(3):
        p_j_pred = p_j + v_j * t
        d_pred = p_j_pred - p_m
        dist_pred = np.linalg.norm(d_pred)
        if dist_pred == 0.0:
            break
        t = dist_pred / speed_m

    aim_point = p_j + v_j * t
    aim_vec = aim_point - p_m
    aim_dist = np.linalg.norm(aim_vec)
    if aim_dist == 0.0:
        return
    dir_world = aim_vec / aim_dist

    R = self.orientation
    dir_body = R.T @ dir_world  # x=fwd, y=up, z=right

    # body-space errors
    pitch_err = math.degrees(math.atan2(dir_body[1], dir_body[0]))
    yaw_err   = math.degrees(math.atan2(dir_body[2], dir_body[0]))

    max_rate = self.max_roll_velocity
    desired_pitch_v = max(-max_rate, min(max_rate, pitch_err * 5.0))
    desired_yaw_v   = max(-max_rate, min(max_rate, yaw_err   * 5.0))

    self.omega_body[2] += (desired_pitch_v - self.omega_body[2]) * 0.5  # Z-axis = pitch
    self.omega_body[1] += (desired_yaw_v   - self.omega_body[1]) * 0.5  # Y-axis = yaw