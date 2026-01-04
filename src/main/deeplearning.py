from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import physics


def _get_simulation_config():
    import simulation as _sim
    return {
        "box": np.array(_sim.SIMULATION_BOX_SIZE, dtype=float).reshape(3),
        "tickrate": float(_sim.SIMULATION_TICKRATE),
        "duration": float(_sim.SIMULATION_DURATION),
    }


SIM_BOX: Optional[np.ndarray] = None
SIM_TICKRATE: Optional[float] = None
SIM_DURATION: Optional[float] = None

POS_SCALE = 40000.0
ALT_SCALE = 10000.0
VEL_SCALE = 1200.0
REL_VEL_SCALE = 1500.0
OMEGA_SCALE = 180.0
AOA_SCALE = 45.0
SIDESLIP_SCALE = 45.0

N_MISSILES_IN_STATE = int(os.getenv("JET_AI_MISSILES_IN_STATE", "2"))
MISSILE_FEATS = 8
JET_FEATS = 24
INPUT_DIM = N_MISSILES_IN_STATE * MISSILE_FEATS + JET_FEATS
ACTION_DIM = 4

GAMMA_PER_SEC = float(os.getenv("JET_AI_GAMMA_PER_SEC", os.getenv("JET_AI_GAMMA", "0.99")))
GAE_LAMBDA_PER_SEC = float(os.getenv("JET_AI_GAE_LAMBDA_PER_SEC", os.getenv("JET_AI_GAE_LAMBDA", "0.95")))
GAMMA_TICK = 0.99
GAE_LAMBDA_TICK = 0.95

PPO_CLIP_EPS = float(os.getenv("JET_AI_CLIP_EPS", "0.2"))
LEARNING_RATE = float(os.getenv("JET_AI_LR", "3e-4"))
MAX_GRAD_NORM = float(os.getenv("JET_AI_MAX_GRAD_NORM", "5.0"))

VALUE_COEF = float(os.getenv("JET_AI_VALUE_COEF", "0.5"))
ENTROPY_COEF = float(os.getenv("JET_AI_ENTROPY_COEF", "0.002"))
TARGET_KL = float(os.getenv("JET_AI_TARGET_KL", "0.02"))

UPDATE_EPOCHS = int(os.getenv("JET_AI_UPDATE_EPOCHS", "4"))
MINIBATCH_SIZE = int(os.getenv("JET_AI_MINIBATCH", "256"))
ROLLOUT_STEPS = int(os.getenv("JET_AI_ROLLOUT_STEPS", "2048"))
MIN_STEPS_TO_TRAIN = int(os.getenv("JET_AI_MIN_STEPS_TO_TRAIN", "256"))

CONTROL_HZ = float(os.getenv("JET_AI_CONTROL_HZ", "15"))
CONTROL_INTERVAL_TICKS = 1

ACTION_SMOOTHING_ALPHA = float(os.getenv("JET_AI_SMOOTH_ALPHA", "0.15"))
MAX_ACTION_DELTA_PER_DECISION = float(os.getenv("JET_AI_MAX_DELTA", "0.10"))

LOG_STD_MIN = float(os.getenv("JET_AI_LOG_STD_MIN", "-2.0"))
LOG_STD_MAX = float(os.getenv("JET_AI_LOG_STD_MAX", "-0.3"))
INIT_STD = float(os.getenv("JET_AI_INIT_STD", "0.4"))

REWARD_PHASE = int(os.getenv("JET_AI_REWARD_PHASE", "2"))

_DT = 1.0 / 60.0
MAX_EPISODE_TICKS = 1200

EVAL_EVERY_EPISODES = int(os.getenv("JET_AI_EVAL_EVERY", "0"))
EVAL_WITH_DETERMINISTIC_POLICY = bool(int(os.getenv("JET_AI_EVAL_DETERMINISTIC", "1")))

MODEL_TAG = os.getenv("JET_AI_MODEL_TAG", "jet_ppo")
REPO_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
MODEL_DIR = REPO_ROOT / "models" / MODEL_TAG
MODEL_PATH = MODEL_DIR / "actor_critic.keras"
BEST_MODEL_PATH = MODEL_DIR / "actor_critic_best.keras"
STATS_PATH = MODEL_DIR / "training_stats.json"

DEATH_PENALTY = float(os.getenv("JET_AI_DEATH_PENALTY", "20.0"))
GROUND_PENALTY = float(os.getenv("JET_AI_GROUND_PENALTY", "20.0"))
OUT_OF_BOUNDS_PENALTY = float(os.getenv("JET_AI_OOB_PENALTY", "20.0"))

P1_ALIVE_REWARD_PER_SEC = float(os.getenv("JET_AI_P1_ALIVE_PER_SEC", "1.0"))
P1_LOW_ALT_BUFFER_M = 300.0
P1_LOW_ALT_PENALTY_PER_SEC = 2.00
P1_MIN_SPEED_MPS = 80.0
P1_STALL_PENALTY_PER_SEC = 1.00
P1_AOA_SOFT_DEG = 25.0
P1_AOA_HARD_DEG = 45.0
P1_AOA_PENALTY_PER_SEC = 0.50
P1_CTRL_EFF_PENALTY_PER_SEC = 0.30
P1_EFFORT_PENALTY_PER_SEC = 0.01
P1_JERK_PENALTY_PER_SEC = 0.05
P1_OMEGA_PENALTY_PER_SEC = float(os.getenv("JET_AI_P1_OMEGA_PENALTY_PER_SEC", "0.02"))

P2_ALIVE_REWARD_PER_SEC = 0.10
P2_HIT_RADIUS_M = 200.0
P2_CLOSE_RADIUS_M = 2500.0
P2_DISTANCE_RATE_REWARD_PER_SEC = 1.50
P2_CLOSING_PENALTY_PER_SEC = 1.00
P2_CLOSE_BARRIER_PENALTY_PER_SEC = 2.00
P2_AOA_SIGMA_DEG = 6.0
P2_AOA_REWARD_PER_SEC = 0.30
P2_SPEED_TARGET_MPS = 250.0
P2_SPEED_REWARD_PER_SEC = 0.20
P2_OMEGA_TARGET_DPS = 90.0
P2_OMEGA_REWARD_PER_SEC = 0.20
P2_OMEGA_MAX_DPS = 220.0
P2_OMEGA_OVER_PENALTY_PER_SEC = 0.35
P2_EDGE_BUFFER_FRAC = 0.15
P2_EDGE_PENALTY_PER_SEC = 0.50


P3_ALIVE_REWARD_PER_SEC = 0.07
P3_HIT_RADIUS_M = 200.0
P3_CLOSE_RADIUS_M = 3000.0
P3_DISTANCE_RATE_REWARD_PER_SEC = 1.75
P3_CLOSING_PENALTY_PER_SEC = 1.25
P3_CLOSE_BARRIER_PENALTY_PER_SEC = 2.50

P4_ALIVE_REWARD_PER_SEC = 0.03
P4_HIT_RADIUS_M = 200.0
P4_CLOSE_RADIUS_M = 3500.0
P4_DISTANCE_RATE_REWARD_PER_SEC = 0.50
P4_CLOSING_PENALTY_PER_SEC = 0.50
P4_CLOSE_BARRIER_PENALTY_PER_SEC = 0.50


def _ensure_sim_config() -> None:
    global SIM_BOX, SIM_TICKRATE, SIM_DURATION
    global POS_SCALE, ALT_SCALE, _DT, MAX_EPISODE_TICKS, CONTROL_INTERVAL_TICKS, GAMMA_TICK, GAE_LAMBDA_TICK

    if SIM_BOX is not None:
        return

    sim_cfg = _get_simulation_config()
    SIM_BOX = sim_cfg["box"]
    SIM_TICKRATE = sim_cfg["tickrate"]
    SIM_DURATION = sim_cfg["duration"]

    POS_SCALE = float(max(float(SIM_BOX[0]), float(SIM_BOX[2]), 1.0))
    ALT_SCALE = float(max(float(SIM_BOX[1]), 1.0))
    _DT = 1.0 / float(max(1.0, SIM_TICKRATE))
    MAX_EPISODE_TICKS = int(max(1.0, SIM_DURATION) * max(1.0, SIM_TICKRATE))

    effective_control_hz = float(max(1e-6, CONTROL_HZ))
    CONTROL_INTERVAL_TICKS = int(max(1, round(float(SIM_TICKRATE) / effective_control_hz)))

    GAMMA_TICK = float(GAMMA_PER_SEC) ** float(_DT)
    GAE_LAMBDA_TICK = float(GAE_LAMBDA_PER_SEC) ** float(_DT)


class GlobalLogStd(tf.keras.layers.Layer):
    def __init__(self, action_dim: int, init_std: float = INIT_STD, **kwargs: Any):
        super().__init__(**kwargs)
        self.action_dim = int(action_dim)
        self.init_std = float(init_std)

    def build(self, input_shape: tf.TensorShape) -> None:
        init_value = float(math.log(max(self.init_std, 1e-6)))
        self.log_std = self.add_weight(
            name="log_std",
            shape=(self.action_dim,),
            initializer=tf.keras.initializers.Constant(init_value),
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        clipped = tf.clip_by_value(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        batch_count = tf.shape(inputs)[0]
        return tf.broadcast_to(clipped, (batch_count, self.action_dim))

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"action_dim": self.action_dim, "init_std": self.init_std})
        return cfg


def _build_model() -> tf.keras.Model:
    print("=== Building new AI model")
    inp = tf.keras.Input(shape=(INPUT_DIM,), name="state")
    x1 = tf.keras.layers.Dense(256, activation="relu")(inp)
    x2 = tf.keras.layers.Dense(256, activation="relu")(x1)
    x3 = tf.keras.layers.Dense(128, activation="relu")(x2)
    mu = tf.keras.layers.Dense(ACTION_DIM, activation="linear", name="mu")(x3)
    value = tf.keras.layers.Dense(1, activation="linear", name="value")(x3)
    log_std = GlobalLogStd(ACTION_DIM, init_std=INIT_STD, name="global_log_std")(inp)
    return tf.keras.Model(inputs=inp, outputs=[mu, value, log_std], name="jet_actor_critic")


def _load_model(model_path: Path) -> Optional[tf.keras.Model]:
    print("=== Loading AI model")
    try:
        loaded = tf.keras.models.load_model(str(model_path), custom_objects={"GlobalLogStd": GlobalLogStd})
        if not isinstance(loaded.outputs, list) or len(loaded.outputs) != 3:
            return None
        return loaded
    except Exception:
        return None


def _build_or_load_model() -> tf.keras.Model:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in (BEST_MODEL_PATH, MODEL_PATH):
        if candidate.exists():
            loaded = _load_model(candidate)
            if loaded is not None:
                return loaded
    return _build_model()


_model: Optional[tf.keras.Model] = None
_optimizer: Optional[tf.keras.optimizers.Optimizer] = None

_roll_states: List[np.ndarray] = []
_roll_actions: List[np.ndarray] = []
_roll_logps: List[float] = []
_roll_values: List[float] = []
_roll_rewards: List[float] = []
_roll_dones: List[float] = []

_episode_reward_sum: float = 0.0
_episode_tick_count: int = 0
_episode_done: bool = False
_episode_jet_id: Optional[int] = None
_episode_index: int = 0

_best_avg_reward: Optional[float] = None
_prev_nearest_missile_dist: Optional[float] = None
_prev_controls_for_reward: Optional[np.ndarray] = None
_last_controls: Optional[np.ndarray] = None

_update_count: int = 0
_started: bool = False
_decision_state: Optional[np.ndarray] = None
_decision_action_raw: Optional[np.ndarray] = None
_decision_logp: Optional[float] = None
_decision_value: Optional[float] = None
_reward_accum: float = 0.0
_since_decision_ticks: int = 0
_is_eval_episode: bool = False


def initialize_deeplearning(seed: Optional[int] = None) -> None:
    global _model, _optimizer
    _ensure_sim_config()
    if seed is not None:
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))
    if _model is None:
        _model = _build_or_load_model()
    if _optimizer is None:
        _optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    _reset_episode(None)


def _reset_episode(jet_id: Optional[int]) -> None:
    global _episode_reward_sum, _episode_tick_count, _episode_done, _episode_jet_id, _episode_index
    global _prev_nearest_missile_dist, _prev_controls_for_reward, _last_controls
    global _started, _decision_state, _decision_action_raw, _decision_logp, _decision_value, _reward_accum, _since_decision_ticks, _is_eval_episode

    _episode_reward_sum = 0.0
    _episode_tick_count = 0
    _episode_done = False
    _episode_jet_id = jet_id
    _prev_nearest_missile_dist = None
    _prev_controls_for_reward = None
    _last_controls = None
    _started = False
    _decision_state = None
    _decision_action_raw = None
    _decision_logp = None
    _decision_value = None
    _reward_accum = 0.0
    _since_decision_ticks = 0

    _episode_index += 1
    if EVAL_EVERY_EPISODES > 0 and (_episode_index % EVAL_EVERY_EPISODES) == 0:
        _is_eval_episode = True
    else:
        _is_eval_episode = False


def _is_missile(entity_obj: Any) -> bool:
    return entity_obj.__class__.__name__.lower() == "missile"


def _safe_unit(vector_value: np.ndarray) -> np.ndarray:
    vector_norm = float(np.linalg.norm(vector_value))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=float)
    return vector_value / vector_norm


def _clip_finite(array_value: np.ndarray, low: float = -10.0, high: float = 10.0) -> np.ndarray:
    clean = np.nan_to_num(array_value, nan=0.0, posinf=high, neginf=low)
    return np.clip(clean, low, high)


def _out_of_bounds(position_value: np.ndarray) -> bool:
    assert SIM_BOX is not None
    half_extents = SIM_BOX
    return (
        abs(float(position_value[0])) > float(half_extents[0])
        or float(position_value[1]) < 0.0
        or float(position_value[1]) > float(half_extents[1])
        or abs(float(position_value[2])) > float(half_extents[2])
    )


def _build_state(entities: List[Any], jet: Any) -> np.ndarray:
    position_value = np.array(getattr(jet, "position", np.zeros(3)), dtype=float).reshape(3)
    velocity_value = np.array(getattr(jet, "velocity", np.zeros(3)), dtype=float).reshape(3)
    orientation_value = np.array(getattr(jet, "orientation", np.eye(3)), dtype=float).reshape(3, 3)
    transpose_orientation = orientation_value.T
    omega_value = np.array(getattr(jet, "omega", np.zeros(3)), dtype=float).reshape(3)

    control_data = getattr(jet, "control_inputs", {"pitch": 0.0, "roll": 0.0, "yaw": 0.0})
    pitch_value = float(control_data.get("pitch", 0.0))
    roll_value = float(control_data.get("roll", 0.0))
    yaw_value = float(control_data.get("yaw", 0.0))
    throttle_value = float(getattr(jet, "throttle", 0.0))

    missile_list = [e for e in entities if e is not jet and _is_missile(e)]
    missile_list.sort(
        key=lambda m: float(
            np.linalg.norm(np.array(getattr(m, "position", position_value), dtype=float).reshape(3) - position_value)
        )
    )

    feats: List[float] = []
    for index_value in range(int(N_MISSILES_IN_STATE)):
        if index_value >= len(missile_list):
            feats.extend([0.0] * MISSILE_FEATS)
            continue
        missile_obj = missile_list[index_value]
        missile_pos = np.array(getattr(missile_obj, "position", position_value), dtype=float).reshape(3)
        missile_vel = np.array(getattr(missile_obj, "velocity", np.zeros(3)), dtype=float).reshape(3)
        relative_pos_world = missile_pos - position_value
        relative_vel_world = missile_vel - velocity_value
        relative_pos_body = transpose_orientation @ relative_pos_world
        relative_vel_body = transpose_orientation @ relative_vel_world
        distance_value = float(np.linalg.norm(relative_pos_world))
        line_of_sight = _safe_unit(relative_pos_world)
        closing_speed = float(-np.dot(relative_vel_world, line_of_sight))
        feats.extend((relative_pos_body / POS_SCALE).tolist())
        feats.extend((relative_vel_body / REL_VEL_SCALE).tolist())
        feats.append(distance_value / POS_SCALE)
        feats.append(closing_speed / REL_VEL_SCALE)

    feats.append(float(position_value[0] / POS_SCALE))
    feats.append(float(position_value[2] / POS_SCALE))
    feats.append(float(position_value[1] / ALT_SCALE))

    body_velocity = transpose_orientation @ velocity_value
    feats.extend((body_velocity / VEL_SCALE).tolist())
    speed_value = float(np.linalg.norm(velocity_value))
    feats.append(speed_value / VEL_SCALE)

    forward_dir = physics.get_forward_dir(orientation_value)
    up_dir = physics.get_up_dir(orientation_value)
    feats.extend(np.array(forward_dir, dtype=float).reshape(3).tolist())
    feats.extend(np.array(up_dir, dtype=float).reshape(3).tolist())

    feats.extend((omega_value / OMEGA_SCALE).tolist())

    aoa_value = float(physics.get_angle_of_attack(velocity_value, orientation_value))
    sideslip_value = float(physics.get_sideslip(velocity_value, orientation_value)) if hasattr(physics, "get_sideslip") else 0.0
    feats.append(aoa_value / AOA_SCALE)
    feats.append(sideslip_value / SIDESLIP_SCALE)

    optimal_aoa = float(getattr(jet, "optimal_lift_aoa", 10.0))
    ctrl_eff_value = float(physics.get_control_effectiveness(velocity_value, orientation_value, optimal_aoa))
    feats.append(ctrl_eff_value)

    feats.append(throttle_value)
    feats.append(pitch_value)
    feats.append(roll_value)
    feats.append(yaw_value)

    time_norm_value = float(_episode_tick_count) / float(max(1, MAX_EPISODE_TICKS))
    feats.append(time_norm_value)

    state_array = np.array(feats, dtype=np.float32)
    state_array = _clip_finite(state_array, -10.0, 10.0)
    if state_array.shape[0] != INPUT_DIM:
        raise RuntimeError(f"State size mismatch {state_array.shape[0]} vs {INPUT_DIM}")
    return state_array


def _nearest_missile_metrics(entities: List[Any], jet: Any) -> Tuple[Optional[float], Optional[float]]:
    position_value = np.array(getattr(jet, "position", np.zeros(3)), dtype=float).reshape(3)
    velocity_value = np.array(getattr(jet, "velocity", np.zeros(3)), dtype=float).reshape(3)
    missile_list = [e for e in entities if e is not jet and _is_missile(e)]
    if not missile_list:
        return None, None
    best_distance: Optional[float] = None
    best_closing: Optional[float] = None
    for missile_obj in missile_list:
        missile_pos = np.array(getattr(missile_obj, "position", position_value), dtype=float).reshape(3)
        missile_vel = np.array(getattr(missile_obj, "velocity", np.zeros(3)), dtype=float).reshape(3)
        relative_pos = missile_pos - position_value
        distance_value = float(np.linalg.norm(relative_pos))
        if best_distance is None or distance_value < best_distance:
            line_of_sight = _safe_unit(relative_pos)
            closing_speed = float(-np.dot(missile_vel - velocity_value, line_of_sight))
            best_distance = distance_value
            best_closing = closing_speed
    return best_distance, best_closing


def _reward_and_done(entities: List[Any], jet: Any) -> Tuple[float, float]:
    global _prev_nearest_missile_dist, _prev_controls_for_reward, _episode_done

    if _episode_done:
        return 0.0, 1.0

    position_value = np.array(getattr(jet, "position", np.zeros(3)), dtype=float).reshape(3)
    velocity_value = np.array(getattr(jet, "velocity", np.zeros(3)), dtype=float).reshape(3)
    orientation_value = np.array(getattr(jet, "orientation", np.eye(3)), dtype=float).reshape(3, 3)
    omega_value = np.array(getattr(jet, "omega", np.zeros(3)), dtype=float).reshape(3)

    alive_flag = bool(getattr(jet, "alive", True))
    altitude_value = float(position_value[1])
    speed_value = float(np.linalg.norm(velocity_value))

    if not alive_flag:
        _episode_done = True
        _prev_nearest_missile_dist = None
        _prev_controls_for_reward = None
        return -DEATH_PENALTY, 1.0

    if altitude_value <= 0.0:
        _episode_done = True
        _prev_nearest_missile_dist = None
        _prev_controls_for_reward = None
        return -GROUND_PENALTY, 1.0

    if _out_of_bounds(position_value):
        _episode_done = True
        _prev_nearest_missile_dist = None
        _prev_controls_for_reward = None
        return -OUT_OF_BOUNDS_PENALTY, 1.0

    def envelope_reward(scale_value: float) -> float:
        reward_value = 0.0
        reward_value -= scale_value * P1_LOW_ALT_PENALTY_PER_SEC * math.exp(-max(0.0, altitude_value) / max(1.0, P1_LOW_ALT_BUFFER_M)) * _DT
        if speed_value < P1_MIN_SPEED_MPS:
            reward_value -= scale_value * P1_STALL_PENALTY_PER_SEC * (1.0 - speed_value / max(1e-6, P1_MIN_SPEED_MPS)) * _DT
        aoa_abs = abs(float(physics.get_angle_of_attack(velocity_value, orientation_value)))
        if aoa_abs > P1_AOA_SOFT_DEG:
            aoa_span = max(1e-6, (P1_AOA_HARD_DEG - P1_AOA_SOFT_DEG))
            aoa_ratio = min((aoa_abs - P1_AOA_SOFT_DEG) / aoa_span, 1.0)
            reward_value -= scale_value * P1_AOA_PENALTY_PER_SEC * (aoa_ratio * aoa_ratio) * _DT
        optimal_aoa_local = float(getattr(jet, "optimal_lift_aoa", 10.0))
        ctrl_eff_value = float(physics.get_control_effectiveness(velocity_value, orientation_value, optimal_aoa_local))
        reward_value -= scale_value * P1_CTRL_EFF_PENALTY_PER_SEC * (1.0 - ctrl_eff_value) * _DT
        omega_mag = float(np.linalg.norm(omega_value))
        omega_ratio = min(omega_mag / max(1e-6, OMEGA_SCALE), 5.0)
        reward_value -= scale_value * P1_OMEGA_PENALTY_PER_SEC * (omega_ratio * omega_ratio) * _DT
        return float(reward_value)

    def control_regularizers(scale_value: float) -> float:
        global _prev_controls_for_reward
        control_data = getattr(jet, "control_inputs", {"pitch": 0.0, "roll": 0.0, "yaw": 0.0})
        pitch_value = float(control_data.get("pitch", 0.0))
        roll_value = float(control_data.get("roll", 0.0))
        yaw_value = float(control_data.get("yaw", 0.0))
        throttle_value = float(getattr(jet, "throttle", 0.0))
        control_vector = np.array([pitch_value, roll_value, yaw_value, throttle_value], dtype=float)
        control_weight = np.array([1.0, 1.0, 1.0, 0.3], dtype=float)
        reward_value = 0.0
        reward_value -= scale_value * P1_EFFORT_PENALTY_PER_SEC * float(np.sum((control_vector * control_vector) * control_weight)) * _DT
        if _prev_controls_for_reward is not None:
            control_delta = control_vector - _prev_controls_for_reward
            reward_value -= scale_value * P1_JERK_PENALTY_PER_SEC * float(np.sum((control_delta * control_delta) * control_weight)) * _DT
        _prev_controls_for_reward = control_vector
        return float(reward_value)

    def missile_shaping(hit_radius: float, close_radius: float, close_barrier: float, dist_rate_reward: float, closing_penalty: float) -> float:
        global _prev_nearest_missile_dist, _episode_done
        nearest_dist, nearest_closing = _nearest_missile_metrics(entities, jet)
        if nearest_dist is None:
            _prev_nearest_missile_dist = None
            return 0.0
        if nearest_dist < hit_radius:
            _episode_done = True
            _prev_nearest_missile_dist = None
            _prev_controls_for_reward = None
            return -DEATH_PENALTY
        reward_value = 0.0
        reward_value -= close_barrier * math.exp(-max(0.0, nearest_dist) / max(1.0, close_radius)) * _DT
        if _prev_nearest_missile_dist is not None:
            dist_rate = (nearest_dist - _prev_nearest_missile_dist) / max(1e-6, _DT)
            reward_value += dist_rate_reward * (dist_rate / REL_VEL_SCALE) * _DT
        _prev_nearest_missile_dist = nearest_dist
        if nearest_closing is not None and nearest_closing > 0.0:
            reward_value -= closing_penalty * (nearest_closing / REL_VEL_SCALE) * _DT
        return float(reward_value)

    total_reward = 0.0

    if REWARD_PHASE == 1:
        total_reward += P1_ALIVE_REWARD_PER_SEC * _DT
        total_reward += envelope_reward(1.0)
        total_reward += control_regularizers(1.0)
        return float(total_reward), 0.0

    if REWARD_PHASE == 2:
        total_reward += P2_ALIVE_REWARD_PER_SEC * _DT
        total_reward += envelope_reward(1.0)

        aoa_value = float(physics.get_angle_of_attack(velocity_value, orientation_value))
        optimal_aoa = float(getattr(jet, "optimal_lift_aoa", 10.0))
        aoa_error = abs(aoa_value - optimal_aoa)
        aoa_sigma = float(max(1e-6, P2_AOA_SIGMA_DEG))
        aoa_score = math.exp(-(aoa_error * aoa_error) / (2.0 * aoa_sigma * aoa_sigma))
        total_reward += P2_AOA_REWARD_PER_SEC * aoa_score * _DT

        speed_target = float(max(1e-6, P2_SPEED_TARGET_MPS))
        speed_score = min(max(speed_value / speed_target, 0.0), 1.0)
        total_reward += P2_SPEED_REWARD_PER_SEC * speed_score * _DT

        omega_mag = float(np.linalg.norm(omega_value))
        omega_target = float(max(1e-6, P2_OMEGA_TARGET_DPS))
        omega_score = min(max(omega_mag / omega_target, 0.0), 1.0)
        total_reward += P2_OMEGA_REWARD_PER_SEC * omega_score * _DT

        omega_max = float(max(1e-6, P2_OMEGA_MAX_DPS))
        if omega_mag > omega_max:
            omega_over = (omega_mag - omega_max) / omega_max
            total_reward -= P2_OMEGA_OVER_PENALTY_PER_SEC * (omega_over * omega_over) * _DT

        margin_x = float(SIM_BOX[0]) - abs(float(position_value[0]))
        margin_z = float(SIM_BOX[2]) - abs(float(position_value[2]))
        horizontal_margin = max(0.0, min(margin_x, margin_z))
        edge_buffer = float(max(1.0, P2_EDGE_BUFFER_FRAC * min(float(SIM_BOX[0]), float(SIM_BOX[2]))))
        total_reward -= P2_EDGE_PENALTY_PER_SEC * math.exp(-horizontal_margin / edge_buffer) * _DT

        total_reward += control_regularizers(1.0)
        return float(total_reward), float(1.0 if _episode_done else 0.0)


    if REWARD_PHASE == 3:
        total_reward += P3_ALIVE_REWARD_PER_SEC * _DT
        total_reward += envelope_reward(0.75)
        missile_list = [e for e in entities if e is not jet and _is_missile(e)]
        threat_list: List[Tuple[float, float]] = []
        jet_pos = position_value
        jet_vel = velocity_value
        for missile_obj in missile_list:
            missile_pos = np.array(getattr(missile_obj, "position", jet_pos), dtype=float).reshape(3)
            missile_vel = np.array(getattr(missile_obj, "velocity", np.zeros(3)), dtype=float).reshape(3)
            relative_pos = missile_pos - jet_pos
            distance_value = float(np.linalg.norm(relative_pos))
            if distance_value > 0.0:
                line_of_sight = _safe_unit(relative_pos)
                closing_speed = float(-np.dot(missile_vel - jet_vel, line_of_sight))
                threat_list.append((distance_value, closing_speed))
        if threat_list:
            threat_list.sort(key=lambda value_pair: value_pair[0])
            distance_one, closing_one = threat_list[0]
            if distance_one < P3_HIT_RADIUS_M:
                _episode_done = True
                return -DEATH_PENALTY, 1.0
            total_reward -= P3_CLOSE_BARRIER_PENALTY_PER_SEC * math.exp(-max(0.0, distance_one) / max(1.0, P3_CLOSE_RADIUS_M)) * _DT
            if len(threat_list) > 1:
                distance_two, closing_two = threat_list[1]
                total_reward -= 0.50 * P3_CLOSE_BARRIER_PENALTY_PER_SEC * math.exp(-max(0.0, distance_two) / max(1.0, P3_CLOSE_RADIUS_M)) * _DT
                if closing_two > 0.0:
                    total_reward -= 0.50 * P3_CLOSING_PENALTY_PER_SEC * (closing_two / REL_VEL_SCALE) * _DT
            if _prev_nearest_missile_dist is not None:
                dist_rate = (distance_one - _prev_nearest_missile_dist) / max(1e-6, _DT)
                total_reward += P3_DISTANCE_RATE_REWARD_PER_SEC * (dist_rate / REL_VEL_SCALE) * _DT
            _prev_nearest_missile_dist = distance_one
            if closing_one > 0.0:
                total_reward -= P3_CLOSING_PENALTY_PER_SEC * (closing_one / REL_VEL_SCALE) * _DT
        else:
            _prev_nearest_missile_dist = None
        total_reward += control_regularizers(0.75)
        return float(total_reward), float(1.0 if _episode_done else 0.0)

    if REWARD_PHASE == 4:
        total_reward += P4_ALIVE_REWARD_PER_SEC * _DT
        total_reward += envelope_reward(0.25)
        total_reward += missile_shaping(P4_HIT_RADIUS_M, P4_CLOSE_RADIUS_M, P4_CLOSE_BARRIER_PENALTY_PER_SEC, P4_DISTANCE_RATE_REWARD_PER_SEC, P4_CLOSING_PENALTY_PER_SEC)
        total_reward += control_regularizers(0.25)
        return float(total_reward), float(1.0 if _episode_done else 0.0)

    total_reward += P1_ALIVE_REWARD_PER_SEC * _DT
    total_reward += envelope_reward(1.0)
    total_reward += control_regularizers(1.0)
    return float(total_reward), 0.0


def _normal_log_prob(action_tensor: tf.Tensor, mean_tensor: tf.Tensor, log_std_tensor: tf.Tensor) -> tf.Tensor:
    std_tensor = tf.exp(log_std_tensor)
    var_tensor = std_tensor * std_tensor
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype=tf.float32)
    return tf.reduce_sum(-0.5 * (((action_tensor - mean_tensor) ** 2) / var_tensor + 2.0 * log_std_tensor + log_two_pi), axis=1)


def _normal_entropy(log_std_tensor: tf.Tensor) -> tf.Tensor:
    entropy_const = 0.5 * float(math.log(2.0 * math.pi * math.e))
    return tf.reduce_sum(log_std_tensor + entropy_const, axis=1)


def _raw_to_controls(action_raw: np.ndarray) -> np.ndarray:
    pitch_value = float(np.tanh(action_raw[0]))
    roll_value = float(np.tanh(action_raw[1]))
    yaw_value = float(np.tanh(action_raw[2]))
    throttle_value = float(1.0 / (1.0 + math.exp(-float(action_raw[3]))))
    return np.array([pitch_value, roll_value, yaw_value, throttle_value], dtype=np.float32)


def _smooth_controls(control_cmd: np.ndarray) -> np.ndarray:
    global _last_controls
    if _last_controls is None:
        _last_controls = control_cmd.copy()
        return control_cmd
    smoothed = (1.0 - ACTION_SMOOTHING_ALPHA) * _last_controls + ACTION_SMOOTHING_ALPHA * control_cmd
    delta_value = smoothed - _last_controls
    limited_delta = np.clip(delta_value, -MAX_ACTION_DELTA_PER_DECISION, MAX_ACTION_DELTA_PER_DECISION)
    final_value = _last_controls + limited_delta
    _last_controls = final_value.copy()
    return final_value


def _policy_value(state_value: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    assert _model is not None
    state_tensor = tf.convert_to_tensor(state_value[None, :], dtype=tf.float32)
    mean_actions, value_output, log_std_output = _model(state_tensor, training=False)
    mean_np = mean_actions.numpy()[0].astype(np.float32)
    value_np = float(value_output.numpy()[0, 0])
    log_std_np = log_std_output.numpy()[0].astype(np.float32)
    return mean_np, value_np, log_std_np


def _sample_action_from_policy(state_value: np.ndarray, deterministic: bool) -> Tuple[np.ndarray, float, float]:
    mean_np, value_np, log_std_np = _policy_value(state_value)
    std_np = np.exp(log_std_np).astype(np.float32)
    if deterministic:
        action_raw = mean_np
    else:
        noise_np = np.random.normal(0.0, 1.0, size=ACTION_DIM).astype(np.float32)
        action_raw = mean_np + std_np * noise_np
    action_raw = np.clip(action_raw, -5.0, 5.0).astype(np.float32)
    action_tensor = tf.convert_to_tensor(action_raw[None, :], dtype=tf.float32)
    mean_tensor = tf.convert_to_tensor(mean_np[None, :], dtype=tf.float32)
    log_std_tensor = tf.convert_to_tensor(log_std_np[None, :], dtype=tf.float32)
    logp_value = float(_normal_log_prob(action_tensor, mean_tensor, log_std_tensor).numpy()[0])
    return action_raw, logp_value, value_np


def _compute_gae(reward_array: np.ndarray, done_array: np.ndarray, value_array: np.ndarray, bootstrap_value: float) -> Tuple[np.ndarray, np.ndarray]:
    count_steps = int(reward_array.shape[0])
    advantages = np.zeros(count_steps, dtype=np.float32)
    last_advantage = 0.0
    for index_value in reversed(range(count_steps)):
        next_non_terminal = 1.0 - float(done_array[index_value])
        next_value = bootstrap_value if index_value == count_steps - 1 else float(value_array[index_value + 1])
        delta_value = float(reward_array[index_value]) + GAMMA_TICK * next_value * next_non_terminal - float(value_array[index_value])
        last_advantage = delta_value + GAMMA_TICK * GAE_LAMBDA_TICK * next_non_terminal * last_advantage
        advantages[index_value] = last_advantage
    returns = advantages + value_array.astype(np.float32)
    return advantages.astype(np.float32), returns.astype(np.float32)


def _append_stats(entry: Dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if STATS_PATH.exists():
        try:
            data_list = json.loads(STATS_PATH.read_text(encoding="utf-8"))
            if not isinstance(data_list, list):
                data_list = []
        except Exception:
            data_list = []
    else:
        data_list = []
    data_list.append(entry)
    STATS_PATH.write_text(json.dumps(data_list, indent=2), encoding="utf-8")


def _ppo_update(bootstrap_value: float) -> None:
    global _roll_states, _roll_actions, _roll_logps, _roll_values, _roll_rewards, _roll_dones
    global _model, _optimizer, _update_count, _best_avg_reward

    if _model is None or _optimizer is None:
        return

    total_steps = len(_roll_states)
    if total_steps < MIN_STEPS_TO_TRAIN:
        return

    state_batch = np.asarray(_roll_states, dtype=np.float32)
    action_batch = np.asarray(_roll_actions, dtype=np.float32)
    logp_old_batch = np.asarray(_roll_logps, dtype=np.float32)
    value_batch = np.asarray(_roll_values, dtype=np.float32)
    reward_batch = np.asarray(_roll_rewards, dtype=np.float32)
    done_batch = np.asarray(_roll_dones, dtype=np.float32)

    advantage_batch, return_batch = _compute_gae(reward_batch, done_batch, value_batch, float(bootstrap_value))
    adv_mean = float(np.mean(advantage_batch))
    adv_std = float(np.std(advantage_batch) + 1e-8)
    advantage_norm = (advantage_batch - adv_mean) / adv_std

    state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
    action_tensor = tf.convert_to_tensor(action_batch, dtype=tf.float32)
    logp_old_tensor = tf.convert_to_tensor(logp_old_batch, dtype=tf.float32)
    advantage_tensor = tf.convert_to_tensor(advantage_norm, dtype=tf.float32)
    return_tensor = tf.convert_to_tensor(return_batch, dtype=tf.float32)

    batch_size = int(state_batch.shape[0])
    indices_base = np.arange(batch_size)

    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    approx_kl_sum = 0.0
    clip_frac_sum = 0.0
    minibatch_count = 0

    for epoch_index in range(UPDATE_EPOCHS):
        np.random.shuffle(indices_base)
        for start_index in range(0, batch_size, MINIBATCH_SIZE):
            end_index = min(batch_size, start_index + MINIBATCH_SIZE)
            mini_indices = indices_base[start_index:end_index]

            mini_states = tf.gather(state_tensor, mini_indices)
            mini_actions = tf.gather(action_tensor, mini_indices)
            mini_logp_old = tf.gather(logp_old_tensor, mini_indices)
            mini_adv = tf.gather(advantage_tensor, mini_indices)
            mini_returns = tf.gather(return_tensor, mini_indices)

            with tf.GradientTape() as tape:
                new_mean, new_value, new_log_std = _model(mini_states, training=True)
                new_value_flat = tf.squeeze(new_value, axis=1)

                new_logp = _normal_log_prob(mini_actions, new_mean, new_log_std)
                log_ratio = new_logp - mini_logp_old
                ratio = tf.exp(log_ratio)

                clipped_ratio = tf.clip_by_value(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS)
                surrogate_one = ratio * mini_adv
                surrogate_two = clipped_ratio * mini_adv
                policy_loss = -tf.reduce_mean(tf.minimum(surrogate_one, surrogate_two))

                value_loss = tf.reduce_mean(tf.square(mini_returns - new_value_flat))
                entropy = tf.reduce_mean(_normal_entropy(new_log_std))
                total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            grads = tape.gradient(total_loss, _model.trainable_variables)
            clipped_grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
            _optimizer.apply_gradients(zip(clipped_grads, _model.trainable_variables))

            approx_kl = tf.reduce_mean(mini_logp_old - new_logp)
            clipped_mask = tf.greater(tf.abs(ratio - 1.0), PPO_CLIP_EPS)
            clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, tf.float32))

            policy_loss_sum += float(policy_loss.numpy())
            value_loss_sum += float(value_loss.numpy())
            entropy_sum += float(entropy.numpy())
            approx_kl_sum += float(approx_kl.numpy())
            clip_frac_sum += float(clip_fraction.numpy())
            minibatch_count += 1

        avg_kl_epoch = approx_kl_sum / float(max(1, minibatch_count))
        if avg_kl_epoch > TARGET_KL:
            break

    mean_reward = float(np.mean(reward_batch))
    update_entry = {
        "update": int(_update_count),
        "steps": int(batch_size),
        "avg_reward": mean_reward,
        "policy_loss": float(policy_loss_sum / float(max(1, minibatch_count))),
        "value_loss": float(value_loss_sum / float(max(1, minibatch_count))),
        "entropy": float(entropy_sum / float(max(1, minibatch_count))),
        "approx_kl": float(approx_kl_sum / float(max(1, minibatch_count))),
        "clip_frac": float(clip_frac_sum / float(max(1, minibatch_count))),
        "phase": int(REWARD_PHASE),
        "control_interval": int(CONTROL_INTERVAL_TICKS),
    }
    _append_stats(update_entry)
    print(f"=== Episode ended: Avg reward/tick: {mean_reward:.4f}, Duration: {batch_size} ticks")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _model.save(str(MODEL_PATH), include_optimizer=False)
    except Exception:
        pass

    if _best_avg_reward is None or mean_reward > float(_best_avg_reward):
        _best_avg_reward = mean_reward
        try:
            _model.save(str(BEST_MODEL_PATH), include_optimizer=False)
        except Exception:
            pass

    _update_count += 1

    _roll_states = []
    _roll_actions = []
    _roll_logps = []
    _roll_values = []
    _roll_rewards = []
    _roll_dones = []


def _store_transition(reward_value: float, done_value: float, bootstrap_value: float) -> None:
    global _roll_states, _roll_actions, _roll_logps, _roll_values, _roll_rewards, _roll_dones
    global _decision_state, _decision_action_raw, _decision_logp, _decision_value, _reward_accum, _since_decision_ticks

    if _decision_state is None or _decision_action_raw is None or _decision_logp is None or _decision_value is None:
        _reward_accum = 0.0
        _since_decision_ticks = 0
        return

    if not _is_eval_episode:
        _roll_states.append(_decision_state)
        _roll_actions.append(_decision_action_raw)
        _roll_logps.append(float(_decision_logp))
        _roll_values.append(float(_decision_value))
        _roll_rewards.append(float(reward_value))
        _roll_dones.append(float(done_value))

        if len(_roll_states) >= ROLLOUT_STEPS:
            _ppo_update(float(bootstrap_value))

    _reward_accum = 0.0
    _since_decision_ticks = 0


def jet_ai_step(entities: List[Any], jet: Any) -> Tuple[float, float, float, float]:
    global _model, _optimizer
    global _episode_reward_sum, _episode_tick_count, _episode_done, _episode_jet_id
    global _started, _decision_state, _decision_action_raw, _decision_logp, _decision_value, _reward_accum, _since_decision_ticks

    if _model is None or _optimizer is None:
        initialize_deeplearning()

    jet_identity = id(jet)
    if _episode_jet_id is None or _episode_jet_id != jet_identity:
        _reset_episode(jet_identity)

    state_now = _build_state(entities, jet)

    if _started:
        reward_tick, done_tick = _reward_and_done(entities, jet)
        _reward_accum += float(reward_tick)
        _episode_reward_sum += float(reward_tick)
        _episode_tick_count += 1
        _since_decision_ticks += 1

        decision_due = (_since_decision_ticks >= int(CONTROL_INTERVAL_TICKS))
        if done_tick > 0.5 or _episode_tick_count >= MAX_EPISODE_TICKS:
            _store_transition(float(_reward_accum), 1.0, 0.0)
            if not _is_eval_episode:
                _ppo_update(0.0)
            _episode_done = True
            avg_reward_per_tick = _episode_reward_sum / float(max(1, _episode_tick_count))
            print(f"=== Episode ended: Avg reward/tick: {avg_reward_per_tick:.4f}, Duration: {_episode_tick_count} ticks")
            if _last_controls is not None:
                c = _last_controls
                return float(c[0]), float(c[1]), float(c[2]), float(c[3])
            return 0.0, 0.0, 0.0, float(getattr(jet, "throttle", 0.0))

        if decision_due:
            mean_np, value_now, _ = _policy_value(state_now)
            _store_transition(float(_reward_accum), 0.0, float(value_now))

    if _episode_done:
        if _last_controls is not None:
            c = _last_controls
            return float(c[0]), float(c[1]), float(c[2]), float(c[3])
        return 0.0, 0.0, 0.0, float(getattr(jet, "throttle", 0.0))

    decision_due_now = (not _started) or (_since_decision_ticks == 0)
    if decision_due_now:
        deterministic = bool(_is_eval_episode and EVAL_WITH_DETERMINISTIC_POLICY)
        action_raw, logp_value, value_used = _sample_action_from_policy(state_now, deterministic=deterministic)
        controls_cmd = _raw_to_controls(action_raw)
        controls_cmd = _smooth_controls(controls_cmd)
        try:
            jet.control_inputs["pitch"] = float(controls_cmd[0])
            jet.control_inputs["roll"] = float(controls_cmd[1])
            jet.control_inputs["yaw"] = float(controls_cmd[2])
            jet.throttle = float(controls_cmd[3])
        except Exception:
            pass

        _decision_state = state_now
        _decision_action_raw = action_raw
        _decision_logp = float(logp_value)
        _decision_value = float(value_used)

        _started = True
        _reward_accum = 0.0
        _since_decision_ticks = 0

        return float(controls_cmd[0]), float(controls_cmd[1]), float(controls_cmd[2]), float(controls_cmd[3])

    if _last_controls is not None:
        c = _last_controls
        try:
            jet.control_inputs["pitch"] = float(c[0])
            jet.control_inputs["roll"] = float(c[1])
            jet.control_inputs["yaw"] = float(c[2])
            jet.throttle = float(c[3])
        except Exception:
            pass
        return float(c[0]), float(c[1]), float(c[2]), float(c[3])

    return 0.0, 0.0, 0.0, float(getattr(jet, "throttle", 0.0))


def cleanup_deeplearning() -> None:
    global _model, _optimizer
    global _episode_done, _started, _reward_accum

    if _model is None or _optimizer is None:
        return

    if _started and (not _episode_done) and (len(_roll_states) >= MIN_STEPS_TO_TRAIN):
        _ppo_update(0.0)

    _reward_accum = 0.0
