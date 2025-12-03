# pyright: reportAttributeAccessIssue=false

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import json
import numpy as np
import tensorflow as tf
import physics
from config import BOX_SIZE

# ==========================
# CONFIG
# ==========================

INPUT_DIM = 21
MODEL_DIR = "models/jet_ai_control"
MODEL_PATH = os.path.join(MODEL_DIR, "policy.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "policy_best.keras")
STATS_PATH = os.path.join(MODEL_DIR, "training_stats.json")

# Policy-gradient hyperparams
GAMMA = 0.95                  # discount factor
POLICY_STD = 1.0              # fixed Gaussian std for exploration in action space
LEARNING_RATE = 3e-4          # Adam learning rate
MAX_GRAD_NORM = 5.0           # gradient clipping for stability

# "Memory" – how many past episodes to keep in replay
REPLAY_MAX_EPISODES = 20

# ==========================
# GLOBAL STATE
# ==========================

_model: tf.keras.Model | None = None
_optimizer: tf.keras.optimizers.Optimizer | None = None

_prev_state: np.ndarray | None = None
_prev_action: np.ndarray | None = None  # raw action (pre-tanh/sigmoid)

_episode_states: list[np.ndarray] = []
_episode_actions: list[np.ndarray] = []   # raw actions used by policy
_episode_rewards: list[float] = []

_reward_sum: float = 0.0
_tick_count: int = 0

# Replay buffer: list of episodes, each is dict("states","actions","rewards")
_replay_episodes: list[dict] = []

_best_avg_reward: float | None = None


# ==========================
# MODEL
# ==========================

def _build_model() -> tf.keras.Model:
    """MLP policy: state -> mean action (3D continuous)."""
    inp = tf.keras.Input(shape=(INPUT_DIM,), name="state")

    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # 3 outputs = mean of Gaussian for [pitch_raw, roll_raw, throttle_raw]
    out = tf.keras.layers.Dense(3, activation="linear")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


def _build_or_load_model() -> tf.keras.Model:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Prefer best-known model if it exists
    if os.path.exists(BEST_MODEL_PATH):
        print("Loading BEST JET AI model")
        return tf.keras.models.load_model(BEST_MODEL_PATH)

    if os.path.exists(MODEL_PATH):
        print("Loading latest JET AI model")
        return tf.keras.models.load_model(MODEL_PATH)

    print("Building new JET AI model")
    return _build_model()


def initialize_deeplearning():
    """Call once at the start of a simulation run."""
    global _model, _optimizer
    _model = _build_or_load_model() if _model is None else _model
    _optimizer = _optimizer or tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    _reset_episode()


def _reset_episode():
    """Reset data for the current episode only (not the replay memory)."""
    global _prev_state, _prev_action
    global _episode_states, _episode_actions, _episode_rewards
    global _reward_sum, _tick_count

    _prev_state = None
    _prev_action = None
    _episode_states = []
    _episode_actions = []
    _episode_rewards = []
    _reward_sum = 0.0
    _tick_count = 0


# ==========================
# STATE + REWARD
# ==========================

def _build_state(entities, jet) -> np.ndarray:
    """21-float context vector."""
    R = physics.get_rotation_matrix(jet.roll, jet.pitch, jet.yaw)
    Rt = R.T

    missiles = [e for e in entities if e is not jet and getattr(e, "shape", "") == "missile"]
    missile = None
    if missiles:
        missile = min(missiles, key=lambda m: np.linalg.norm(m.p - jet.p))

    ctx: list[float] = []

    # Missile relative pos/vel in body (6)
    if missile is None:
        ctx.extend([0.0, 0.0, 0.0])
        ctx.extend([0.0, 0.0, 0.0])
    else:
        rel_pos_world = missile.p - jet.p
        rel_pos_body = Rt @ rel_pos_world
        rel_vel_body = Rt @ (missile.v - jet.v)
        ctx.extend(rel_pos_body.tolist())
        ctx.extend(rel_vel_body.tolist())

    # Jet world position (3)
    ctx.extend(jet.p.tolist())

    # Jet velocity in body (3)
    v_body_jet = Rt @ jet.v
    ctx.extend(v_body_jet.tolist())

    # Jet orientation (pitch, yaw, roll) (3)
    ctx.extend([jet.pitch, jet.yaw, jet.roll])

    # Jet angular speeds (3)
    ctx.extend([jet.pitch_v, jet.yaw_v, jet.roll_v])

    # AoA, Cl, Cd (3)
    aoa = physics.get_aoa(jet.v, R)
    cl = physics.get_cl(jet.v, R, jet.max_lift_coefficient)
    cd = physics.get_cd(jet.v, R, jet.min_drag_coefficient, jet.max_drag_coefficient)
    ctx.append(aoa)
    ctx.append(cl)
    ctx.append(cd)

    return np.array(ctx, dtype=np.float32)


def _compute_reward(entities, jet) -> float:
    """
    Per-tick reward: maximize lift coefficient Cl.
    """
    # If on/under ground: clear penalty
    if jet.p[1] <= 0.0:
        return 0.0

    R = physics.get_rotation_matrix(jet.roll, jet.pitch, jet.yaw)
    cl = physics.get_cl(jet.v, R, jet.max_lift_coefficient)

    # Reward only positive lift; negative or tiny Cl gets ~0
    reward = max(float(cl), 0.0)/max(jet.roll_v, 1.0)
    return (float(np.linalg.norm(jet.v) * max(jet.p[1], 0))/1000000)*physics.get_control_surface_weight(jet.v, R)
    return reward


# ==========================
# POLICY STEP
# ==========================

def _sample_action(state: np.ndarray) -> np.ndarray:
    """
    Given a state, get mean action from the network and sample from a Gaussian around it.
    Returns raw action (pre-tanh/sigmoid) of shape (3,).
    """
    global _model

    assert _model is not None
    state_batch = state[None, :]
    mu = _model(state_batch, training=False).numpy()[0]  # mean action (3,)

    noise = np.random.normal(loc=0.0, scale=POLICY_STD, size=3).astype(np.float32)
    a_raw = mu + noise

    # Clip throttle raw a bit so sigmoid doesn't explode
    a_raw[2] = float(np.clip(a_raw[2], -5.0, 5.0))

    return a_raw.astype(np.float32)


def jet_ai_step(entities, jet):
    """
    Called every tick from guidance / entities.
    Returns (pitch_input, roll_input, throttle).
    """
    global _model, _prev_state, _prev_action
    global _episode_states, _episode_actions, _episode_rewards
    global _reward_sum, _tick_count

    if _model is None:
        initialize_deeplearning()

    # Build current state
    state = _build_state(entities, jet)

    # Reward previous action and store experience for that step
    if _prev_state is not None and _prev_action is not None:
        r = _compute_reward(entities, jet)
        jet.ai_reward = r

        _episode_states.append(_prev_state)
        _episode_actions.append(_prev_action)
        _episode_rewards.append(float(r))

        _reward_sum += r
        _tick_count += 1

    # Sample new raw action from current policy
    a_raw = _sample_action(state)
    pitch_raw, roll_raw, thr_raw = float(a_raw[0]), float(a_raw[1]), float(a_raw[2])

    # Map raw action to control inputs
    pitch_input = math.tanh(pitch_raw)              # [-1, 1]
    roll_input = math.tanh(roll_raw)                # [-1, 1]
    throttle = 1.0 / (1.0 + math.exp(-thr_raw))     # [0, 1]

    # Store for next step's reward assignment
    _prev_state = state
    _prev_action = a_raw

    return pitch_input, roll_input, throttle


# ==========================
# TRAINING (POLICY GRADIENT)
# ==========================

def _compute_discounted_returns(rewards: np.ndarray) -> np.ndarray:
    """Compute discounted returns G_t = r_t + γ r_{t+1} + ... for a single episode."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = float(rewards[t] + GAMMA * G)
        returns[t] = G
    return returns


def _append_epoch_stats(avg_reward: float, total_reward: float, tick_count: int):
    """
    Append one epoch's stats to training_stats.json in MODEL_DIR.
    Each entry: { "epoch": n, "avg_reward": ..., "total_reward": ..., "tick_count": ... }.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    epoch_index = len(data) + 1
    entry = {
        "epoch": epoch_index,
        "avg_reward": float(avg_reward),
        "total_reward": float(total_reward),
        "tick_count": int(tick_count)
    }
    data.append(entry)

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def cleanup_deeplearning():
    """
    Call once at the end of a simulation run.
    Uses collected (state, action, reward) to:
      - Add this episode to replay memory,
      - Train policy on ALL episodes in memory,
      - Log stats, save latest and best models,
      - Then reset current-episode buffers.
    """
    global _model, _optimizer
    global _episode_states, _episode_actions, _episode_rewards
    global _reward_sum, _tick_count, _prev_state, _prev_action
    global _replay_episodes, _best_avg_reward

    if (_model is None or
        not _episode_states or
        not _episode_actions or
        not _episode_rewards):
        # Nothing to train on for this episode
        _reset_episode()
        return

    assert _optimizer is not None

    # Convert current episode to arrays
    states_ep = np.array(_episode_states, dtype=np.float32)    # [T, INPUT_DIM]
    actions_ep = np.array(_episode_actions, dtype=np.float32)  # [T, 3]
    rewards_ep = np.array(_episode_rewards, dtype=np.float32)  # [T]

    # Episode-level stats (only current episode)
    avg_reward_ep = _reward_sum / float(max(_tick_count, 1))
    print(f"=====| Episode Average Reward: {avg_reward_ep}")

    # Log this episode's reward stats to JSON
    _append_epoch_stats(avg_reward_ep, _reward_sum, _tick_count)

    # Add current episode to replay memory
    ep = {
        "states": states_ep,
        "actions": actions_ep,
        "rewards": rewards_ep,
    }
    _replay_episodes.append(ep)
    if len(_replay_episodes) > REPLAY_MAX_EPISODES:
        _replay_episodes.pop(0)

    # Build training batch from ALL episodes in memory
    all_states = []
    all_actions = []
    all_returns = []

    for e in _replay_episodes:
        r = e["rewards"]
        G = _compute_discounted_returns(r)   # per-episode discounted returns
        all_states.append(e["states"])
        all_actions.append(e["actions"])
        all_returns.append(G)

    states = np.concatenate(all_states, axis=0)    # [N, INPUT_DIM]
    actions = np.concatenate(all_actions, axis=0)  # [N, 3]
    returns = np.concatenate(all_returns, axis=0)  # [N]

    # Normalize returns to get advantages
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns) + 1e-8)
    advantages = (returns - mean_ret) / std_ret    # zero-mean, unit-std

    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
    adv_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)

    var = POLICY_STD ** 2
    log_two_pi_var = math.log(2.0 * math.pi * var)

    with tf.GradientTape() as tape:
        mu = _model(states_tf, training=True)                   # [N, 3]
        diff = actions_tf - mu                                  # [N, 3]

        # log N(a | mu, sigma^2 I) per sample
        log_prob_per_dim = -0.5 * ((diff * diff) / var + log_two_pi_var)  # [N, 3]
        log_prob = tf.reduce_sum(log_prob_per_dim, axis=1)                 # [N]

        # REINFORCE loss: maximize E[adv * log_prob] => minimize negative
        loss = -tf.reduce_mean(log_prob * adv_tf)

    grads = tape.gradient(loss, _model.trainable_variables)
    if grads is not None:
        if MAX_GRAD_NORM is not None:
            grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        _optimizer.apply_gradients(zip(grads, _model.trainable_variables))
    else:
        print("Warning: No gradients computed; skipping optimizer step.")

    print(f"Policy loss: {float(loss):.6f}")

    # Save updated "latest" model
    os.makedirs(MODEL_DIR, exist_ok=True)
    _model.save(MODEL_PATH)
    print("Saved latest JET AI model")

    # Save BEST model if this episode beat previous best avg reward
    if _best_avg_reward is None or avg_reward_ep > _best_avg_reward:
        _best_avg_reward = avg_reward_ep
        _model.save(BEST_MODEL_PATH)
        print("New BEST JET AI model saved")

    # Reset episode buffers (but keep replay memory and model for next episodes)
    _reset_episode()
