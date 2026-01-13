import sys
import os
import warnings

# Suppress warnings before any heavy imports
warnings.filterwarnings("ignore")

import deeplearning as dl
import time
from typing import Any, List

import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import viz
from res.presets import sim_presets

SIMULATION_DURATION: float = 40.0
SIMULATION_TICKRATE: float = 60.0
SIMULATION_SPEED: float = 1.0
SIMULATION_BOX_SIZE = np.array([40000.0, 10000.0, 40000.0], dtype=float)
SIMULATION_RESOLUTION = np.array([1500, 500], dtype=int)


def run(epochs: int = 1, sprint: bool = False) -> None:
    epoch = 0
    while epochs <= 0 or epoch < epochs:
        phase = int(getattr(dl, "REWARD_PHASE", getattr(dl, "PHASE", 1)))
        entities = sim_presets.create_scenario(phase)
        run_epoch(entities, SIMULATION_DURATION, SIMULATION_TICKRATE, SIMULATION_SPEED, sprint)
        epoch += 1


def run_epoch(
    entities: List[Any],
    duration_s: float,
    tick_rate: float,
    sim_speed: float,
    sprint: bool,
) -> None:
    dl.initialize_deeplearning()
    dt_sim = 1.0 / float(tick_rate)
    dt_real_target = dt_sim / float(sim_speed)
    steps = int(float(duration_s) * float(tick_rate))
    jet_obj = next((e for e in entities if e.__class__.__name__.lower() == "jet"), None)

    print("=== Running epoch ===")
    next_tick = time.perf_counter()
    for _ in range(steps):
        for ent in entities:
            if not bool(getattr(ent, "alive", True)):
                continue
            ent.think(entities)
            ent.tick()

        if not sprint:
            viz.update_instances(entities)

        # MOD: flush final transition when physics kills the jet during tick()
        if jet_obj is not None and not bool(getattr(jet_obj, "alive", True)):
            try:
                dl.jet_ai_step(entities, jet_obj)
            except Exception:
                pass
            break

        if not sprint:
            next_tick += dt_real_target
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    dl.cleanup_deeplearning()
