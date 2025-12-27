import numpy as np
import random
import presets
import time
from viz import update_instances, clean_viz
SIMULATION_DURATION = 120
SIMULATION_TICKRATE = 60  # ticks per second
SIMULATION_SPEED = 1  # real-time speed multiplier
SIMULATION_BOX_SIZE = np.array([20000.0, 10000.0, 20000.0])  # 20km x 10km x 20km box
SIMULATION_RESOLUTION = np.array([1500,500])

def setup_entities():
    from entities import jet
    starting_position = np.array([-10000.0, 10000.0, 0.0])
    starting_velocity = np.array([400.0, 0.0, 0.0])
    Sukoi = presets.create_Sukoi57(starting_position, starting_velocity, manual_control=True)
    Patriot1 = presets.create_PAC3(
        starting_position=np.array([9000.0, 10.0, 5000.0]),
        starting_velocity=np.array([0.0, 10.0, 0.0]),
        target_entity=Sukoi
    )
    Patriot2 = presets.create_PAC3(
        starting_position=np.array([9000.0, 10.0, 0.0]),
        starting_velocity=np.array([0.0, 10.0, 0.0]),
        target_entity=Sukoi
    )
    Patriot3 = presets.create_PAC3(
        starting_position=np.array([9000.0, 10.0, -5000.0]),
        starting_velocity=np.array([0.0, 10.0, 0.0]),
        target_entity=Sukoi
    )
    return [Sukoi, Patriot1, Patriot2, Patriot3]

def run(epochs: int = 1, sprint: bool = False):
    epoch = 0
    while epochs <= 0 or epoch < epochs:
        entities = setup_entities()
        run_epoch(entities, SIMULATION_DURATION, SIMULATION_TICKRATE, SIMULATION_SPEED, sprint)
        clean_viz()
        epoch += 1

def run_epoch(entities, duration_s: float, tick_rate: float, sim_speed: float, sprint: bool):
    dt_sim = 1.0 / tick_rate
    dt_real_target = dt_sim / sim_speed
    steps = int(duration_s * tick_rate)

    next_tick = time.perf_counter()
    for _ in range(steps):
        for ent in entities:
            ent.tick()
        update_instances(entities)

        if not sprint:
            next_tick += dt_real_target
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
