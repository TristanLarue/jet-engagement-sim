import numpy as np
from . import ent_presets
import random

def create_phase1_scenario():
    jet = ent_presets.create_Sukoi57(
        np.array([0.0, 1000.0, 0.0], dtype=float),
        np.array([400.0, 0.0, 0.0], dtype=float),
    )
    return [jet]

def create_phase2_scenario():
    jet = ent_presets.create_Sukoi57(
        np.array([-20000.0, 100.0, 0.0], dtype=float),
        np.array([400.0, 0.0, 0.0], dtype=float),
    )
    m1 = ent_presets.create_PAC3(
        starting_position=np.array([random.uniform(10000.0, 20000.0), 10.0, random.uniform(-20000.0, 20000.0)], dtype=float),
        starting_velocity=np.array([0.0, 10.0, 0.0], dtype=float),
        target_entity=jet,
    )
    return [jet, m1]


def create_base_scenario():
    jet = ent_presets.create_Sukoi57(
        np.array([-20000.0, 9000.0, 0.0], dtype=float),
        np.array([400.0, 0.0, 0.0], dtype=float),
    )
    m1 = ent_presets.create_PAC3(
        starting_position=np.array([random.uniform(10000.0, 20000.0), 10.0, random.uniform(-20000.0, 20000.0)], dtype=float),
        starting_velocity=np.array([0.0, 10.0, 0.0], dtype=float),
        target_entity=jet,
    )
    m2 = ent_presets.create_PAC3(
        starting_position=np.array([random.uniform(10000.0, 20000.0), 10.0, random.uniform(-20000.0, 20000.0)], dtype=float),
        starting_velocity=np.array([0.0, 10.0, 0.0], dtype=float),
        target_entity=jet,
    )
    m3 = ent_presets.create_PAC3(
        starting_position=np.array([random.uniform(10000.0, 20000.0), 10.0, random.uniform(-20000.0, 20000.0)], dtype=float),
        starting_velocity=np.array([0.0, 10.0, 0.0], dtype=float),
        target_entity=jet,
    )
    return [jet, m1, m2, m3]
