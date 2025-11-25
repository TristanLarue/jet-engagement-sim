import numpy as np
from viz import update_instance
from config import TICK_RATE, setup_entities, SIMULATION_DURATION
import time
import physics

def initialize_simulation():
    tick_total = 0
    # Setup entities from config
    entities = setup_entities()
    while tick_total < TICK_RATE * SIMULATION_DURATION:
        try:
            render_tick(tick_total, entities)
            tick_total += 1
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            break

def check_events(tick_count):
    # Added in for spontaneous event testing
    if tick_count == 180:
        print("Test event triggered at tick 180")

def render_tick(tick_count: int, entities: list):
    check_events(tick_count)
    tick_start_time = time.perf_counter()
    #Cycle entities one by one
    for ent in entities:
        #=====APPLY ALL PHYSICS FOR THAT TICK=====
        # 1. Decision-Making
        ent.think(entities) #entities covers all simulation data to analyze & predict
        # 2. Apply angular velocity
        physics.apply_angular_velocity(ent)
        physics.apply_restoring_torque(ent)
        # 3. Linear Velocity
        physics.apply_thrust(ent)
        physics.apply_gravity(ent)
        physics.apply_lift_with_g_limit(ent)
        #Drag at the end to assure stability at cruising speed
        physics.apply_drag(ent) #Same system as minecraft physics
        # 4. Apply linear velocity changes
        physics.apply_velocity(ent)
        update_instance(ent)
    
    # Balance FPS with the render speed
    if tick_start_time + (1/TICK_RATE) > time.perf_counter():
        time.sleep((tick_start_time + (1/TICK_RATE)) - time.perf_counter())
