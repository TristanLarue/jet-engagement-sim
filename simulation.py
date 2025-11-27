import numpy as np
from viz import update_instance,cleanup_viz
from config import TICK_RATE, setup_entities, SIMULATION_DURATION, SIMULATION_SPEED
from deeplearning import initialize_deeplearning, cleanup_deeplearning
import time
import physics

def simulate_epochs(epochs,sprint):
    for epoch in range(epochs) if epochs > 0 else iter(int, 1):
        print(f"===| Starting epoch {epoch + 1} |===")
        #initialize_deeplearning()
        initialize_simulation(sprint)
        #cleanup_deeplearning()
        cleanup_viz()
        print(f"===| Epoch {epoch + 1} completed |===")

def initialize_simulation(sprint):
    tick_total = 0
    # Setup entities from config
    entities = setup_entities()
    while tick_total < TICK_RATE / SIMULATION_SPEED * SIMULATION_DURATION:
        tick_start_time = time.perf_counter()
        try:
            render_tick(tick_total, entities)
            tick_total += 1
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            break
        # Balance FPS with the render speed
        if not sprint:
            if tick_start_time + (SIMULATION_SPEED/TICK_RATE) > time.perf_counter():
                time.sleep((tick_start_time + (SIMULATION_SPEED/TICK_RATE)) - time.perf_counter())
    
def check_events(tick_count):
    # Added in for spontaneous event testing
    if tick_count == 180:
        print("Test event triggered at tick 180")

def render_tick(tick_count: int, entities: list):
    check_events(tick_count)
    #Cycle entities one by one
    for ent in entities:
        if ent.shape == "missile":
            continue
        #=====APPLY ALL PHYSICS FOR THAT TICK=====
        # 1. Decision-Making
        ent.think(entities) #entities covers all simulation data to analyze & predict
        # 3. Apply angular velocity changes
        #Could apply all at once or one by one, no big difference for now
        ent.apply_thrust()
        ent.apply_drag()
        ent.apply_lift()
        ent.apply_gravity()
        # 5. Apply linear velocity changes
        ent.apply_rotation_inputs()
        ent.apply_rotation_damping()
        ent.apply_translation_velocity()
        ent.apply_rotation_velocity()
        update_instance(ent)
    
    
