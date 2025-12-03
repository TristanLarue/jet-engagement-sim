import numpy as np
import random
SIMULATION_DURATION = 120
TICK_RATE = 24  # ticks per second
SIMULATION_SPEED = 1  # real-time speed multiplier
AIR_DENSITY = 1.225  # kg/m^3 at sea level
BOX_SIZE = np.array([20000.0, 15000.0, 20000.0])  # 10km x 15km x 10km

def setup_entities():
    from entities import missile, jet
    
    # Create jet at far side of box (X=9000, Y=2000, Z=0)
    target_jet = jet(
        position=(-4000.0, 10000.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        size=500.0,
        opacity=1.0,
        make_trail=True,
        trail_radius=250.0,
        mass=23000.0,  # Su-27
        length=22.0,  # Su-27 length in meters
        cp_diff=(-1.0, 0.0, 0.0),
        min_drag_coefficient=0.02,
        max_drag_coefficient=0.40,
        reference_area=62.0,
        thrust_force=245000.0,
        max_lift_coefficient=1.5,
        manual_control=True
    )  # Facing towards negative X direction
    
    # Create 3 missiles at near side of box (X=-9000), equally spaced along Z
    patriot_missile = missile(
        position=(-4000.0+random.randint(-1000,1000), 0000.0, random.randint(-1000,1000)),#-4000.0
        velocity=(0.0, 100.0, 0.0),
        size=300.0,
        opacity=1.0,
        make_trail=True,
        trail_radius=150.0,
        mass=900.0,  # Patriot PAC-2
        length=5.0,  # Patriot PAC-2 length in meters
        cp_diff=(-1.0, 0.0, 0.0),
        min_drag_coefficient=0.08,
        max_drag_coefficient=1.60,
        reference_area=0.13,
        burn_time=60.0,
        target_entity=target_jet,
        thrust_force=130000.0,
        max_lift_coefficient=0.6
    )
    patriot_missile.pitch = 90.0  # Facing towards positive Y direction

    return [patriot_missile, target_jet]