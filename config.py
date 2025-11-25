import numpy as np
SIMULATION_DURATION = 30
TICK_RATE = 120  # ticks per second
SIMULATION_SPEED = 1  # real-time speed multiplier
AIR_DENSITY = 1.225  # kg/m^3 at sea level
BOX_SIZE = np.array([20000.0, 5000.0, 20000.0])  # 20km x 5km x 20km

def setup_entities():
    from entities import missile, jet
    
    # Create jet at far side of box (X=9000, Y=2000, Z=0)
    target_jet = jet(
        position=(9000.0, 2000.0, 0.0),
        velocity=(-400.0, 0.0, 0.0),
        size=200.0,
        opacity=1.0,
        make_trail=True,
        trail_radius=100.0,
        mass=23000.0,
        min_drag_coefficient=0.04,
        max_drag_coefficient=0.80,
        reference_area=62.0,
        thrust_force=250000.0,
        max_lift_coefficient=1.50
    )
    target_jet.yaw = 180.0  # Facing towards negative X direction
    
    # Create 3 missiles at near side of box (X=-9000), equally spaced along Z
    missile_1 = missile(
        position=(-9000.0, 0.0, 0.0),
        velocity=(0.0, 1000.0, 0.0),
        size=100.0,
        opacity=1.0,
        make_trail=True,
        trail_radius=50.0,
        mass=900.0,
        min_drag_coefficient=0.40,
        max_drag_coefficient=8.0,
        reference_area=0.13,
        burn_time=60.0,
        target_entity=target_jet,
        thrust_force=100000.0,
        max_lift_coefficient=0.3
    )
    missile_1.yaw = 90.0  # Facing towards positive Y direction
    missile_2 = missile(
        position=(-9000.0, 0.0, 5000.0),
        velocity=(0.0, 1000.0, 0.0),
        size=100.0,
        opacity=1.0,
        make_trail=True,
        trail_radius=50.0,
        mass=900.0,
        min_drag_coefficient=0.40,
        max_drag_coefficient=8.0,
        reference_area=0.13,
        burn_time=60.0,
        target_entity=target_jet,
        thrust_force=100000.0,
        max_lift_coefficient=0.3
    )
    missile_2.yaw = 90.0  # Facing towards positive Y direction
    missile_3 = missile(
        position=(-9000.0, 0.0, -5000.0),
        velocity=(0.0, 1000.0, 0.0),
        size=100.0,
        opacity=1.0,
        make_trail=True,
        trail_radius=50.0,
        mass=900.0,
        min_drag_coefficient=0.40,
        max_drag_coefficient=8.0,
        reference_area=0.13,
        burn_time=60.0,
        target_entity=target_jet,
        thrust_force=100000.0,
        max_lift_coefficient=0.3
    )
    missile_3.yaw = 90.0  # Facing towards positive Y direction
    return [missile_1, missile_2, missile_3, target_jet]