import numpy as np
import vpython as vp

def create_Sukoi57(starting_position: np.ndarray=np.zeros(3), starting_velocity: np.ndarray=np.zeros(3), starting_orientation: np.ndarray=np.eye(3)):
    from src.main.entities import jet

    return jet(
        starting_position=starting_position,
        starting_velocity=starting_velocity,
        starting_orientation=starting_orientation,
        mass=26700.0,
        wingspan=14.1,
        length=20.1,
        thrust_force=284400.0,
        reference_area=78.8,
        min_drag_coefficient=0.02,
        max_drag_coefficient=0.80,
        max_lift_coefficient=1.9,
        moment_of_inertia_roll=4.89e5,
        moment_of_inertia_pitch=9.46e5,
        moment_of_inertia_yaw=1.34e6,
        optimal_lift_aoa=12.0,
        viz_shape={"compound_shape":"Su57",
                   "make_trail":True,
                   "trail_radius":10,
                   "color": vp.color.red,
                   "starting_position": starting_position}
        )


def create_PAC3(starting_position=np.zeros(3), starting_velocity=np.zeros(3), chase_strategy=None, target_entity=None):
    from src.main.entities import missile
    from src.main.guidance import missile_direct_attack_DEBUG
    return missile(
        starting_position=np.array(starting_position, dtype=float),
        starting_velocity=np.array(starting_velocity, dtype=float),
        starting_orientation=np.array([[0.0, -1.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 1.0]], dtype=float),
        mass=312.0,
        thrust_force=120000.0,
        max_g=35.0,
        reference_area=0.051,
        min_drag_coefficient=0.08,
        max_drag_coefficient=10.0, #1.2 in normal condition
        max_lift_coefficient=1.8,
        moment_of_inertia_roll=2.54,
        moment_of_inertia_pitch=704.0,
        moment_of_inertia_yaw=704.0,
        optimal_lift_aoa=10.0,
        length=5.2,
        target_entity=target_entity,
        chase_strategy=missile_direct_attack_DEBUG,
        viz_shape={
            "compound_shape":"Pac3",
            "make_trail":True,
            "trail_radius":5.0,
            "color":vp.color.blue,
            "starting_position":np.array(starting_position)}
    )