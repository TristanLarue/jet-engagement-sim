import numpy as np
import vpython as vp

def create_Sukoi57(starting_position: np.ndarray=np.zeros(3), starting_velocity: np.ndarray=np.zeros(3), starting_orientation: np.ndarray=np.eye(3), manual_control: bool = False):
    from entities import jet
    return jet(
        starting_position=starting_position,
        starting_velocity=starting_velocity,
        starting_orientation=starting_orientation,
        manual_control=manual_control,
        mass=18000.0,
        wingspan=11.36,
        length=22.0,
        thrust_force=152000.0,
        reference_area=38.0,
        min_drag_coefficient=0.02,
        max_drag_coefficient=0.25,
        max_lift_coefficient=1.5,
        moment_of_inertia_roll=5000.0,
        moment_of_inertia_pitch=15000.0,
        moment_of_inertia_yaw=20000.0,
        optimal_lift_aoa=15.0,
        viz_shape={"compound_shape":"jet",
                   "size":50,
                   "opacity":1.0,
                   "make_trail":True,
                   "trail_radius":25,
                   "color": vp.color.red,
                   "starting_position": starting_position}
        )