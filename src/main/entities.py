import numpy as np
from viz import create_viz_instance
from simulation import SIMULATION_TICKRATE
import physics


# A bit of object oriented programming to manage entities in a smart way
# Base entity class that can enact common behaviors for all simulated entities
class entity():
    def __init__(self, starting_position: np.ndarray, starting_velocity: np.ndarray, starting_orientation: np.ndarray, viz_shape: dict):
        # Below are the physics attributes
        self.position = starting_position #Position
        self.velocity = starting_velocity #Velocity
        self.orientation = starting_orientation # Orientation matrix
        self.omega = np.array([0.0, 0.0, 0.0]) # [X, Y, Z] body axes (deg/s)
        # Create visualization instance automatically
        self.shape = viz_shape.get("compound_shape")
        self.viz_instance = create_viz_instance(viz_shape, self)


class jet(entity):
    def __init__(self, starting_position: np.ndarray, starting_velocity: np.ndarray, starting_orientation: np.ndarray, manual_control: bool, mass: float, wingspan: float, length: float, thrust_force: float, reference_area: float, min_drag_coefficient: float, max_drag_coefficient: float, max_lift_coefficient: float, moment_of_inertia_roll: float, moment_of_inertia_pitch: float, moment_of_inertia_yaw: float, optimal_lift_aoa: float, viz_shape: dict):
        # Below are the control attributes
        self.manual_control = manual_control
        self.control_inputs = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.throttle = 1.0
        # Entity initialization
        super().__init__(
            starting_position,
            starting_velocity,
            starting_orientation,
            viz_shape,
            )
        # Below are the config attributes (jet dependencies)
        self.mass = mass # Technically excluding control surfaces and dynamic weight changes (fuel,weapons,etc.)
        self.wingspan = wingspan
        self.length = length
        self.thrust_force = thrust_force
        self.reference_area = reference_area
        self.min_drag_coefficient = min_drag_coefficient
        self.max_drag_coefficient = max_drag_coefficient
        self.max_lift_coefficient = max_lift_coefficient
        self.moment_of_inertia = np.array([moment_of_inertia_roll, moment_of_inertia_pitch, moment_of_inertia_yaw])
        self.optimal_lift_aoa = optimal_lift_aoa
    
    def tick(self):
        air_density = physics.get_air_density(self.position[1])
        aoa = physics.get_angle_of_attack(self.velocity, self.orientation)
        cd = physics.get_drag_coefficient(aoa, self.min_drag_coefficient, self.max_drag_coefficient)
        cl = physics.get_lift_coefficient(aoa, self.max_lift_coefficient, self.optimal_lift_aoa)
        # Collect all force/torque pairs
        forces = [
            [physics.get_gravity_force(self.mass), np.array([0.0, 0.0, 0.0])],
            [physics.get_thrust_force(self.orientation, self.throttle, self.thrust_force, self.length), np.array([-0.5 * self.length, 0.0, 0.0])],
            [physics.get_drag_force(self.velocity, air_density, self.reference_area, cd), np.array([-0.2 * self.length, 0.0, 0.0])],
            [physics.get_fuselage_lift_force(self.velocity, self.reference_area, self.orientation, air_density, cl), np.array([-0.2 * self.length, 0.0, 0.0])],
            #physics.get_sideforce_force(),
            [physics.get_elevator_force(self.velocity, air_density, self.reference_area, self.max_lift_coefficient, self.orientation, self.control_inputs["pitch"], self.optimal_lift_aoa, 0.30, 1.40), np.array([-0.45 * self.length, 0.0, 0.0])],
            [physics.get_aileron_force(self.velocity, air_density, self.reference_area, self.max_lift_coefficient, self.orientation, self.control_inputs["roll"], self.optimal_lift_aoa, 0.04, 0.6)[0], np.array([-0.05 * self.length, 0.0, 0.50 * self.wingspan])],
            [physics.get_aileron_force(self.velocity, air_density, self.reference_area, self.max_lift_coefficient, self.orientation, self.control_inputs["roll"], self.optimal_lift_aoa, 0.04, 0.6)[1], np.array([-0.05 * self.length, 0.0, -0.50 * self.wingspan])],
            [physics.get_rudder_force(self.velocity, air_density, self.reference_area, self.max_lift_coefficient, self.orientation, self.control_inputs["yaw"], self.optimal_lift_aoa, 0.03, 0.30), np.array([-0.45 * self.length, 0.0, 0.0])],
        ]


        # Filter out None values
        total_force = np.array([0.0, 0.0, 0.0])
        total_torque = np.array([0.0, 0.0, 0.0])
        for force in forces:
            total_force += force[0]
            total_torque += physics.get_omega(force[0], self.orientation, force[1], self.moment_of_inertia)
        
        # Update state
        self.omega += total_torque / SIMULATION_TICKRATE
        self.omega *= 0.95  # GOOD ENOUGH DAMPING FOR RELEASE
        self.orientation = physics.integrate_orientation(self.orientation, self.omega, 1.0 / SIMULATION_TICKRATE)
        acceleration = total_force / self.mass
        self.velocity = physics.integrate_velocity(self.velocity, acceleration, 1.0 / SIMULATION_TICKRATE)
        self.position = physics.integrate_position(self.position, self.velocity, 1.0 / SIMULATION_TICKRATE)