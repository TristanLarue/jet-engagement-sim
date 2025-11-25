import numpy as np
from viz import create_instance
from guidance import jet_ai_think,missile_direct_attack_think
import time

class entity():
    def __init__(self, shape: str, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, thrust: float, max_lift_coefficient: float = 0.0, throttle: float = 1.0, g_limit: float = 9.0, max_angular_rate: float = 180.0):
        self.viz_instance = create_instance(shape, position, size, opacity, make_trail, trail_radius)
        self.shape = shape
        self.p = np.array(position) #Position
        self.v = np.array(velocity) #Velocity
        self.thrust = thrust #Max thrust acceleration
        self.throttle = throttle #Throttle setting (0.0 to 1.0)
        self.mass = mass #Mass in kg
        self.min_drag_coefficient = min_drag_coefficient #Best-case Cd
        self.max_drag_coefficient = max_drag_coefficient #Worst-case Cd
        self.reference_area = reference_area #Reference area for aerodynamic calculations
        self.max_lift_coefficient = max_lift_coefficient #Max Cl at optimal AoA
        self.g_limit = g_limit #Maximum g-force limit
        self.max_angular_rate = max_angular_rate #Maximum rotation rate (deg/s)
        self.pitch = 0.0 #Pitch angle (deg)
        self.yaw = 0.0 #Yaw angle (deg)
        self.roll = 0.0 #Roll angle (deg)
        self.pitch_input = 0.0 #Pitch control input (-1.0 to 1.0)
        self.yaw_input = 0.0 #Yaw control input (-1.0 to 1.0)
        self.roll_input = 0.0 #Roll control input (-1.0 to 1.0)
        
        # Snap orientation to initial velocity
        from physics import snap_orientation_to_velocity
        snap_orientation_to_velocity(self)

    def think(self, entities):
        pass

class missile(entity):
    def __init__(self, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, burn_time: float, target_entity: entity, thrust: float, max_lift_coefficient: float = 0.0, g_limit: float = 20.0, max_angular_rate: float = 720.0):
        super().__init__("missile", position, velocity, size, opacity, make_trail, trail_radius, mass, min_drag_coefficient, max_drag_coefficient, reference_area, thrust, max_lift_coefficient, throttle=1.0, g_limit=g_limit, max_angular_rate=max_angular_rate)
        self.throttle = 1.0
        self.burn_time = burn_time
        self.launch_time = time.perf_counter()
        self.target = target_entity

    def think(self, entities):
        if time.perf_counter() - self.launch_time > self.burn_time:
            self.throttle = 0.0
        missile_direct_attack_think(entities, self)

class jet(entity):
    def __init__(self, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, thrust: float, max_lift_coefficient: float, g_limit: float = 9.0, max_angular_rate: float = 360.0):
        super().__init__("jet", position, velocity, size, opacity, make_trail, trail_radius, mass, min_drag_coefficient, max_drag_coefficient, reference_area, thrust, max_lift_coefficient, throttle=1.0, g_limit=g_limit, max_angular_rate=max_angular_rate)

    def think(self, entities):
        pass
