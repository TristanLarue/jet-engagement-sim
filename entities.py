import numpy as np
from viz import create_instance
from guidance import jet_float_think,missile_direct_attack_think
import time
import physics

# A bit of object oriented programming to manage entities in a smart way
# Base entity class that can enact common behaviors for all simulated entities
class entity():
    def __init__(self, shape: str, position: tuple, velocity: tuple, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, thrust_force: float, max_lift_coefficient: float = 0.0, throttle: float = 1.0):
        self.shape = shape
        self.p = np.array(position) #Position
        self.v = np.array(velocity) #Velocity
        self.mass = mass #Mass in kg
        self.cp_dist = -0.5 # Distance of the center of pressure from the center of mass (USED FOR MOMENTS)
        self.min_drag_coefficient = min_drag_coefficient #Best-case Cd
        self.max_drag_coefficient = max_drag_coefficient #Worst-case Cd
        self.reference_area = reference_area #Reference area for aerodynamic calculations
        self.max_lift_coefficient = max_lift_coefficient #Max Cl at optimal AoA
        self.thrust_force = thrust_force #Max thrust force in Newtons
        self.viz_instance = None #Visualization instance
        # Not using *heading* vector since it can be derived from pitch/yaw/roll
        self.pitch = 0.0 #Pitch angle (deg) 
        self.pitch_v = 0.0 #(deg/s)
        self.yaw = 0.0 #Yaw angle (deg)
        self.yaw_v = 0.0 #(deg/s)
        self.roll = 0.0 #Roll angle (deg)
        self.roll_v = 0.0 #(deg/s)
        self.max_roll_velocity = 360.0 #(deg/s) or 60rpm 
        #MAX PITCH VELOCITY WILL BE DYNAMIC BASED ON SPEED
        # Below are the control inputs from think()
        self.pitch_input = 0.0 #Pitch control input (-1.0 to 1.0)
        #self.yaw_input = 0.0 No yaw inputs allowed due to its ineffectiveness in my simulation for now
        self.roll_input = 0.0 #Roll control input (-1.0 to 1.0)
        self.throttle = throttle #Throttle setting (0.0 to 1.0)
        self.viz_instanciate()
        
    def viz_instanciate(self):
        if self.viz_instance is None:
            self.viz_instance = create_instance(self.shape, tuple(self.p), size=100, opacity=1.0, make_trail=True, trail_radius=30)

    def think(self, entities):
        pass
    
    def apply_thrust(self):
        R = physics.get_rotation_matrix(self.roll, self.pitch, self.yaw)
        self.v += physics.get_thrust_acc(self.throttle, self.thrust_force, self.mass, R)
    
    def apply_drag(self):
        R = physics.get_rotation_matrix(self.roll, self.pitch, self.yaw)
        force_world = physics.get_drag_force(self.v, self.reference_area, self.min_drag_coefficient, self.max_drag_coefficient, R)
        self.v += physics.get_drag_acc(self.v, self.mass, self.reference_area, self.min_drag_coefficient, self.max_drag_coefficient, R)
        pitch_offset, yaw_offset = physics.get_moment(force_world, R, self.cp_dist, self.mass)
        self.pitch_v += pitch_offset
        self.yaw_v += yaw_offset
    
    def apply_lift(self):
        R = physics.get_rotation_matrix(self.roll, self.pitch, self.yaw)
        self.v += physics.get_lift_acc(self.v, self.mass, self.reference_area, self.max_lift_coefficient, R)
        force_world = physics.get_lift_force(self.v, self.reference_area, self.max_lift_coefficient, R)
        pitch_offset, yaw_offset = physics.get_moment(force_world, R, self.cp_dist, self.mass)
        self.pitch_v += pitch_offset
        self.yaw_v += yaw_offset
    
    def apply_gravity(self):
        self.v += physics.get_gravity_acc()
    
    def apply_translation_velocity(self):
        self.p = physics.get_next_pos(self.p, self.v)

    def apply_rotation_inputs(self):
        # Apply pitch input
        max_pitch_v = self.max_roll_velocity * (np.linalg.norm(self.v) / 300.0)  # Scale max pitch velocity with speed
        self.pitch_v += float(self.pitch_input * max_pitch_v / physics.TICK_RATE)
        # Apply roll input
        self.roll_v += float(self.roll_input * self.max_roll_velocity / physics.TICK_RATE)

    def apply_rotation_velocity(self):
        self.roll,self.pitch,self.yaw = physics.get_next_rotation(self.roll, self.pitch, self.yaw, self.roll_v, self.pitch_v, self.yaw_v)
    
    def apply_rotation_damping(self):
        # Simple rotational damping to prevent infinite spin
        damping_factor = 0.98
        self.pitch_v *= damping_factor
        self.yaw_v *= damping_factor
        self.roll_v *= damping_factor
        

# After careful consideration, 6DOF is worth it for missiles too
# 3DOF does simplify it, but having orientation aswell isnt much more complex and will allow for more realistic behaviors
class missile(entity):
    def __init__(self, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, burn_time: float, target_entity: entity, thrust_force: float, max_lift_coefficient: float = 0.0):
        super().__init__("missile", position, velocity, mass, min_drag_coefficient, max_drag_coefficient, reference_area, thrust_force, max_lift_coefficient, throttle=1.0)
        self.throttle = 1.0
        self.burn_time = burn_time
        self.launch_time = time.perf_counter()
        self.target = target_entity

    def think(self, entities):
        if time.perf_counter() - self.launch_time > self.burn_time:
            self.throttle = 0.0
        missile_direct_attack_think(entities, self)

class jet(entity):
    def __init__(self, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, thrust_force: float, max_lift_coefficient: float):
        super().__init__("jet", position, velocity, mass, min_drag_coefficient, max_drag_coefficient, reference_area, thrust_force, max_lift_coefficient, throttle=1.0)

    def think(self, entities):
        jet_float_think(entities, self)
