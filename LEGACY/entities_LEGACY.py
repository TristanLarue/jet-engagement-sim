import numpy as np
from viz import create_instance
import guidance
import time
import physics

# A bit of object oriented programming to manage entities in a smart way
# Base entity class that can enact common behaviors for all simulated entities
class entity():
    def __init__(self, shape: str, position: tuple, velocity: tuple, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, thrust_force: float, max_lift_coefficient: float = 0.0, throttle: float = 1.0, cp_diff: tuple = (-0.5, 0.0, 0.0), length: float = 10.0, moment_of_inertia: float = None, manual_control: bool = False):
        self.shape = shape
        self.p = np.array(position) #Position
        self.v = np.array(velocity) #Velocity
        self.mass = mass #Mass in kg
        self.length = length # Length of the body (for moment of inertia: I = 1/12 * m * L²)
        self.moment_of_inertia = moment_of_inertia if moment_of_inertia is not None else (1.0 / 12.0) * mass * (length ** 2) # Moment of inertia [kg⋅m²]
        self.cp_diff = np.array(cp_diff) # 3D vector from CM to center of pressure in body space (USED FOR MOMENTS)
        self.min_drag_coefficient = min_drag_coefficient #Best-case Cd
        self.max_drag_coefficient = max_drag_coefficient #Worst-case Cd
        self.reference_area = reference_area #Reference area for aerodynamic calculations
        self.max_lift_coefficient = max_lift_coefficient #Max Cl at optimal AoA
        self.thrust_force = thrust_force #Max thrust force in Newtons
        self.viz_instance = None #Visualization instance
        # Orientation as 3x3 rotation matrix (body -> world)
        self.orientation = np.eye(3) # Start with identity (no rotation)
        # Body angular velocity [roll_rate, pitch_rate, yaw_rate] in body frame
        self.omega_body = np.array([0.0, 0.0, 0.0]) # [X, Y, Z] body axes (deg/s)
        self.max_roll_velocity = 360.0 #(deg/s) or 60rpm
        self.manual_control = manual_control #If true, entity is controlled manually (no AI)
        # Below are the control inputs from think()
        self.pitch_input = 0.0 #Pitch control input (-1.0 to 1.0)
        self.roll_input = 0.0 #Roll control input (-1.0 to 1.0)
        self.throttle = throttle #Throttle setting (0.0 to 1.0)
        self.viz_instanciate()
        
    def viz_instanciate(self):
        if self.viz_instance is None:
            self.viz_instance = create_instance(self.shape, tuple(self.p), size=100, opacity=1.0, make_trail=True, trail_radius=30)

    def think(self, entities):
        pass
    
    def apply_thrust(self):
        self.v += physics.get_thrust_acc(self.throttle, self.thrust_force, self.mass, self.orientation)
    
    def apply_drag(self):
        force_world = physics.get_drag_force(self.v, self.reference_area, self.min_drag_coefficient, self.max_drag_coefficient, self.orientation)
        self.v += physics.get_drag_acc(self.v, self.mass, self.reference_area, self.min_drag_coefficient, self.max_drag_coefficient, self.orientation)
        pitch_offset, yaw_offset = physics.get_force_torque(force_world, self.orientation, self.cp_diff, self.mass, self.length)
        # Reduce angular force magnitude and apply proper time scaling
        angular_force_scale = 0.1  # Reduce by 90% to prevent oscillations
        self.omega_body[2] += (pitch_offset * angular_force_scale) / physics.TICK_RATE  # Z-axis = pitch
        self.omega_body[1] += (yaw_offset * angular_force_scale) / physics.TICK_RATE    # Y-axis = yaw
    
    def apply_lift(self):
        self.v += physics.get_lift_acc(self.v, self.mass, self.reference_area, self.max_lift_coefficient, self.orientation)
        force_world = physics.get_lift_force(self.v, self.reference_area, self.max_lift_coefficient, self.orientation)
        pitch_offset, yaw_offset = physics.get_force_torque(force_world, self.orientation, self.cp_diff, self.mass, self.length)
        # Reduce angular force magnitude and apply proper time scaling
        angular_force_scale = 0.1  # Reduce by 90% to prevent oscillations
        self.omega_body[2] += (pitch_offset * angular_force_scale) / physics.TICK_RATE  # Z-axis = pitch
        self.omega_body[1] += (yaw_offset * angular_force_scale) / physics.TICK_RATE    # Y-axis = yaw
    
    def apply_gravity(self):
        self.v += physics.get_gravity_acc()
    
    def apply_translation_velocity(self):
        self.p = physics.get_next_pos(self.p, self.v)

    def apply_rotation_inputs(self):
        if np.linalg.norm(self.v) < 10.0: return
        self.apply_pitch_control()
        self.apply_roll_control()
    
    def apply_pitch_control(self):
        if abs(self.pitch_input) < 0.01: return
        v_mag = np.linalg.norm(self.v)
        # Aerodynamic effectiveness scales with dynamic pressure
        dynamic_pressure = 0.5 * physics.AIR_DENSITY * v_mag**2
        aero_force = dynamic_pressure * 1.0 * abs(self.pitch_input)  # Increased scale factor for control authority
        # Limit by actuator power and structural constraints
        max_actuator_force = 20000.0  # Increased F-22 elevator authority for better control
        control_force = min(aero_force, max_actuator_force) * physics.get_control_surface_weight(self.v, self.orientation)
        
        tail_pos = np.array([-self.length * 0.4, 0.0, 0.0])
        elevator_force = self.orientation @ np.array([0.0, self.pitch_input * control_force, 0.0])
        pitch_torque, _ = physics.get_force_torque(elevator_force, self.orientation, tail_pos, self.mass, self.length)
        self.omega_body[2] += pitch_torque / physics.TICK_RATE
    
    def apply_roll_control(self):
        if abs(self.roll_input) < 0.01: return
        v_mag = np.linalg.norm(self.v)
        # Aerodynamic effectiveness scales with dynamic pressure
        dynamic_pressure = 0.5 * physics.AIR_DENSITY * v_mag**2
        aero_force = dynamic_pressure * 0.5 * abs(self.roll_input)  # Increased aileron authority
        # Limit by actuator power and structural constraints
        max_actuator_force = 25000.0  # Increased F-22 aileron authority for better control
        aileron_force = min(aero_force, max_actuator_force) * physics.get_control_surface_weight(self.v, self.orientation)
        
        wing_span = self.length * 0.4
        roll_moment = wing_span * self.roll_input * aileron_force
        self.omega_body[0] += np.degrees(roll_moment / self.moment_of_inertia) / physics.TICK_RATE

    def apply_rotation_velocity(self):
        # Convert body angular velocity to small rotation matrix
        delta_R = physics.get_delta_rotation_matrix(self.omega_body, physics.TICK_RATE)
        # Apply rotation: R_new = R_old @ delta_R (body-local rotation)
        self.orientation = self.orientation @ delta_R
        # Orthonormalize to prevent drift
        self.orientation = physics.orthonormalize_matrix(self.orientation)
    
    def apply_rotation_damping(self):
        # Enhanced rotational damping to prevent oscillations
        base_damping = 0.85  # Stronger base damping (was 0.95)
        
        # Add velocity-based damping - higher damping at higher angular velocities
        velocity_damping = 1.0 - (np.linalg.norm(self.omega_body) * 0.001)
        velocity_damping = max(velocity_damping, 0.5)  # Don't over-damp
        
        combined_damping = base_damping * velocity_damping
        self.omega_body *= combined_damping
        

# After careful consideration, 6DOF is worth it for missiles too
# 3DOF does simplify it, but having orientation aswell isnt much more complex and will allow for more realistic behaviors
class missile(entity):
    def __init__(self, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, burn_time: float, target_entity: entity, thrust_force: float, max_lift_coefficient: float = 0.0, cp_diff: tuple = (-0.5, 0.0, 0.0), length: float = 5.0, moment_of_inertia: float = None):
        super().__init__("missile", position, velocity, mass, min_drag_coefficient, max_drag_coefficient, reference_area, thrust_force, max_lift_coefficient, throttle=1.0, cp_diff=cp_diff, length=length, moment_of_inertia=moment_of_inertia)
        self.throttle = 1.0
        self.burn_time = burn_time
        self.launch_time = time.perf_counter()
        self.target = target_entity

    def think(self, entities):
        if time.perf_counter() - self.launch_time > self.burn_time:
            self.throttle = 0.0
        guidance.missile_direct_attack_think(self, entities)

class jet(entity):
    def __init__(self, position: tuple, velocity: tuple, size: float, opacity: float, make_trail: bool, trail_radius: float, mass: float, min_drag_coefficient: float, max_drag_coefficient: float, reference_area: float, thrust_force: float, max_lift_coefficient: float, cp_diff: tuple = (-0.5, 0.0, 0.0), length: float = 22.0, moment_of_inertia: float = None, manual_control: bool = False):
        super().__init__("jet", position, velocity, mass, min_drag_coefficient, max_drag_coefficient, reference_area, thrust_force, max_lift_coefficient, throttle=1.0, cp_diff=cp_diff, length=length, moment_of_inertia=moment_of_inertia, manual_control=manual_control)
        self.ai_reward = 0.0
    def think(self, entities):
        guidance.jet_manual_control(self, entities)