import vpython as vp
from typing import Tuple, Optional
import numpy as np
import physics

# Global storage for UI panels and velocity arrows
ui_panels = {}
velocity_arrows = {}
forward_arrows = {}  # Store forward direction arrows
joystick_canvases = {}  # Store canvas, background, orientation circle, velocity circle for each entity

def create_instance(shape: str = "missile", position: Tuple[float, float, float] = (0.0, 0.0, 0.0), size: float = 100, opacity: float = 1.0, make_trail: bool = False, trail_radius: float = 0.05):
    pos = vp.vector(*position)
    
    # Create invisible placeholder object for ID tracking
    obj = vp.sphere(pos=pos, radius=0.1, visible=False, make_trail=make_trail, trail_radius=trail_radius)
    
    # Create velocity arrow (yellow)
    velocity_arrow = vp.arrow(pos=pos, axis=vp.vector(1, 0, 0), color=vp.color.yellow, shaftwidth=size*0.2)
    velocity_arrows[id(obj)] = velocity_arrow
    
    # Create forward direction arrow (red)
    forward_arrow = vp.arrow(pos=pos, axis=vp.vector(1, 0, 0), color=vp.color.red, shaftwidth=size*0.2)
    forward_arrows[id(obj)] = forward_arrow
    
    # Create text info and joystick side by side
    ui_label = vp.wtext(text="")
    
    # Create joystick panel
    joystick_canvas = vp.graph(width=200, height=200, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, 
                                xticks=False, yticks=False, background=vp.color.gray(0.9))
    
    # Draw square boundary
    boundary = vp.gcurve(color=vp.color.black, graph=joystick_canvas)
    boundary.plot(pos=[(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)])
    
    # Draw crosshair
    crosshair_h = vp.gcurve(color=vp.color.gray(0.7), graph=joystick_canvas)
    crosshair_h.plot(pos=[(-1, 0), (1, 0)])
    crosshair_v = vp.gcurve(color=vp.color.gray(0.7), graph=joystick_canvas)
    crosshair_v.plot(pos=[(0, -1), (0, 1)])
    
    # Create orientation circle (big, opaque)
    orientation_circle = vp.gdots(color=vp.color.blue, size=15, graph=joystick_canvas)
    orientation_circle.plot(pos=(0, 0))
    
    # Create velocity circle (small, semi-transparent)
    velocity_circle = vp.gdots(color=vp.color.red, size=8, graph=joystick_canvas)
    velocity_circle.plot(pos=(0, 0))
    
    ui_panels[id(obj)] = ui_label
    joystick_canvases[id(obj)] = {
        'canvas': joystick_canvas,
        'orientation': orientation_circle,
        'velocity': velocity_circle
    }
    
    return obj


def update_instance(entity):
    entity.viz_instance.pos = vp.vector(*tuple(entity.p))
    
    R = physics.get_rotation_matrix(entity.roll, entity.pitch, entity.yaw)
    
    # Body axes in world frame

    forward = physics.get_forward_dir(R)
    
    arrow_id = id(entity.viz_instance)
    arrow_length = 200.0  # Fixed arrow length for visibility
    
    # Update velocity arrow (yellow)
    if arrow_id in velocity_arrows:
        velocity_arrow = velocity_arrows[arrow_id]
        velocity_arrow.pos = vp.vector(*tuple(entity.p))
        
        velocity_magnitude = np.linalg.norm(entity.v)
        if velocity_magnitude > 0:
            velocity_normalized = entity.v / velocity_magnitude
            velocity_arrow.axis = vp.vector(*velocity_normalized) * arrow_length
        else:
            velocity_arrow.axis = vp.vector(1, 0, 0)
    
    # Update forward direction arrow (red)
    if arrow_id in forward_arrows:
        forward_arrow = forward_arrows[arrow_id]
        forward_arrow.pos = vp.vector(*tuple(entity.p))
        forward_arrow.axis = vp.vector(*forward) * arrow_length
    
    # Update UI panel
    panel_id = id(entity.viz_instance)
    if panel_id in ui_panels:
        ui_label = ui_panels[panel_id]
        
        # Check entity type and format appropriate info
        aoa = physics.get_aoa(entity.v, R)
        if hasattr(entity, 'target'):  # Missile
            distance_to_target = np.linalg.norm(entity.target.p - entity.p)
            speed = np.linalg.norm(entity.v)
            if speed > 0:
                interception_time = distance_to_target / speed
            else:
                interception_time = float('inf')
            ui_label.text = (
                f"<span style='font-size:18px'>"
                f"<b style='color:blue'>MISSILE</b><br>"
                f"<b>AoA: {aoa:.1f}°</b><br>"
                f"Distance: {distance_to_target:.1f} m<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Throttle: {entity.throttle*100:.0f}%<br>"
                f"ETA: {interception_time:.2f} s<br>"
                f"Pitch Rate: {entity.pitch_v:.1f}°/s<br>"
                f"Yaw Rate: {entity.yaw_v:.1f}°/s<br>"
                f"Roll Rate: {entity.roll_v:.1f}°/s<br>"
                f"</span>"
            )
        elif hasattr(entity, 'throttle'):  # Jet
            speed = np.linalg.norm(entity.v)
            ui_label.text = (
                f"<span style='font-size:18px'>"
                f"<b style='color:red'>JET</b><br>"
                f"<b>AoA: {aoa:.1f}°</b><br>"
                f"Pitch: {entity.pitch:.1f}°<br>"
                f"Roll: {entity.roll:.1f}°<br>"
                f"Yaw: {entity.yaw:.1f}°<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Throttle: {entity.throttle*100:.0f}%<br>"
                f"Pitch Rate: {entity.pitch_v:.1f}°/s<br>"
                f"Yaw Rate: {entity.yaw_v:.1f}°/s<br>"
                f"Roll Rate: {entity.roll_v:.1f}°/s<br>"
                f"</span>"
            )
    
    # Update joystick panel
    if panel_id in joystick_canvases:
        joystick = joystick_canvases[panel_id]
        # Map angles to -1 to 1 range (clamp at ±90 degrees for visualization)
        roll_normalized = np.clip(entity.roll / 90.0, -1.0, 1.0)  # Left-right (x-axis)
        pitch_normalized = np.clip(entity.pitch / 90.0, -1.0, 1.0)  # Up-down (y-axis)
        
        # Map angular velocities to -1 to 1 range (scale for visibility)
        # Using ±180°/s as max for visualization
        roll_vel_normalized = np.clip(entity.roll_v / 180.0, -1.0, 1.0)
        pitch_vel_normalized = np.clip(entity.pitch_v / 180.0, -1.0, 1.0)
        
        # Clear and update orientation circle (current angles)
        joystick['orientation'].data = []
        joystick['orientation'].plot(pos=(roll_normalized, pitch_normalized))
        
        # Clear and update velocity circle (angular velocities)
        joystick['velocity'].data = []
        joystick['velocity'].plot(pos=(roll_vel_normalized, pitch_vel_normalized))
    
    return



def initialize_viz():
    from config import BOX_SIZE
    scene = vp.canvas(width=1920, height=480, background=vp.color.white)
    ground = vp.box(pos=vp.vector(0, 0, 0), size=vp.vector(BOX_SIZE[0], 1, BOX_SIZE[2]), color=vp.color.green)
    half_x, height, half_z = BOX_SIZE[0]/2, BOX_SIZE[1], BOX_SIZE[2]/2
    # Bottom 4 edges
    vp.curve(pos=[vp.vector(-half_x, 0, -half_z), vp.vector(half_x, 0, -half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, 0, half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(half_x, 0, half_z), vp.vector(-half_x, 0, half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, 0, -half_z)], color=vp.color.black)
    # Top 4 edges
    vp.curve(pos=[vp.vector(-half_x, height, -half_z), vp.vector(half_x, height, -half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(half_x, height, -half_z), vp.vector(half_x, height, half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(half_x, height, half_z), vp.vector(-half_x, height, half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(-half_x, height, half_z), vp.vector(-half_x, height, -half_z)], color=vp.color.black)
    # Vertical 4 edges
    vp.curve(pos=[vp.vector(-half_x, 0, -half_z), vp.vector(-half_x, height, -half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(half_x, 0, -half_z), vp.vector(half_x, height, -half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(half_x, 0, half_z), vp.vector(half_x, height, half_z)], color=vp.color.black)
    vp.curve(pos=[vp.vector(-half_x, 0, half_z), vp.vector(-half_x, height, half_z)], color=vp.color.black)
    scene.center = vp.vector(0, height/2, 0)
    return scene


def cleanup_viz():
    try:
        # Delete all VPython objects and close the canvas
        for obj in vp.scene.objects:
            obj.visible = False
            del obj
        vp.scene.delete()
    except:
        pass
