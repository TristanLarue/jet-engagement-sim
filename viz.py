import vpython as vp
from typing import Tuple, Optional
import numpy as np

# Global storage for UI panels and velocity arrows
ui_panels = {}
velocity_arrows = {}

def create_instance(shape: str = "missile", position: Tuple[float, float, float] = (0.0, 0.0, 0.0), size: float = 100, opacity: float = 1.0, make_trail: bool = False, trail_radius: float = 0.05):
    pos = vp.vector(*position)
    
    # Create shape based on type
    if shape == "jet":
        # Jet as a box (flat rectangle) - length is forward, height is up, width is wingspan
        obj = vp.box(pos=pos, length=size*2, height=size*0.2, width=size*1.5, opacity=opacity, color=vp.color.red, make_trail=make_trail)
    else:
        # Missile as a cylinder
        obj = vp.cylinder(pos=pos, axis=vp.vector(size, 0, 0), radius=size*0.3, opacity=opacity, color=vp.color.blue, make_trail=make_trail)
    
    obj.visible = True
    
    # Create velocity arrow
    arrow = vp.arrow(pos=pos, axis=vp.vector(1, 0, 0), color=vp.color.yellow, shaftwidth=size*0.2)
    velocity_arrows[id(obj)] = arrow
    
    # Create 2D UI panel for this instance without background box
    ui_label = vp.wtext(text="")
    ui_panels[id(obj)] = ui_label
    
    return obj


def update_instance(entity):
    entity.viz_instance.pos = vp.vector(*tuple(entity.p))
    
    # Update orientation using pitch, yaw, roll
    pitch_rad = np.radians(entity.pitch)
    yaw_rad = np.radians(entity.yaw)
    roll_rad = np.radians(entity.roll)
    
    # Calculate rotation matrix to get axis direction
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    
    # Forward direction (X axis after rotation)
    forward_x = cy * cp
    forward_y = sy * cp
    forward_z = -sp
    
    # Up direction (Y axis after rotation)
    up_x = cy * sp * sr - sy * cr
    up_y = sy * sp * sr + cy * cr
    up_z = cp * sr
    
    # Set axis and up for the shape
    if hasattr(entity, 'target'):  # Missile (cylinder)
        entity.viz_instance.axis = vp.vector(forward_x, forward_y, forward_z) * entity.viz_instance.length
    else:  # Jet (box)
        entity.viz_instance.axis = vp.vector(forward_x, forward_y, forward_z)
    
    entity.viz_instance.up = vp.vector(up_x, up_y, up_z)
    
    # Update velocity arrow
    arrow_id = id(entity.viz_instance)
    if arrow_id in velocity_arrows:
        arrow = velocity_arrows[arrow_id]
        arrow.pos = vp.vector(*tuple(entity.p))
        
        velocity_magnitude = np.linalg.norm(entity.v)
        if velocity_magnitude > 0:
            velocity_normalized = entity.v / velocity_magnitude
            arrow_length = 200.0  # Fixed arrow length for visibility
            arrow.axis = vp.vector(*velocity_normalized) * arrow_length
        else:
            arrow.axis = vp.vector(1, 0, 0)
    
    # Update UI panel
    panel_id = id(entity.viz_instance)
    if panel_id in ui_panels:
        ui_label = ui_panels[panel_id]
        
        # Check entity type and format appropriate info
        if hasattr(entity, 'target'):  # Missile
            # Calculate distance to target
            distance_to_target = np.linalg.norm(entity.target.p - entity.p)
            
            # Calculate current speed
            speed = np.linalg.norm(entity.v)
            
            # Calculate interception time estimation
            if speed > 0:
                interception_time = distance_to_target / speed
            else:
                interception_time = float('inf')
            
            ui_label.text = (
                f"<span style='font-size:18px'>"
                f"<b style='color:blue'>MISSILE</b><br>"
                f"Distance: {distance_to_target:.1f} m<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Throttle: {entity.throttle*100:.0f}%<br>"
                f"ETA: {interception_time:.2f} s<br>"
                f"</span>"
                f"<hr>"
            )
        
        elif hasattr(entity, 'throttle'):  # Jet
            # Calculate current speed
            speed = np.linalg.norm(entity.v)
            
            ui_label.text = (
                f"<span style='font-size:18px'>"
                f"<b style='color:red'>JET</b><br>"
                f"Pitch: {entity.pitch:.1f}°<br>"
                f"Roll: {entity.roll:.1f}°<br>"
                f"Yaw: {entity.yaw:.1f}°<br>"
                f"Speed: {speed:.1f} m/s<br>"
                f"Throttle: {entity.throttle*100:.0f}%<br>"
                f"</span>"
                f"<hr>"
            )
    
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
