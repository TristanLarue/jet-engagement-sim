import numpy as np

def missile_direct_attack_DEBUG(missile_entity):
    f = (missile_entity.target_entity.position - missile_entity.position); 
    f = f / max(float(np.linalg.norm(f)), 1e-9)
    u = np.array([0.0, 1.0, 0.0], dtype=float)
    r = np.cross(u, f)
    if float(np.linalg.norm(r)) < 1e-9: r = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), f)
    r = r / max(float(np.linalg.norm(r)), 1e-9)
    missile_entity.orientation = np.column_stack((f, np.cross(f, r), r))

def missile_predictive_attack(missile_entity):
    """
    Guidance method that predicts the jet's future position and aligns the missile to intercept it.
    Uses iterative calculation to find the intercept point.
    """
    # Get initial positions and velocities
    jet_pos = missile_entity.target_entity.position
    jet_velocity = missile_entity.target_entity.velocity
    missile_pos = missile_entity.position
    
    # Calculate missile speed with minimum of 400 m/s
    missile_speed = max(float(np.linalg.norm(missile_entity.velocity)), 800.0)
    
    # Initial difference vector
    diff = jet_pos - missile_pos
    distance = float(np.linalg.norm(diff))
    
    # Convergence parameters
    margin = 0.1  # meters
    max_iterations = 20
    predicted_jet_pos = jet_pos.copy()
    
    # Iterative prediction
    for _ in range(max_iterations):
        # Calculate time to intercept (eta)
        distance = float(np.linalg.norm(diff))
        if distance < margin:
            break
        
        eta = distance / missile_speed
        
        # Predict jet position at intercept time
        predicted_jet_pos = jet_pos + eta * jet_velocity
        
        # Recalculate difference vector
        new_diff = predicted_jet_pos - missile_pos
        new_distance = float(np.linalg.norm(new_diff))
        
        # Check convergence
        if abs(new_distance - distance) < margin:
            diff = new_diff
            break
        
        diff = new_diff
        distance = new_distance
    
    # Normalize the final direction vector
    f = diff / max(float(np.linalg.norm(diff)), 1e-9)
    
    # Create orthonormal basis for orientation matrix
    u = np.array([0.0, 1.0, 0.0], dtype=float)
    r = np.cross(u, f)
    if float(np.linalg.norm(r)) < 1e-9: 
        r = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), f)
    r = r / max(float(np.linalg.norm(r)), 1e-9)
    
    # Set missile orientation
    missile_entity.orientation = np.column_stack((f, np.cross(f, r), r))
