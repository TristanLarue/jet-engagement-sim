import numpy as np

def missile_direct_attack_DEBUG(missile_entity):
    f = (missile_entity.target_entity.position - missile_entity.position); 
    f = f / max(float(np.linalg.norm(f)), 1e-9)
    u = np.array([0.0, 1.0, 0.0], dtype=float)
    r = np.cross(u, f)
    if float(np.linalg.norm(r)) < 1e-9: r = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), f)
    r = r / max(float(np.linalg.norm(r)), 1e-9)
    missile_entity.orientation = np.column_stack((f, np.cross(f, r), r))
