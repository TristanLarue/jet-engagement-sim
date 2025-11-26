# missile-jet-engagement-sim
[ALL RIGHTS RESERVED]

This project is a 3D physics-based simulation of missile and jet engagements. It models a guided missile attempting to intercept a maneuvering fighter jet using simplified six-degree-of-freedom (6-DOF) dynamics and classical guidance laws.

## Features
- Realistic 3D simulation of missile and jet flight
- Physics engine with forces, moments, and rotation matrices
- Guidance systems: Pure Pursuit, Proportional Navigation, and jet stabilization
- Visualization using VPython: shows velocity and orientation arrows for each entity
- Easily extensible for new guidance algorithms, including AI-based approaches
- Modular codebase: entities, physics, guidance, and visualization are separated

## What It Does
- Simulates the flight of a missile and a jet in a 3D environment
- Applies physics-based forces and moments to update position and orientation
- Uses guidance logic for missile pursuit and jet stabilization
- Visualizes the simulation in real time, showing key flight vectors

## Getting Started
1. Install Python 3 and required packages (see `requirements.txt`)
2. Run `main.py` to start the simulation
3. Observe the missile attempting to intercept the jet in the VPython window

## Extending the Project
- Add new guidance laws in `guidance.py`
- Modify physics or entity properties in `physics.py` and `entities.py`
- Customize visualization in `viz.py`

---
[ALL RIGHTS RESERVED]
