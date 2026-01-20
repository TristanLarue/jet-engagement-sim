# Missile-Jet Engagement Simulation

This project is a 3D physics simulation of aerial combat between a fighter jet and guided missiles. The simulation implements six-degree-of-freedom rigid body dynamics with aerodynamic forces, control surfaces, and torque-based rotations. It runs at 60Hz and supports both real-time visualization and training modes.

## Overview
The visualization displays two screens: the left shows the 3D spatial positions and trajectories of entities, while the right shows a force visualization diagram with the jet's orientation vectors and real-time physical data.




## Core Components

**Physics Engine** ([physics.py](src/main/physics.py))  
Implements aerodynamic calculations and numerical integration:
- Air density calculation using a barometric formula with temperature lapse rate
- Angle of attack and sideslip angle computation in body-frame coordinates
- Lift coefficient curve with linear region and post-stall cubic falloff
- Drag coefficient as quadratic function of angle of attack
- Dynamic pressure calculation for control surface forces
- Control effectiveness that degrades linearly beyond optimal angle of attack
- Rodrigues rotation formula for orientation updates from angular velocity
- SVD re-orthonormalization to maintain rotation matrix validity

**Entity System** ([entities.py](src/main/entities.py))  
Two entity classes inherit from a base entity that stores position, velocity, orientation, angular velocity, mass, aerodynamic coefficients, and moment of inertia.

*Jet*  
- Stores control inputs (pitch, roll, yaw) and throttle
- Calls deep learning module to set control inputs each tick
- Accumulates nine forces with application points: gravity, thrust, drag, lift, sideforce, elevator, two ailerons, and rudder
- Applies 0.95 damping factor to angular velocity
- Dies if altitude drops below zero

*Missile*  
- Stores target entity reference and guidance strategy function
- Calls guidance strategy each tick to update orientation
- Uses simplified roll alignment that sets roll to match velocity direction
- Accumulates six forces: gravity, thrust, drag, lift, elevator, and rudder
- Applies 0.9 damping factor to angular velocity
- Proximity detonation at 30 meter radius kills both missile and target

**Simulation Controller** ([simulation.py](src/main/simulation.py))  
- Runs at 60Hz with 40 second maximum duration per epoch
- Each tick calls `think()` then `tick()` on all alive entities
- In non-sprint mode, sleeps to maintain real-time pacing and updates visualization
- Terminates early if jet dies during physics update
- Supports multi-epoch training with scenario phase progression

**Guidance** ([guidance.py](src/main/guidance.py))  
Contains `missile_direct_attack_DEBUG()` which constructs a rotation matrix that orients the missile's forward axis toward the target position. This directly sets the missile orientation without using control surfaces.

**Scenario Generation** ([sim_presets.py](res/presets/sim_presets.py), [ent_presets.py](res/presets/ent_presets.py))  
Five difficulty phases with randomized initial conditions:
- **Phase 1**: Single jet, rewarded on survivability & stability
- **Phase 2**: Single jet, single missile, rewarded on survivability and distance from the missile's velocity vector
- **Phase 3**: Single jet with 3 missiles, good luck.

Entity presets define Su-57 specifications (26.7t mass, 78.8m² reference area, 284kN max thrust) and PAC-3 specifications (312kg mass, 120kN thrust, 5.2m length).

**Main Entry Point** ([main.py](src/main/main.py))  
- Prompts user for training mode (y/n)
-   y: Soft-Actor-Critic model will train silently with no vizualisation
-   n: Vizualisation will be initialized and will play a single epoch
- Handles keyboard interrupt and cleanup

## Physical Model

All positions, velocities, and forces are in world-space coordinates. Orientation is a 3x3 rotation matrix.

**Force Calculation**  
Forces are computed using standard aerodynamic formulas with dynamic pressure, reference areas, and coefficients. Control surfaces multiply dynamic pressure by area fraction, lift coefficient, and control input. Thrust applies at -50% of entity length. Drag and lift apply at -20% of entity length. Control surfaces apply at -45% of entity length, except ailerons at -5% length and ±50% wingspan.

**Torque and Rotation**  
Each force transforms to body-frame, computes torque via cross product with application point, then divides by moment of inertia to get angular acceleration. Angular velocity integrates from angular acceleration. Orientation updates using Rodrigues formula: constructs skew-symmetric matrix from rotation axis, applies matrix exponential approximation, then re-orthonormalizes via SVD.

**Limitations**  
Missiles bypass proper aerodynamics by aligning roll to velocity each tick, noted as "overcheated" in code comments. Thrust technically produces no torque despite off-center application point. Control surface areas are approximations (vertical stabilizer = 10% reference area, control surfaces = 80% reference area for jets).

## Installation and Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run simulation:
```bash
python src/main/main.py
```

Respond 'n' for single visualization epoch or 'y' for 10,000 training epochs without visualization.

Run tests:
```bash
pytest src/test/physics_tests.py
```

## Project Structure

```
missile-jet-engagement-sim/
├── src/main/
│   ├── main.py           # Entry point with mode selection
│   ├── simulation.py     # 60Hz tick loop and epoch management
│   ├── entities.py       # Jet and missile classes
│   ├── physics.py        # Force calculations and integration
│   ├── guidance.py       # Missile orientation logic
│   ├── viz.py            # VPython 3D rendering
│   └── deeplearning.py   # Neural network interface
├── res/presets/
│   ├── ent_presets.py    # Su-57 and PAC-3 specifications
│   └── sim_presets.py    # Phase-based scenario generation
├── src/test/
│   └── physics_tests.py  # Unit tests
└── requirements.txt
```

## Dependencies

- NumPy: Vector and matrix operations
- VPython: 3D visualization
- TensorFlow/Keras: Neural network training
- Matplotlib: Coefficient curve plotting
- Pytest: Testing framework

---
[ALL RIGHTS RESERVED]
