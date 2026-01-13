# Missile-Jet Engagement Simulation
[ALL RIGHTS RESERVED]

This project is a high-fidelity 3D physics simulation of aerial combat scenarios, modeling the dynamic interaction between guided missiles and maneuvering fighter jets. The simulation implements six-degree-of-freedom (6-DOF) rigid body dynamics with accurate aerodynamic modeling, control surface physics, and real-time engagement scenarios.

## Overview

The simulation environment models realistic aerial engagements between a Sukhoi Su-57 fighter jet and PAC-3 Patriot missiles. Each entity operates under complete physical constraints including atmospheric effects, thrust dynamics, drag forces, lift generation, and control surface responses. The simulation runs at 60Hz and supports both real-time visualization and accelerated training modes.

When running the simulation, users observe two primary displays: the left screen shows the 3D spatial visualization of the engagement with entity positions and trajectories, while the right screen presents a real-time force visualization diagram showing the jet's orientation, velocity vectors, and physical state data including altitude, speed, angle of attack, and G-forces.

## Architecture

### Core Simulation Components

**Physics Engine** ([physics.py](src/main/physics.py))  
The physics module implements fundamental aerodynamic calculations and state integration. Key systems include:
- Atmospheric density modeling using barometric formulas for altitude-dependent air density
- Angle of attack and sideslip calculations in body-frame coordinates
- Lift and drag coefficient curves with stall characteristics for symmetrical airfoils
- Control surface force generation (elevators, ailerons, rudders) with effectiveness degradation at high angles of attack
- Quaternion-based orientation integration using rotation matrices with SVD re-orthonormalization
- Force-to-torque conversion accounting for moment arms and moments of inertia

**Entity System** ([entities.py](src/main/entities.py))  
The entity architecture defines two primary classes inheriting from a base entity:

*Jet Class*  
Models a fighter aircraft with:
- Thrust vectoring through throttle control (0-284kN for Su-57)
- Three-axis control inputs (pitch, roll, yaw) mapped to control surface deflections
- Composite force accumulation from gravity, thrust, drag, lift, sideforce, and control surfaces
- Angular velocity damping and torque-based rotation dynamics
- Ground collision detection

*Missile Class*  
Models a guided interceptor with:
- Constant maximum thrust propulsion
- Simplified two-axis guidance (pitch and yaw control)
- Roll stabilization aligned with velocity vector
- Proximity detonation logic with configurable explosion radius
- Target tracking through guidance strategy callbacks

Both entities compute their next state through a tick-based update cycle, accumulating forces and torques before integrating accelerations into velocity and position using explicit Euler integration.

**Simulation Controller** ([simulation.py](src/main/simulation.py))  
Manages the execution environment:
- Fixed timestep simulation at 60Hz with configurable real-time pacing
- Entity lifecycle management (spawning, state updates, destruction)
- Integration with visualization and machine learning systems
- Multi-epoch training support with scenario generation
- Graceful termination on entity destruction or timeout

**Guidance Systems** ([guidance.py](src/main/guidance.py))  
Contains missile guidance algorithms. Currently implements direct attack guidance where the missile continuously reorients toward the target position. The modular structure allows for implementation of advanced guidance laws such as proportional navigation or predictive intercept algorithms.

**Scenario Presets** ([sim_presets.py](res/presets/sim_presets.py), [ent_presets.py](res/presets/ent_presets.py))  
Defines five progressive difficulty phases for training:
- **Phase 1**: Single jet with moderate speed and minimal angular velocity
- **Phase 2**: Jet with increased speed range and rotational dynamics
- **Phase 3**: Jet at high altitude with one missile threat
- **Phase 4**: Two-missile engagement scenario
- **Phase 5**: Three-missile engagement with maximum difficulty parameters

Entity presets contain realistic physical specifications including mass, reference areas, moment of inertia tensors, and aerodynamic coefficients derived from representative aircraft and missile characteristics.

### Entry Point

**Main Controller** ([main.py](src/main/main.py))  
Orchestrates simulation execution:
- Environment configuration and warning suppression for TensorFlow integration
- Visualization setup in real-time mode
- Training vs. demonstration mode selection
- Keyboard interrupt handling and cleanup procedures

The main loop prompts for training mode selection, configuring either single-epoch demonstration runs or multi-epoch training sessions (default 10,000 epochs).

## Physical Modeling

The simulation implements a comprehensive force model for each entity:

**Primary Forces**:
- **Gravity**: Constant 9.81 m/s² downward acceleration
- **Thrust**: Directional force along the forward axis, magnitude scaled by throttle
- **Drag**: Opposes velocity, magnitude determined by dynamic pressure, reference area, and drag coefficient curve
- **Lift**: Perpendicular to velocity in the roll-up plane, generated by angle of attack
- **Sideforce**: Lateral stabilization force from vertical stabilizer based on sideslip angle

**Control Surface Forces**:
- **Elevator**: Generates pitch moment through horizontal tail surfaces
- **Aileron**: Produces roll moment through differential wing surface deflection
- **Rudder**: Creates yaw moment via vertical tail surface

Each force includes proper torque calculations based on application points relative to the center of mass, enabling realistic rotational dynamics. The simulation accounts for control effectiveness degradation as angle of attack exceeds optimal values, simulating post-stall behavior.

## Installation and Usage

**Requirements**:
- Python 3.10 or higher
- NumPy for numerical computations
- VPython for 3D visualization
- TensorFlow and Keras for machine learning integration
- Matplotlib for coefficient curve visualization

Install dependencies:
```bash
pip install -r requirements.txt
```

**Running the Simulation**:
```bash
python src/main/main.py
```

Upon execution, the program prompts for training mode. Selecting 'n' runs a single demonstration epoch with full visualization, while 'y' initiates accelerated training mode without rendering overhead.

**Testing**:
The project includes physics unit tests verifiable through pytest:
```bash
pytest src/test/physics_tests.py
```

## Project Structure

```
missile-jet-engagement-sim/
├── src/main/
│   ├── main.py           # Entry point and execution controller
│   ├── simulation.py     # Simulation loop and epoch management
│   ├── entities.py       # Jet and missile entity implementations
│   ├── physics.py        # Aerodynamic calculations and integration
│   ├── guidance.py       # Missile guidance algorithms
│   ├── viz.py            # VPython visualization (3D rendering)
│   └── deeplearning.py   # Neural network training interface
├── res/
│   ├── presets/
│   │   ├── ent_presets.py  # Entity physical specifications
│   │   └── sim_presets.py  # Scenario generation functions
│   └── viz_models/         # 3D model data for entities
├── src/test/
│   └── physics_tests.py  # Physics module unit tests
└── requirements.txt      # Python dependencies
```

## Technical Notes

- The simulation uses world-space coordinates for all positions and velocities
- Orientation is represented as 3x3 rotation matrices (SO(3) group)
- Angular velocity is tracked in body-frame coordinates as degrees per second
- The physics engine employs explicit Euler integration at 60Hz timestep
- Missiles currently use simplified roll alignment to velocity for stability (noted as "overcheated" in code)
- Control surface effectiveness includes stall behavior beyond optimal angle of attack
- Air density calculation uses a simplified barometric formula valid up to stratospheric altitudes

## Extension Points

The modular architecture facilitates several extension opportunities:
- Implement advanced guidance laws (proportional navigation, augmented proportional navigation, optimal guidance)
- Add sensor modeling with field-of-view constraints and noise characteristics
- Introduce countermeasure systems (chaff, flares, electronic warfare)
- Develop more sophisticated missile flight dynamics with proper roll control
- Create multiplayer scenarios with coordinated missile salvos
- Integrate reinforcement learning agents for autonomous evasion maneuvers

---
[ALL RIGHTS RESERVED]
