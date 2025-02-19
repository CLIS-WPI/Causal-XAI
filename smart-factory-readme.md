# Smart Factory Simulation with AGV Beamforming

## Project Overview
This project simulates an indoor smart factory environment featuring Autonomous Guided Vehicles (AGVs) with dynamic beamforming adaptation. The simulation uses Sionna NVIDIA' to simulate how the beam changes dynamically when an AGV moves behind an obstacle, temporarily blocking the LoS (Line of Sight), forcing the BS (Base Station) to switch beams to maintain connectivity.

## Environment Specifications

### Factory Layout
- Dimensions: 20m × 20m × 5m (Length × Width × Height)
- Wall Material: Concrete (high reflectivity)
- Base Station Location: [10, 0.5, 4.5] (ceiling-mounted, downward-facing)
- Obstacles: 5 metallic shelves at fixed positions

### Base Station Configuration
- Antenna: 16×4 UPA (Uniform Planar Array)
- Frequency: 28 GHz (mmWave)
- Beamforming: AI-assisted adaptive beam selection
- Beam Configuration:
  - 32 possible directions
  - 15 degrees beam width
  - 60 degrees max steering angle
  - 10 dB minimum SNR threshold

### AGV Specifications
- Quantity: 2 mobile robots
- Antenna: Single-antenna receiver (1×1 MIMO)
- Speed: 3 km/h (0.83 m/s)
- Movement: Predefined trajectories

## Fixed Obstacle Layout

| Obstacle # | Position (x, y, z) | Dimensions (L×W×H) |
|------------|-------------------|-------------------|
| 1          | (4, 4, 0)        | 4m × 1m × 4m     |
| 2          | (12, 4, 0)       | 2m × 1m × 4m     |
| 3          | (4, 10, 0)       | 1m × 2m × 5m     |
| 4          | (14, 12, 0)      | 1m × 2m × 5m     |
| 5          | (8, 16, 0)       | 4m × 1m × 5m     |

## AGV Movement Paths

| AGV | Start Position (x, y, z) | Path Type | Waypoints |
|-----|-------------------------|------------|-----------|
| AGV 1 | (2, 2, 0.5) | Straight-line (back & forth) | [(2,2), (18,2), (2,2)] |
| AGV 2 | (17, 18, 0.5) | Rectangular loop | [(17,18), (3,18), (3,3), (17,3), (17,18)] |

## Technical Components

### Wireless Channel Model (Sionna)
- Propagation Model: Indoor Factory (InF) mmWave
- Path Loss: Dynamic LoS/NLoS calculations
- Multipath Effects: High reflectivity modeling
- Fading Models:
  - LoS: Rician fading
  - NLoS: Rayleigh fading

### AI Components


## Core Functionalities

1. Environment Setup
   - Factory layout initialization
   - Obstacle and AGV placement

2. AGV Movement Simulation
   - Path-based movement
   - LoS/NLoS detection

3. Wireless Channel Simulation
   - Dynamic signal strength calculation

4. Beamforming Adaptation
   - Position-based beam selection


## Expected Outcomes
   -observing beam changes dynamically when an AGV moves behind an obstacle.
   -Performance metrics to observe beam adaptation:
   -SNR drop when AGV moves behind a shelf.
   -Time taken for beam switch to restore connectivity.
   -Packet success rate & BER impact during beam changes.