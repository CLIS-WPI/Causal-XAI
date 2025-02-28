# src/agv_path_manager.py
# agv_path_manager.py
import mitsuba

"""
AGV Path Manager (agv_path_manager.py)
=====================================

This module manages the navigation and movement of Automated Guided Vehicles (AGVs) 
in a smart factory environment. It ensures safe and efficient operation of multiple AGVs
by handling their trajectories and positions while maintaining safety protocols.

Key Responsibilities:
-------------------
1. Path Management:
- Tracks current positions and waypoints for each AGV
- Calculates next positions based on predefined trajectories
- Manages movement between waypoints

2. Safety Features:
- Validates positions within room boundaries
- Maintains safe distances between AGVs
- Prevents collisions with obstacles (e.g., shelves)
- Implements emergency stop procedures when needed

3. Status Monitoring:
- Tracks AGV velocities and positions
- Monitors movement speeds within safe limits
- Provides status updates and logging

Configuration Requirements:
------------------------
- AGV trajectories (predefined paths)
- Room dimensions and boundaries
- Obstacle positions and dimensions
- AGV movement parameters (speed, update interval)
- Safety margins and thresholds

Usage:
-----
The AGVPathManager class should be instantiated with a configuration object
containing all necessary parameters for AGV operation in the factory space.

Example:
    path_manager = AGVPathManager(config)
    next_position = path_manager.get_next_position('agv_1', current_position)
"""
import numpy as np
import tensorflow as tf
import logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AGVPathManager:
    def __init__(self, config, scene=None):
        """Initialize AGV path manager with configuration and optional scene"""
        self.config = config
        self.scene = scene
        self.num_agvs = config.num_agvs
        self.current_step = 0
        
        # Precompute paths based on trajectories
        self.paths = self._generate_paths()
        self.current_positions = {
            f'agv_{i}': self.paths[i][0] for i in range(self.num_agvs)
        }
        self.current_velocities = {
            f'agv_{i}': np.zeros(2, dtype=np.float32) for i in range(self.num_agvs)
        }
        self.movement_history = {
            f'agv_{i}': {
                'positions': [],
                'velocities': [],
                'timestamps': [],
                'los_status': []
            } for i in range(self.num_agvs)
        }
        
    def _generate_paths(self):
        """Generate smooth paths for AGVs based on trajectories"""
        steps = self.config.num_time_steps
        paths = []
        for i in range(self.num_agvs):
            agv_id = f'agv_{i}'
            traj_key = f'agv_{i+1}'
            traj = self.config.agv_trajectories[traj_key]
            num_points = len(traj)
            path = np.zeros((steps, 3), dtype=np.float32)
            step_per_segment = steps // (num_points - 1)
            
            for j in range(num_points - 1):
                start = np.array(traj[j] + [self.config.agv_height])
                end = np.array(traj[j + 1] + [self.config.agv_height])
                segment_steps = min(step_per_segment, steps - j * step_per_segment)
                path[j * step_per_segment:j * step_per_segment + segment_steps] = np.linspace(start, end, segment_steps)
            
            # Fill remaining steps with last position
            if j * step_per_segment + segment_steps < steps:
                path[j * step_per_segment + segment_steps:] = end
            
            # Adjust to avoid collisions
            for step in range(steps):
                max_attempts = 100  # Limit iterations
                attempts = 0
                while (self.check_collision(path[step], self.config.scene_objects) or 
                    not self.validate_position(path[step])):
                    if attempts >= max_attempts:
                        logger.warning(f"Could not find safe path for {agv_id} at step {step}, keeping original position")
                        break
                    if path[step, 1] < 10.0:
                        path[step, 1] += 0.5
                    else:
                        path[step, 1] -= 0.5
                    path[step, 1] = np.clip(path[step, 1], 0.5, self.config.room_dim[1] - 0.5)
                    attempts += 1
            
            paths.append(path)
        return paths

    def update_positions(self):
        """Update AGV positions for the current step"""
        if self.current_step < self.config.num_time_steps:
            for i in range(self.num_agvs):
                agv_id = f'agv_{i}'
                old_pos = self.current_positions[agv_id]
                new_pos = self.paths[i][self.current_step]
                velocity = (new_pos[:2] - old_pos[:2]) / self.config.agv_movement['update_interval']
                self.current_positions[agv_id] = new_pos
                self.current_velocities[agv_id] = velocity
            self.current_step += 1
        return np.array([self.current_positions[f'agv_{i}'] for i in range(self.num_agvs)])

    def check_collision(self, position, scene_objects):
        """Check if proposed position collides with any obstacle"""
        obstacles = self.config.get_obstacle_list()
        
        for obstacle in obstacles:
            try:
                obs_pos = obstacle['position']
                obs_dim = obstacle['dimensions']
                safety_margin = self.config.agv_movement['safety_margin']
                
                if (position[0] >= obs_pos[0] - (obs_dim[0]/2 + safety_margin) and 
                    position[0] <= obs_pos[0] + (obs_dim[0]/2 + safety_margin) and
                    position[1] >= obs_pos[1] - (obs_dim[1]/2 + safety_margin) and
                    position[1] <= obs_pos[1] + (obs_dim[1]/2 + safety_margin)):
                    return True
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Error processing obstacle {obstacle}: {str(e)}")
                continue
        return False

    def validate_position(self, position):
        """Validate if position is within safe bounds"""
        margin = self.config.agv_movement['safety_margin']
        if (position[0] < margin or 
            position[0] > self.config.room_dim[0] - margin or
            position[1] < margin or 
            position[1] > self.config.room_dim[1] - margin):
            return False
        return True

    def record_movement(self, agv_id, position, velocity, los_status):
        """Record AGV movement data with LOS status"""
        self.movement_history[agv_id]['positions'].append(position)
        self.movement_history[agv_id]['velocities'].append(velocity)
        self.movement_history[agv_id]['timestamps'].append(time.time())
        self.movement_history[agv_id]['los_status'].append(los_status)
        self.current_positions[agv_id] = position
        self.current_velocities[agv_id] = velocity

    def get_current_status(self, agv_id):
        """Get current status of specified AGV"""
        agv_key = f'agv_{agv_id}' if not str(agv_id).startswith('agv_') else agv_id
        return {
            'position': self.current_positions[agv_key],
            'velocity': self.current_velocities[agv_key]
        }

    def get_movement_history(self):
        """Return complete movement history"""
        return self.movement_history

    def _convert_scene_objects(self):
        """Convert config scene_objects dictionary to list format"""
        scene_objects_list = []
        
        # Convert shelves to list format
        for i in range(self.config.scene_objects['num_shelves']):
            shelf = {
                'position': self.config.scene_objects['shelf_positions'][i],
                'dimensions': self.config.scene_objects['shelf_dimensions'][i],
                'type': 'shelf',
                'material': self.config.scene_objects['shelf_material']
            }
            scene_objects_list.append(shelf)
        
        return scene_objects_list

    def _validate_scene_objects(self):
        """Validate scene objects configuration"""
        if not hasattr(self.config, 'scene_objects'):
            logger.warning("No scene_objects found in configuration")
            self.config.scene_objects = []
            return
        
        # Convert dictionary format to list format
        if isinstance(self.config.scene_objects, dict):
            self.scene_objects = self._convert_scene_objects()
        else:
            logger.error("scene_objects must be a dictionary with proper structure")
            raise ValueError("Invalid scene_objects configuration")
        
        # Validate converted list
        if not isinstance(self.scene_objects, list):
            logger.error("Converted scene_objects must be a list")
            raise ValueError("Invalid scene_objects conversion")
        
    # In agv_path_manager.py, enhance collision detection:
    def get_next_position(self, agv_id, current_pos):
        try:
            new_pos = self._calculate_next_position(agv_id, current_pos)
            
            # Enhanced collision check
            collision = False
            for obstacle in self.obstacle_positions:
                if np.linalg.norm(new_pos[:2] - obstacle[:2]) < self.config.agv_movement['min_distance']:
                    collision = True
                    break
                    
            if collision:
                return self._find_alternative_path(agv_id, current_pos)
                
            return new_pos
            
        except Exception as e:
            logger.warning(f"Path finding error: {str(e)}")
            return current_pos

    def _calculate_next_position(self, agv_id, current_position):
        """Calculate the next position based on trajectory"""
        try:
            trajectory = self.trajectories[agv_id]  # agv_id should already be a string
            current_idx = self.current_waypoint_indices[agv_id]
            target = trajectory[current_idx]
            
            
            logger.debug(f"AGV {agv_id} moving from {current_position} towards {target}")
            # Convert current position to 2D for path calculation
            current_2d = current_position[:2]
            
            # Calculate direction vector
            direction = np.array(target) - current_2d
            distance = np.linalg.norm(direction)
            
            # If close to waypoint, move to next waypoint
            if distance < 0.1:  # threshold
                self.current_waypoint_indices[agv_id] = \
                    (current_idx + 1) % len(trajectory)
                return current_position
            
            # Normalize direction and apply speed
            direction = direction / distance
            step = direction * self.config.agv_speed * self.config.agv_movement['update_interval']
            
            # Create new position (keeping height constant)
            new_position = np.array([
                current_2d[0] + step[0],
                current_2d[1] + step[1],
                self.config.agv_height
            ])
            
            return new_position
            
        except KeyError as e:
            logger.error(f"No trajectory found for AGV {agv_id}")
            raise KeyError(f"No trajectory found for AGV {agv_id}") from e
        except Exception as e:
            logger.error(f"Error calculating next position for AGV {agv_id}: {str(e)}")
            raise

    def check_agv_separation(self, agv1_pos, agv2_pos):
        """Ensure minimum safe distance between AGVs"""
        min_separation = self.config.agv_movement['min_distance']  # Minimum separation from config
        distance = np.linalg.norm(np.array(agv1_pos) - np.array(agv2_pos))
        return distance >= min_separation

    def update_agv_status(self, agv_id, new_position):
        """Update AGV status including position and velocity"""
        if self.last_known_positions[agv_id] is not None:  # Remove f'agv_{agv_id}'
            # Calculate velocity
            old_pos = self.last_known_positions[agv_id]
            velocity = (np.array(new_position) - np.array(old_pos)) / \
                    self.config.agv_movement['update_interval']
            self.current_velocities[agv_id] = velocity[:2]  # Store 2D velocity
            
            # Check for excessive speed
            if np.linalg.norm(velocity[:2]) > self.config.agv_speed * 1.1:  # 10% tolerance
                logger.warning(f"AGV {agv_id} exceeding speed limit")
                self.emergency_stop(agv_id)
        
        self.last_known_positions[agv_id] = new_position
            
    def emergency_stop(self, agv_id):
        """Emergency stop procedure"""
        logger.warning(f"Emergency stop initiated for AGV {agv_id}")
        # Reset velocity
        self.current_velocities[f'agv_{agv_id}'] = np.zeros(2)
        # Keep last known position
        return self.last_known_positions[f'agv_{agv_id}']

    def reset(self):
        """Reset AGV path manager to initial state"""
        self.current_waypoint_indices = {
            f'agv_{i}': 0 for i in range(self.config.num_agvs)
        }
        self.last_known_positions = {
            f'agv_{i}': None for i in range(self.config.num_agvs)
        }
        self.current_velocities = {
            f'agv_{i}': np.zeros(2) for i in range(self.config.num_agvs)
        }
        logger.info("AGV path manager reset to initial state")
    



