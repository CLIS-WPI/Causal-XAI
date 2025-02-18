# src/agv_path_manager.py
# agv_path_manager.py
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AGVPathManager:
    def __init__(self, config):
        """Initialize AGV path manager with configuration"""
        self.config = config
        self.current_waypoint_indices = {
            'agv_1': 0,
            'agv_2': 0
        }
        self.trajectories = config.agv_trajectories  # Make sure trajectories use same format
        self.last_known_positions = {
            'agv_1': None,
            'agv_2': None
        }
        self.current_velocities = {
            'agv_1': np.zeros(2),
            'agv_2': np.zeros(2)
        }
        self.scene_objects = None  # Will be set in _validate_scene_objects
        self._validate_scene_objects()

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
        
    def get_next_position(self, agv_id, current_position):
        """Calculate next position based on trajectory and speed with safety checks"""
        # Get proposed next position
        proposed_position = self._calculate_next_position(agv_id, current_position)
        
        # Safety checks
        if self.check_collision(proposed_position, self.config.scene_objects):
            logger.warning(f"Collision detected for AGV {agv_id}")
            return current_position
            
        if not self.validate_position(proposed_position):
            logger.warning(f"Invalid position detected for AGV {agv_id}")
            return current_position
            
        # Update status
        self.update_agv_status(agv_id, proposed_position)
        
        return proposed_position

    def _calculate_next_position(self, agv_id, current_position):
        """Calculate the next position based on trajectory"""
        trajectory = self.trajectories[agv_id]
        current_idx = self.current_waypoint_indices[agv_id]
        target = trajectory[current_idx]
        
        # Convert current position to 2D for path calculation
        current_2d = current_position[:2]
        
        # Calculate direction vector
        direction = np.array(target) - current_2d
        distance = np.linalg.norm(direction)
        
        # If close to waypoint, move to next waypoint
        if distance < 0.1:  # threshold
            self.current_waypoint_indices[f'agv_{agv_id}'] = \
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

    # check_collision method
    def check_collision(self, position, scene_objects):
        """Check if proposed position collides with any obstacle"""
        # Get obstacles in the correct format
        obstacles = self.config.get_obstacle_list()
        
        for obstacle in obstacles:
            try:
                obs_pos = obstacle['position']
                obs_dim = obstacle['dimensions']
                
                # Add safety margin
                safety_margin = 0.5  # meters
                
                # Check if position is within obstacle bounds with safety margin
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
        # Check room boundaries with safety margin
        margin = 0.5  # meters
        if (position[0] < margin or 
            position[0] > self.config.room_dim[0] - margin or
            position[1] < margin or 
            position[1] > self.config.room_dim[1] - margin):
            return False
        return True

    def check_agv_separation(self, agv1_pos, agv2_pos):
        """Ensure minimum safe distance between AGVs"""
        min_separation = 1.0  # meters
        distance = np.linalg.norm(np.array(agv1_pos) - np.array(agv2_pos))
        return distance >= min_separation

    def update_agv_status(self, agv_id, new_position):
        """Update AGV status including position and velocity"""
        if self.last_known_positions[f'agv_{agv_id}'] is not None:
            # Calculate velocity
            old_pos = self.last_known_positions[f'agv_{agv_id}']
            velocity = (np.array(new_position) - np.array(old_pos)) / \
                    self.config.agv_movement['update_interval']
            self.current_velocities[f'agv_{agv_id}'] = velocity[:2]  # Store 2D velocity
            
            # Check for excessive speed
            if np.linalg.norm(velocity[:2]) > self.config.agv_speed * 1.1:  # 10% tolerance
                logger.warning(f"AGV {agv_id} exceeding speed limit")
                self.emergency_stop(agv_id)
        
        self.last_known_positions[f'agv_{agv_id}'] = new_position

    def emergency_stop(self, agv_id):
        """Emergency stop procedure"""
        logger.warning(f"Emergency stop initiated for AGV {agv_id}")
        # Reset velocity
        self.current_velocities[f'agv_{agv_id}'] = np.zeros(2)
        # Keep last known position
        return self.last_known_positions[f'agv_{agv_id}']

    def get_current_status(self, agv_id):
        """Get current status of specified AGV"""
        return {
            'position': self.last_known_positions[f'agv_{agv_id}'],
            'velocity': self.current_velocities[f'agv_{agv_id}'],
            'waypoint_index': self.current_waypoint_indices[f'agv_{agv_id}']
        }

    def reset(self):
        """Reset AGV path manager to initial state"""
        self.current_waypoint_indices = {
            '1': 0,
            '2': 0
        }
        self.last_known_positions = {
            '1': None,
            '2': None
        }
        self.current_velocities = {
            '1': np.zeros(2),
            '2': np.zeros(2)
        }
        logger.info("AGV path manager reset to initial state")