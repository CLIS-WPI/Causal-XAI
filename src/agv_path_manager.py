# src/agv_path_manager.py
import numpy as np
import tensorflow as tf

class AGVPathManager:
    def __init__(self, config):
        self.config = config
        self.current_waypoint_indices = {
            'agv_1': 0,
            'agv_2': 0
        }
        self.trajectories = config.agv_trajectories
        
    def get_next_position(self, agv_id, current_position):
        """Calculate next position based on trajectory and speed"""
        trajectory = self.trajectories[f'agv_{agv_id+1}']
        current_idx = self.current_waypoint_indices[f'agv_{agv_id+1}']
        target = trajectory[current_idx]
        
        # Convert current position to 2D for path calculation
        current_2d = current_position[:2]
        
        # Calculate direction vector
        direction = np.array(target) - current_2d
        distance = np.linalg.norm(direction)
        
        # If close to waypoint, move to next waypoint
        if distance < 0.1:  # threshold
            self.current_waypoint_indices[f'agv_{agv_id+1}'] = \
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