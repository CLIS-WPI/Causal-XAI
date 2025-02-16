# src/beam_manager.py
import tensorflow as tf
import numpy as np
from dowhy import CausalModel
import pandas as pd

class BeamManager:
    def __init__(self, config):
        self.config = config
        self.current_beam = None
        self.beam_history = []
        self.causal_data = []
        
    def detect_blockage(self, channel_data, agv_positions, obstacle_positions):
        """Detect if AGVs are blocked by obstacles"""
        los_blocked = []
        for agv_pos in agv_positions:
            bs_to_agv = agv_pos - self.config.bs_position
            distance = tf.norm(bs_to_agv)
            
            # Check intersection with obstacles
            for obstacle in obstacle_positions:
                if self._ray_intersects_obstacle(
                    self.config.bs_position, agv_pos, obstacle):
                    los_blocked.append(True)
                    break
            else:
                los_blocked.append(False)
                
        return los_blocked

    def _find_reflection_path(self, agv_pos, obstacle_positions, channel_data):
        """
        Find best reflection path when direct path is blocked
        
        Args:
            agv_pos: Position vector of the AGV
            obstacle_positions: List of obstacle positions
            channel_data: Dictionary containing channel metrics
            
        Returns:
            tf.Tensor: Best beam direction angles [azimuth, elevation]
        """
        # For now, return a simple offset from direct path
        # This should be enhanced with actual reflection calculations
        direct_beam = self._calculate_direct_beam(agv_pos)
        
        # Add small offset to try to find path around obstacle
        offset = tf.constant([15.0, 5.0])  # [azimuth, elevation] offset in degrees
        return direct_beam + offset
    
    def optimize_beam_direction(self, channel_data, agv_positions, obstacle_positions):
        """Optimize beam direction based on channel conditions and blockage"""
        blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
        
        # Calculate optimal beam directions for each AGV
        optimal_beams = []
        for i, agv_pos in enumerate(agv_positions):
            if blocked[i]:
                # Find best reflection path
                best_beam = self._find_reflection_path(
                    agv_pos, obstacle_positions, channel_data)
            else:
                # Direct path beamforming
                best_beam = self._calculate_direct_beam(agv_pos)
            optimal_beams.append(best_beam)
            
        return optimal_beams
        
    def perform_causal_analysis(self):
        """Perform causal inference on beam selection impact"""
        if len(self.causal_data) < self.config.causal['observation_window']:
            return None
            
        # Prepare data for causal analysis
        df = pd.DataFrame(self.causal_data[-self.config.causal['observation_window']:])
        
        # Create causal model
        model = CausalModel(
            data=df,
            treatment=self.config.causal['treatment_variables'],
            outcome=self.config.causal['outcome_variables'],
            common_causes=self.config.causal['confounders']
        )
        
        # Identify causal effect
        identified_estimand = model.identify_effect()
        
        # Estimate effect
        estimate = model.estimate_effect(identified_estimand,
                                    method_name="backdoor.linear_regression")
                                    
        return estimate
        
    def update_causal_data(self, beam_direction, channel_metrics, agv_state):
        """Update dataset for causal analysis"""
        self.causal_data.append({
            'beam_direction': beam_direction,
            'snr': channel_metrics['snr'],
            'throughput': channel_metrics['throughput'],
            'obstacle_presence': channel_metrics['los_blocked'],
            'agv_speed': agv_state['speed'],
            'distance_to_bs': agv_state['distance']
        })

    def _ray_intersects_obstacle(self, start_point, end_point, obstacle_position):
        """
        Check if a ray between two points intersects with an obstacle
        
        Args:
            start_point: Starting point of the ray (e.g., BS position)
            end_point: End point of the ray (e.g., AGV position)
            obstacle_position: Position of the obstacle to check against
            
        Returns:
            bool: True if ray intersects obstacle, False otherwise
        """
        # Convert points to TensorFlow tensors if they aren't already
        start = tf.cast(start_point, tf.float32)
        end = tf.cast(end_point, tf.float32)
        obstacle = tf.cast(obstacle_position, tf.float32)
        
        # Calculate ray direction
        ray_direction = end - start
        ray_length = tf.norm(ray_direction)
        ray_direction = ray_direction / ray_length
        
        # Calculate vector from start to obstacle
        to_obstacle = obstacle - start
        
        # Project obstacle point onto ray
        projection = tf.reduce_sum(to_obstacle * ray_direction)
        
        # Find closest point on ray to obstacle
        closest_point = start + projection * ray_direction
        
        # Calculate distance from obstacle to ray
        distance_to_ray = tf.norm(obstacle - closest_point)
        
        # Check if projection point is between start and end
        is_between = (projection >= 0) & (projection <= ray_length)
        
        # Define obstacle radius (adjust based on your needs)
        obstacle_radius = 1.0  # meters
        
        # Return True if ray intersects obstacle
        return tf.logical_and(distance_to_ray < obstacle_radius, is_between)
    
    def _calculate_direct_beam(self, agv_position):
        """
        Calculate optimal beam direction for direct line-of-sight path
        
        Args:
            agv_position: Position vector of the AGV [x, y, z]
            
        Returns:
            tf.Tensor: Optimal beam direction angles [azimuth, elevation]
        """
        # Convert positions to TensorFlow tensors
        agv_pos = tf.cast(agv_position, tf.float32)
        bs_pos = tf.cast(self.config.bs_position, tf.float32)
        
        # Calculate vector from BS to AGV
        direction_vector = agv_pos - bs_pos
        
        # Calculate azimuth angle (in xy-plane)
        azimuth = tf.math.atan2(
            direction_vector[1],  # y component
            direction_vector[0]   # x component
        )
        
        # Calculate elevation angle
        horizontal_distance = tf.norm(direction_vector[:2])  # Distance in xy-plane
        elevation = tf.math.atan2(
            direction_vector[2],  # z component
            horizontal_distance
        )
        
        # Convert to degrees
        azimuth_deg = tf.math.degrees(azimuth)
        elevation_deg = tf.math.degrees(elevation)
        
        # Ensure angles are within valid ranges
        azimuth_deg = tf.where(azimuth_deg < 0, azimuth_deg + 360, azimuth_deg)
        elevation_deg = tf.clip_by_value(elevation_deg, -90, 90)
        
        return tf.stack([azimuth_deg, elevation_deg])