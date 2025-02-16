# src/beam_manager.py
import tensorflow as tf
import numpy as np
from dowhy import CausalModel
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class BeamManager:
    def __init__(self, config):
        self.config = config
        self.current_beam = None
        self.beam_history = []
        self.causal_data = []
        self.current_channel_state = None 
        self.channel_state_history = []    
        
    def get_current_beams(self):
        """Return the current beam configuration"""
        return self.current_beam
    
    def get_beam_history(self):
        """Return the history of beam configurations"""
        try:
            history_list = []
            for beam in self.beam_history:
                # Convert tensors to numpy arrays for easier serialization
                if isinstance(beam, tf.Tensor):
                    history_list.append(beam.numpy())
                else:
                    history_list.append(beam)
                    
            return history_list
            
        except Exception as e:
            logger.error(f"Error getting beam history: {str(e)}")
            return []
        
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
        
        # Update current_beam with the new optimal beams as a tensor
        self.current_beam = tf.stack(optimal_beams)
            
        return self.current_beam
        
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
        
        # Convert to degrees (multiply by 180/pi)
        azimuth_deg = azimuth * 180.0 / tf.constant(np.pi, dtype=tf.float32)
        elevation_deg = elevation * 180.0 / tf.constant(np.pi, dtype=tf.float32)
        
        # Ensure angles are within valid ranges
        azimuth_deg = tf.where(azimuth_deg < 0, azimuth_deg + 360, azimuth_deg)
        elevation_deg = tf.clip_by_value(elevation_deg, -90, 90)
        
        return tf.stack([azimuth_deg, elevation_deg])
    
    def update_channel_state(self, state_info):
        """Update the channel state information for beam management
        
        Args:
            state_info: Dictionary containing:
                - paths: Ray tracing paths object
                - los_available: Boolean indicating if LOS path is available
                - scene_state: Dictionary with AGV and obstacle positions
        """
        try:
            self.current_channel_state = state_info
            
            # If you want to store the state in history (optional)
            if hasattr(self, 'channel_state_history'):
                self.channel_state_history.append(state_info)
            else:
                self.channel_state_history = [state_info]
                
            logger.debug(f"Channel state updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating channel state: {str(e)}")
            
    def get_optimization_history(self):
        """Return the history of beam optimization steps"""
        try:
            # If we have channel state history, include that in the optimization history
            history = []
            for idx, beam in enumerate(self.beam_history):
                history_entry = {
                    'beam_configuration': beam.numpy(),
                    'channel_state': self.channel_state_history[idx] if idx < len(self.channel_state_history) else None,
                    'timestamp': idx
                }
                history.append(history_entry)
            
            return history
        except Exception as e:
            logger.error(f"Error getting optimization history: {str(e)}")
            return None

    def get_snr_improvement(self):
        """Calculate SNR improvement from initial to current beam configuration"""
        try:
            if not hasattr(self, 'channel_state_history') or len(self.channel_state_history) < 2:
                return 0.0
            
            # Get initial and current SNR values
            initial_state = self.channel_state_history[0]
            current_state = self.channel_state_history[-1]
            
            # Extract SNR values if they exist in the channel state
            initial_snr = 0.0
            current_snr = 0.0
            
            if isinstance(initial_state, dict) and 'paths' in initial_state:
                initial_snr = tf.reduce_mean(tf.abs(initial_state['paths'].A))
            
            if isinstance(current_state, dict) and 'paths' in current_state:
                current_snr = tf.reduce_mean(tf.abs(current_state['paths'].A))
                
            # Calculate improvement in dB
            improvement = 20 * tf.math.log(current_snr / (initial_snr + 1e-10)) / tf.math.log(10.0)
            
            return float(improvement.numpy())
        except Exception as e:
            logger.error(f"Error calculating SNR improvement: {str(e)}")
            return 0.0
        
    def get_convergence_time(self):
        """Calculate the time taken for beam adaptation to converge"""
        try:
            if not hasattr(self, 'channel_state_history') or len(self.channel_state_history) < 2:
                return 0.0
                
            # Get timestamps from history
            start_time = self.channel_state_history[0].get('timestamp', 0)
            
            # Find when SNR stabilizes (convergence)
            convergence_threshold = 0.1  # dB
            convergence_time = 0.0
            
            for i in range(1, len(self.channel_state_history)):
                current_state = self.channel_state_history[i]
                prev_state = self.channel_state_history[i-1]
                
                # Extract SNR values
                current_snr = 0.0
                prev_snr = 0.0
                
                if isinstance(current_state, dict) and 'paths' in current_state:
                    current_snr = tf.reduce_mean(tf.abs(current_state['paths'].A))
                if isinstance(prev_state, dict) and 'paths' in prev_state:
                    prev_snr = tf.reduce_mean(tf.abs(prev_state['paths'].A))
                
                # Check if SNR has stabilized
                snr_change = abs(current_snr - prev_snr)
                if snr_change < convergence_threshold:
                    convergence_time = current_state.get('timestamp', i) - start_time
                    break
                    
            return float(convergence_time)
        except Exception as e:
            logger.error(f"Error calculating convergence time: {str(e)}")
            return 0.0    