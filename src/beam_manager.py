# src/beam_manager.py
import tensorflow as tf
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)

class BeamManager:
    def __init__(self, config):
        self.config = config
        self.last_switch_time = None  # Track when the last switch occurred
        self.current_beam = None
        self.beam_history = []
        self.snr_history = []
        self.switch_times = []
        self.current_channel_state = None 
        self.channel_state_history = []
        self.packet_stats = {
            'total': 0,
            'successful': 0,
            'failed_during_switch': 0
        }
        self.current_switch_start = None

        # Initialize beam codebook for 16x4 array
        self.num_beams_azimuth = 16
        self.num_beams_elevation = 4
        self._initialize_beam_codebook()

    def log_beam_switch(self, old_beam, new_beam):
        switch_time = time.time()
        if self.current_switch_start is None:
            self.current_switch_start = switch_time
            switch_duration = switch_time - self.current_switch_start
            self.switch_times.append({
                'timestamp': switch_time,
                'duration': switch_duration,
                'old_beam': old_beam,
                'new_beam': new_beam
            })
            self.current_switch_start = None
    def has_switch_occurred(self):
        """
        Check if a beam switch has occurred by comparing current and previous beam configurations
        Returns:
            bool: True if a switch occurred, False otherwise
        """
        if len(self.beam_history) < 2:
            return False
        
        current = self.current_beam
        previous = self.beam_history[-2]
        
        # Convert tensors to numpy for comparison
        if isinstance(current, tf.Tensor):
            current = current.numpy()
        if isinstance(previous, tf.Tensor):
            previous = previous.numpy()
            
        return not np.array_equal(current, previous)

    def update_packet_stats(self, success, during_switch=False):    
        self.packet_stats['total'] += 1
        if success:
            self.packet_stats['successful'] += 1
        elif during_switch:
            self.packet_stats['failed_during_switch'] += 1
            
    def log_snr(self, snr_value):
        self.snr_history.append({
            'timestamp': time.time(),
            'value': snr_value
        })

    def get_performance_metrics(self):
        return {
            'switch_times': self.switch_times,
            'packet_success_rate': self.packet_stats['successful'] / max(1, self.packet_stats['total']),
            'switch_failure_rate': self.packet_stats['failed_during_switch'] / max(1, self.packet_stats['total']),
            'snr_history': self.snr_history
        }   
        
    def _initialize_beam_codebook(self):
        """Initialize beam codebook with proper bounds checking"""
        # Create beam directions for 16x4 array
        azimuth_angles = np.linspace(-60, 60, self.num_beams_azimuth)
        elevation_angles = np.linspace(-30, 30, self.num_beams_elevation)
        
        # Generate codebook
        self.beam_codebook = []
        for az in azimuth_angles:
            for el in elevation_angles:
                self.beam_codebook.append([az, el])
        
        # Convert to tensor
        self.beam_codebook = tf.convert_to_tensor(self.beam_codebook, dtype=tf.float32)
        logger.info(f"Beam codebook initialized with {len(self.beam_codebook)} beam directions")
    
    def get_current_beams(self):
        """Return the current beam configuration"""
        return self.current_beam
    
    def should_switch_beam(self, current_snr, proposed_snr):
        """Determine if beam switching is needed based on SNR"""
        MIN_SNR_THRESHOLD = 10.0  # dB
        SNR_IMPROVEMENT_THRESHOLD = 2.0  # dB
        
        # Check if current SNR is below minimum threshold
        if current_snr < MIN_SNR_THRESHOLD:
            return True
        
        # Check if proposed beam offers significant improvement
        return (proposed_snr - current_snr) > SNR_IMPROVEMENT_THRESHOLD

    def get_beam_history(self):
        """Return the history of beam configurations"""
        try:
            return [beam.numpy() if isinstance(beam, tf.Tensor) else beam 
                    for beam in self.beam_history]
        except Exception as e:
            logger.error(f"Error getting beam history: {str(e)}")
            return []
    def update_beam(self, new_beam):
        """
        Update the current beam and track it in history
        
        Args:
            new_beam: New beam configuration to be set
            
        Returns:
            None
        """
        try:
            # Store old beam for logging
            old_beam = self.current_beam
            
            # Convert new_beam to tensor if it isn't already
            if not isinstance(new_beam, tf.Tensor):
                new_beam = tf.convert_to_tensor(new_beam, dtype=tf.float32)
                
            # Add current beam to history before updating
            if self.current_beam is not None:
                self.beam_history.append(self.current_beam)
                
            # Update current beam
            self.current_beam = new_beam
            
            # Log the beam switch if it occurred
            if old_beam is not None and not tf.reduce_all(tf.equal(old_beam, new_beam)):
                self.log_beam_switch(old_beam, new_beam)
                
            logger.debug(f"Beam updated - Old: {old_beam}, New: {new_beam}")
            
        except Exception as e:
            logger.error(f"Error updating beam: {str(e)}")
            raise RuntimeError(f"Failed to update beam: {str(e)}")
            
    def detect_blockage(self, channel_data, agv_positions, obstacle_positions):
        """Enhanced blockage detection with SNR threshold"""
        try:
            los_blocked = []
            for agv_pos in agv_positions:
                # Check direct path blockage
                direct_blocked = False
                for obstacle in obstacle_positions:
                    if self._ray_intersects_obstacle(
                        self.config.bs_position, agv_pos, obstacle):
                        direct_blocked = True
                        break
                
                # Check SNR if channel data is available
                if channel_data and 'snr' in channel_data:
                    snr_threshold = self.config.beamforming['min_snr_threshold']
                    snr_blocked = channel_data['snr'] < snr_threshold
                    los_blocked.append(direct_blocked or snr_blocked)
                else:
                    los_blocked.append(direct_blocked)
                    
            return los_blocked
            
        except Exception as e:
            logger.error(f"Error in blockage detection: {str(e)}")
            return [True] * len(agv_positions)  # Conservative approach

    def _find_reflection_path(self, agv_pos, obstacle_positions, channel_data):
        """Enhanced reflection path finding"""
        try:
            direct_beam = self._calculate_direct_beam(agv_pos)
            
            # Search through beam codebook for best alternative
            best_beam = None
            best_metric = -float('inf')
            
            for beam in self.beam_codebook:
                # Skip beams too close to blocked direct path
                if tf.norm(beam - direct_beam) < 15.0:  # Minimum angular separation
                    continue
                    
                # Calculate expected path metric
                metric = self._calculate_path_metric(
                    beam, agv_pos, obstacle_positions, channel_data)
                
                if metric > best_metric:
                    best_metric = metric
                    best_beam = beam
            
            return best_beam if best_beam is not None else direct_beam
            
        except Exception as e:
            logger.error(f"Error finding reflection path: {str(e)}")
            return direct_beam
    
    def _calculate_path_metric(self, beam, agv_pos, obstacle_positions, channel_data):
        """Calculate metric for potential beam direction"""
        try:
            # Convert beam angles to direction vector
            azimuth, elevation = beam[0], beam[1]
            direction = self._angles_to_vector(azimuth, elevation)
            
            # Check for obstacles in path
            for obstacle in obstacle_positions:
                if self._ray_intersects_obstacle(
                    self.config.bs_position, 
                    self.config.bs_position + direction * 20.0,  # Project ray
                    obstacle):
                    return -float('inf')
            
            # Include channel quality if available
            metric = 0.0
            if channel_data and 'snr' in channel_data:
                metric += channel_data['snr']
            
            # Prefer smaller steering angles
            metric -= 0.1 * tf.norm(beam)  # Penalty for large steering angles
            
            return float(metric)
            
        except Exception as e:
            logger.error(f"Error calculating path metric: {str(e)}")
            return -float('inf')
    
    def optimize_beam_direction(self, channel_data, path_manager, obstacle_positions):
        try:
            # Get current AGV positions
            agv_positions = []
            for agv_id in ['agv_1', 'agv_2']:
                agv_status = path_manager.get_current_status(agv_id)
                if agv_status['position'] is not None:
                    agv_positions.append(agv_status['position'])
            
            # Detect blockages
            blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
            
            # Enhanced beam selection logic
            optimal_beams = []
            for i, agv_pos in enumerate(agv_positions):
                current_snr = channel_data.get('snr', 0) if channel_data else 0
                
                if blocked[i] or current_snr < 10.0:  # Check SNR threshold
                    # Find best reflection path
                    best_beam = self._find_reflection_path(
                        agv_pos, obstacle_positions, channel_data)
                    logger.debug(f"AGV {i+1} needs alternative path - SNR: {current_snr}")
                else:
                    best_beam = self._calculate_direct_beam(agv_pos)
                    logger.debug(f"AGV {i+1} using direct path - SNR: {current_snr}")
                
                optimal_beams.append(best_beam)
            
            self.current_beam = tf.stack(optimal_beams)
            self.beam_history.append(self.current_beam)
            
            return self.current_beam
            
        except Exception as e:
            logger.error(f"Error in beam optimization: {str(e)}")
            return self.current_beam

    def _ray_intersects_obstacle(self, start_point, end_point, obstacle_position):
        """Optimized ray-obstacle intersection check"""
        try:
            start = tf.cast(start_point, tf.float32)
            end = tf.cast(end_point, tf.float32)
            obstacle = tf.cast(obstacle_position, tf.float32)
            
            ray_direction = end - start
            ray_length = tf.norm(ray_direction)
            ray_direction = ray_direction / ray_length
            
            to_obstacle = obstacle - start
            projection = tf.reduce_sum(to_obstacle * ray_direction)
            closest_point = start + projection * ray_direction
            
            distance_to_ray = tf.norm(obstacle - closest_point)
            is_between = (projection >= 0) & (projection <= ray_length)
            
            # Adjusted obstacle radius based on shelf dimensions
            obstacle_radius = 1.5  # Increased for safety margin
            
            return tf.logical_and(distance_to_ray < obstacle_radius, is_between)
            
        except Exception as e:
            logger.error(f"Error in ray intersection check: {str(e)}")
            return True  # Conservative approach
    
    def _calculate_direct_beam(self, agv_position):
        """Calculate direct beam angles to AGV"""
        try:
            agv_pos = tf.cast(agv_position, tf.float32)
            bs_pos = tf.cast(self.config.bs_position, tf.float32)
            
            direction_vector = agv_pos - bs_pos
            
            # Calculate angles
            azimuth = tf.math.atan2(direction_vector[1], direction_vector[0])
            horizontal_distance = tf.norm(direction_vector[:2])
            elevation = tf.math.atan2(direction_vector[2], horizontal_distance)
            
            # Convert to degrees
            azimuth_deg = azimuth * 180.0 / np.pi
            elevation_deg = elevation * 180.0 / np.pi
            
            # Ensure angles are within valid ranges
            azimuth_deg = tf.where(azimuth_deg < 0, azimuth_deg + 360, azimuth_deg)
            elevation_deg = tf.clip_by_value(elevation_deg, -30, 30)  # Limited elevation range
            
            MAX_STEERING_ANGLE = 60.0  # degrees
            azimuth_deg = tf.clip_by_value(azimuth_deg, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
            elevation_deg = tf.clip_by_value(elevation_deg, -30, 30)

            return tf.stack([azimuth_deg, elevation_deg])
            
        except Exception as e:
            logger.error(f"Error calculating direct beam: {str(e)}")
            return tf.constant([0.0, 0.0])  # Default to broadside beam

    def _angles_to_vector(self, azimuth, elevation):
        """Convert angles to direction vector"""
        try:
            azimuth_rad = azimuth * np.pi / 180.0
            elevation_rad = elevation * np.pi / 180.0
            
            x = tf.cos(elevation_rad) * tf.cos(azimuth_rad)
            y = tf.cos(elevation_rad) * tf.sin(azimuth_rad)
            z = tf.sin(elevation_rad)
            
            return tf.stack([x, y, z])
            
        except Exception as e:
            logger.error(f"Error converting angles to vector: {str(e)}")
            return tf.constant([1.0, 0.0, 0.0])  # Default direction
    
    #add adaptive beamforming
    def optimize_beam_direction(self, channel_data, path_manager, obstacle_positions):
        # Add ML-based beam prediction
        predicted_beam = self._predict_optimal_beam(channel_data)
        
        # Add beam refinement
        refined_beam = self._refine_beam(predicted_beam, channel_data)
        
        # Add multi-path combining
        combined_beam = self._combine_multipath(refined_beam, channel_data)
        
        return combined_beam
    
    def _predict_optimal_beam(self, channel_data):
        """Predict optimal beam with proper bounds checking"""
        try:
            if 'beam_metrics' not in channel_data:
                return self.current_beam if self.current_beam is not None else tf.zeros([2])
                
            # Extract and reshape SNR data
            snr_data = channel_data['beam_metrics'].get('snr_db', 0)
            if isinstance(snr_data, (tf.Tensor, np.ndarray)):
                # Ensure SNR data matches codebook size
                snr_data = tf.reshape(snr_data, [-1])
                num_beams = len(self.beam_codebook)
                snr_data = snr_data[:num_beams]  # Clip to valid range
                
                # Get best beam index with bounds checking
                best_beam_idx = tf.clip_by_value(
                    tf.argmax(snr_data), 
                    0, 
                    num_beams - 1
                )
                predicted_beam = tf.gather(self.beam_codebook, best_beam_idx)
            else:
                predicted_beam = tf.zeros([2])
                
            return predicted_beam
            
        except Exception as e:
            logger.error(f"Error in beam prediction: {str(e)}")
            return self.current_beam if self.current_beam is not None else tf.zeros([2])
    
    def _refine_beam(self, predicted_beam, channel_data):
        """
        Refines the predicted beam direction based on channel conditions
        
        Args:
            predicted_beam: Initial beam prediction
            channel_data: Channel state information
            
        Returns:
            tf.Tensor: Refined beam direction
        """
        try:
            if predicted_beam is None:
                return self.current_beam if self.current_beam is not None else tf.zeros([2])
                
            # Apply refinement based on channel quality
            if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                current_snr = tf.reduce_mean(channel_data['beam_metrics']['snr_db'])
                
                # If SNR is good, make smaller adjustments
                if current_snr > 20.0:  # Good SNR threshold
                    refinement_factor = 0.1
                else:
                    refinement_factor = 0.3
                    
                # Apply small adjustment based on SNR
                refined_beam = predicted_beam * (1 + refinement_factor * tf.random.normal(predicted_beam.shape, mean=0.0, stddev=0.1))
                
                # Ensure beam stays within valid ranges
                refined_beam = tf.clip_by_value(refined_beam, 
                                            clip_value_min=[-60.0, -30.0],
                                            clip_value_max=[60.0, 30.0])
                
                return refined_beam
            return predicted_beam
            
        except Exception as e:
            logger.error(f"Error in beam refinement: {str(e)}")
            return predicted_beam

    def _combine_multipath(self, refined_beam, channel_data):
        """
        Combines multiple path contributions for final beam direction
        
        Args:
            refined_beam: Refined beam direction
            channel_data: Channel state information
            
        Returns:
            tf.Tensor: Final beam direction considering multipath
        """
        try:
            if refined_beam is None:
                return self.current_beam if self.current_beam is not None else tf.zeros([2])
                
            # Check if we have path information
            if 'path_data' in channel_data:
                paths = channel_data['path_data']
                
                # Weight different paths based on their strength
                path_weights = tf.nn.softmax(paths['path_powers'])
                weighted_directions = tf.reduce_sum(paths['path_directions'] * path_weights[:, tf.newaxis], axis=0)
                
                # Combine with refined beam
                alpha = 0.7  # Weight for refined beam vs multipath combination
                combined_beam = alpha * refined_beam + (1 - alpha) * weighted_directions
                
                # Ensure combined beam is within valid ranges
                combined_beam = tf.clip_by_value(combined_beam,
                                            clip_value_min=[-60.0, -30.0],
                                            clip_value_max=[60.0, 30.0])
                
                return combined_beam
            return refined_beam
            
        except Exception as e:
            logger.error(f"Error in multipath combination: {str(e)}")
            return refined_beam