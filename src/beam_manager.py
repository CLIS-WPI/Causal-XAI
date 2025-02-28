# src/beam_manager.py
import mitsuba
from mitsuba import ScalarTransform4f, Bitmap  # Example import
import tensorflow as tf
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)

class BeamManager:
    def __init__(self, config):
        self.config = config
        self.last_switch_time = None
        self.beam_history = []
        self.snr_history = []
        self.switch_times = []
        self.current_channel_state = None 
        self.channel_state_history = []
        self.current_beam = np.zeros((config.num_agvs, 2), dtype=np.float32)  # [azimuth, elevation] per AGV as NumPy
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
        switch_duration = 0
        if self.current_switch_start is not None:
            switch_duration = switch_time - self.current_switch_start
            self.current_switch_start = None
        else:
            self.current_switch_start = switch_time
        
        switch_data = {
            'timestamp': switch_time,
            'duration': switch_duration,
            'old_beam': old_beam.copy() if isinstance(old_beam, np.ndarray) else tf.identity(old_beam).numpy(),
            'new_beam': new_beam.copy() if isinstance(new_beam, np.ndarray) else tf.identity(new_beam).numpy()
        }
        if self.switch_times and 'reason' in self.switch_times[-1]:
            switch_data['reason'] = self.switch_times[-1]['reason']
        self.switch_times.append(switch_data)
    
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
        total_beams = self.config.beamforming['num_beams']  # 512
        num_rows = self.config.bs_array['num_rows']  # 32
        num_cols = self.config.bs_array['num_cols']  # 32
        max_steering_angle = self.config.beamforming['max_steering_angle']  # 60
        
        # Dynamically calculate number of azimuth and elevation beams to match total_beams
        num_azimuth = 32  # Match num_rows for a reasonable grid
        num_elevation = total_beams // num_azimuth  # 512 / 32 = 16
        
        # Ensure float32 tensors
        azimuth_angles = tf.linspace(tf.cast(-max_steering_angle, tf.float32), 
                                    tf.cast(max_steering_angle, tf.float32), 
                                    num_azimuth)
        elevation_angles = tf.linspace(tf.cast(-max_steering_angle/2, tf.float32), 
                                    tf.cast(max_steering_angle/2, tf.float32), 
                                    num_elevation)
        
        # Use tf.meshgrid for efficiency
        azimuth_grid, elevation_grid = tf.meshgrid(azimuth_angles, elevation_angles)
        self.beam_codebook = tf.stack([azimuth_grid, elevation_grid], axis=-1)
        self.beam_codebook = tf.reshape(self.beam_codebook, [-1, 2])  # Shape: (512, 2)
        
        assert len(self.beam_codebook) == total_beams, f"Beam codebook size {len(self.beam_codebook)} does not match {total_beams}"
        logger.info(f"DFT Beam codebook initialized with {len(self.beam_codebook)} beams, reflecting {num_rows}x{num_cols} array")
    
    def get_current_beams(self):
        """Return the current beam configuration"""
        return self.current_beam  # Already a NumPy array
    
    def should_switch_beam(self, current_snr, proposed_snr, agv_position):
        """Determine if beam switching is needed based on SNR and distance thresholds."""
        MIN_SNR_THRESHOLD = self.config.beamforming['min_snr_threshold']  # e.g., 15.0 dB
        SNR_DROP_THRESHOLD = self.config.beamforming['beam_switching']['switching_threshold']  # e.g., 3.0 dB
        DISTANCE_THRESHOLD = 10.0  # e.g., 10 meters, adjust as needed
        
        bs_pos = np.array(self.config.bs_position, dtype=np.float32)
        distance = np.linalg.norm(agv_position - bs_pos)
        
        reason = None
        if current_snr < MIN_SNR_THRESHOLD:
            reason = f"SNR below threshold: {current_snr:.2f} < {MIN_SNR_THRESHOLD}"
        elif (current_snr - proposed_snr) > SNR_DROP_THRESHOLD:
            reason = f"SNR drop exceeded: {current_snr - proposed_snr:.2f} > {SNR_DROP_THRESHOLD}"
        elif distance > DISTANCE_THRESHOLD:
            reason = f"Distance exceeded: {distance:.2f} > {DISTANCE_THRESHOLD}"
        
        if reason:
            logger.debug(f"Beam switch triggered: {reason}")
            if self.switch_times and 'reason' not in self.switch_times[-1]:
                self.switch_times[-1]['reason'] = reason
            return True
        
        logger.debug(f"No switch needed: SNR {current_snr:.2f}, Distance {distance:.2f}")
        return False
    

    def get_beam_history(self):
        """Return the history of beam configurations"""
        try:
            return [beam.copy() for beam in self.beam_history]  # Return copies of NumPy arrays
        except Exception as e:
            logger.error(f"Error getting beam history: {str(e)}")
            return []
        
    def update_beam(self, new_beam, success=None):
        """
        Update the current beam configuration and track packet stats if success is provided.
        
        Args:
            new_beam: New beam direction (NumPy array or TensorFlow tensor of shape [num_agvs, 2])
            success: Boolean indicating packet success (optional)
        """
        try:
            # Convert current_beam to NumPy if it's a TensorFlow tensor
            if isinstance(self.current_beam, tf.Tensor):
                old_beam = self.current_beam.numpy()
            else:
                old_beam = self.current_beam.copy()
            
            # Convert new_beam to NumPy if it's a TensorFlow tensor
            if isinstance(new_beam, tf.Tensor):
                new_beam = new_beam.numpy()
            elif not isinstance(new_beam, np.ndarray):
                new_beam = np.array(new_beam, dtype=np.float32)
                
            if new_beam.shape != (self.config.num_agvs, 2):
                raise ValueError(f"new_beam shape {new_beam.shape} does not match expected ({self.config.num_agvs}, 2)")
            
            self.beam_history.append(old_beam)
            if not np.allclose(self.current_beam, new_beam, rtol=1e-5, atol=1e-8):
                self.log_beam_switch(old_beam, new_beam)
                logger.info(f"Beam switched from {old_beam} to {new_beam}. Reason: {self.switch_times[-1]['reason'] if self.switch_times else 'Unknown'}")
                if success is not None:
                    during_switch = True
                    self.update_packet_stats(success, during_switch)
            
            self.current_beam = new_beam.copy()
            logger.debug(f"Beam updated - Old: {old_beam}, New: {new_beam}")
            
            self.beam_metrics = {
                'beam_directions': self.current_beam,
                'update_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error updating beam: {str(e)}")
            raise
        
    def has_switch_occurred(self):
        """Check if a beam switch has occurred by comparing current and previous beam configurations"""
        if len(self.beam_history) < 1:
            return False
        current = self.current_beam
        previous = self.beam_history[-1]
        return not np.allclose(current, previous, rtol=1e-5, atol=1e-8)
            
    def detect_blockage(self, channel_data, agv_positions, obstacle_positions):
        try:
            agv_positions = tf.convert_to_tensor(agv_positions, dtype=tf.float32)
            obstacle_positions = tf.convert_to_tensor(obstacle_positions, dtype=tf.float32)
            bs_pos = tf.cast(self.config.bs_position, tf.float32)

            def check_ray_intersection(agv_pos):
                ray_start = bs_pos[tf.newaxis, :]
                ray_end = agv_pos[tf.newaxis, :]
                ray_dir = ray_end - ray_start
                ray_length = tf.norm(ray_dir)
                ray_dir = ray_dir / ray_length
                to_obstacles = obstacle_positions - ray_start
                projections = tf.reduce_sum(to_obstacles * ray_dir, axis=1)
                closest_points = ray_start + projections[:, tf.newaxis] * ray_dir
                distances = tf.norm(obstacle_positions - closest_points, axis=1)
                is_between = tf.logical_and(projections >= 0, projections <= ray_length)
                obstacle_radius = self.config.beamforming['obstacle_radius']
                intersects = tf.logical_and(distances < obstacle_radius, is_between)
                return tf.reduce_any(intersects)

            direct_blocked = tf.map_fn(check_ray_intersection, agv_positions, dtype=tf.bool)
            logger.debug(f"Direct blockage status: {direct_blocked.numpy()}")

            num_agvs = agv_positions.shape[0]
            snr_blocked = tf.ones([num_agvs], dtype=tf.bool)  # Default to blocked if no SNR
            if channel_data and 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                snr_threshold = self.config.beamforming['min_snr_threshold']
                snr_db = tf.convert_to_tensor(channel_data['beam_metrics']['snr_db'], dtype=tf.float32)
                logger.debug(f"SNR dB from channel data: {snr_db.numpy()}")
                # Flatten snr_db to [num_agvs]
                snr_db = tf.reshape(snr_db, [-1])[:num_agvs]  # Take first num_agvs elements
                snr_blocked = tf.cast(snr_db < snr_threshold, tf.bool)
                logger.debug(f"SNR blocked: {snr_blocked.numpy()}")

            los_blocked = tf.logical_or(direct_blocked, snr_blocked)
            logger.debug(f"Final LOS blocked status: {los_blocked.numpy()}")
            return los_blocked
        except Exception as e:
            logger.error(f"Error in blockage detection: {str(e)}")
            return tf.ones([len(agv_positions)], dtype=tf.bool)

    def _find_reflection_path(self, agv_pos, obstacle_positions, channel_data):
        try:
            direct_beam = self._calculate_direct_beam(agv_pos)
            best_beam = None
            best_metric = -float('inf')
            
            for beam in self.beam_codebook:
                if tf.norm(beam - direct_beam) < self.config.beamforming['min_angular_separation']:
                    continue
                    
                metric = self._calculate_path_metric(beam, agv_pos, obstacle_positions, channel_data)
                
                if metric > best_metric:
                    best_metric = metric
                    best_beam = beam
            
            return best_beam if best_beam is not None else direct_beam
            
        except Exception as e:
            logger.error(f"Error finding reflection path: {str(e)}")
            return direct_beam
    
    def _calculate_path_metric(self, beam, agv_pos, obstacle_positions, channel_data):
        try:
            azimuth, elevation = beam[0], beam[1]
            direction = self._angles_to_vector(azimuth, elevation)
            
            for obstacle in obstacle_positions:
                if self._ray_intersects_obstacle(self.config.bs_position, self.config.bs_position + direction * 20.0, obstacle):
                    return -float('inf')
            
            metric = 0.0
            if channel_data and 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                metric += np.mean(channel_data['beam_metrics']['snr_db'])
            
            metric -= self.config.beamforming['steering_penalty'] * tf.norm(beam)
            return float(metric)
            
        except Exception as e:
            logger.error(f"Error calculating path metric: {str(e)}")
            return -float('inf')
    
    def optimize_beam_direction(self, channel_data, path_manager, obstacle_positions):
        try:
            agv_positions = tf.stack([
                tf.convert_to_tensor(path_manager.get_current_status(f'agv_{i}')['position'], dtype=tf.float32)
                if path_manager.get_current_status(f'agv_{i}')['position'] is not None
                else tf.constant([0.0, 0.0, 0.5], dtype=tf.float32)
                for i in range(self.config.num_agvs)
            ])
            logger.debug(f"AGV positions: {agv_positions}")

            blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
            logger.debug(f"Blockage status: {blocked}")

            if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                snr_db = tf.convert_to_tensor(channel_data['beam_metrics']['snr_db'], dtype=tf.float32)
                if len(snr_db.shape) > 1:
                    current_snr = tf.reduce_mean(snr_db, axis=1)
                else:
                    # Fix tensor shape mismatch
                    if tf.rank(snr_db) == 0:  # Scalar value
                        current_snr = tf.fill([self.config.num_agvs], snr_db)
                    else:
                        current_snr = tf.broadcast_to(snr_db, [self.config.num_agvs])
            else:
                current_snr = tf.zeros([self.config.num_agvs], dtype=tf.float32)
            logger.debug(f"Current SNR: {current_snr}")

            bs_pos = tf.cast(self.config.bs_position, tf.float32)
            direction_vectors = agv_positions - bs_pos
            azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
            horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1)
            elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
            azimuths = tf.where(azimuths < 0, azimuths + 360, azimuths)
            elevations = tf.clip_by_value(elevations, -30, 30)
            azimuths = tf.clip_by_value(azimuths, -self.config.beamforming['max_steering_angle'], 
                                        self.config.beamforming['max_steering_angle'])
            direct_beams = tf.stack([azimuths, elevations], axis=1)
            logger.debug(f"Direct beams: {direct_beams}")

            proposed_snr = current_snr  # Placeholder, could be enhanced later

            optimal_beams = []
            for i in range(self.config.num_agvs):
                needs_switch = self.should_switch_beam(current_snr[i].numpy(), proposed_snr[i].numpy(), agv_positions[i].numpy())
                if blocked[i] or needs_switch:
                    best_beam = self._find_reflection_path(agv_positions[i], obstacle_positions, channel_data)
                    logger.debug(f"AGV {i} needs alternative path - SNR: {current_snr[i]}, Blocked: {blocked[i]}")
                else:
                    best_beam = direct_beams[i]
                    logger.debug(f"AGV {i} using direct path - SNR: {current_snr[i]}")
                optimal_beams.append(best_beam)

            optimal_beams = tf.stack(optimal_beams)
            if not np.allclose(self.current_beam, optimal_beams.numpy(), rtol=1e-5, atol=1e-8):
                self.log_beam_switch(self.current_beam, optimal_beams.numpy())
            self.current_beam = optimal_beams.numpy()
            self.beam_history.append(self.current_beam)

            return optimal_beams

        except Exception as e:
            logger.error(f"Error in beam optimization: {str(e)}")
            return tf.zeros([self.config.num_agvs, 2], dtype=tf.float32)

    def _ray_intersects_obstacle(self, start_point, end_point, obstacle_position):
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
            obstacle_radius = self.config.beamforming['obstacle_radius']
            
            return tf.logical_and(distance_to_ray < obstacle_radius, is_between)
            
        except Exception as e:
            logger.error(f"Error in ray intersection check: {str(e)}")
            return True
    
    def _calculate_direct_beam(self, agv_position):
        try:
            agv_pos = tf.cast(agv_position, tf.float32)
            bs_pos = tf.cast(self.config.bs_position, tf.float32)
            
            direction_vector = agv_pos - bs_pos
            azimuth = tf.math.atan2(direction_vector[1], direction_vector[0])
            horizontal_distance = tf.norm(direction_vector[:2])
            elevation = tf.math.atan2(direction_vector[2], horizontal_distance)
            
            azimuth_deg = azimuth * 180.0 / np.pi
            elevation_deg = elevation * 180.0 / np.pi
            
            azimuth_deg = tf.where(azimuth_deg < 0, azimuth_deg + 360, azimuth_deg)
            elevation_deg = tf.clip_by_value(elevation_deg, -30, 30)
            azimuth_deg = tf.clip_by_value(azimuth_deg, -self.config.beamforming['max_steering_angle'], 
                                          self.config.beamforming['max_steering_angle'])

            return tf.stack([azimuth_deg, elevation_deg])
            
        except Exception as e:
            logger.error(f"Error calculating direct beam: {str(e)}")
            return tf.constant([0.0, 0.0])

    def _angles_to_vector(self, azimuth, elevation):
        try:
            azimuth_rad = azimuth * np.pi / 180.0
            elevation_rad = elevation * np.pi / 180.0
            
            x = tf.cos(elevation_rad) * tf.cos(azimuth_rad)
            y = tf.cos(elevation_rad) * tf.sin(azimuth_rad)
            z = tf.sin(elevation_rad)
            
            return tf.stack([x, y, z])
            
        except Exception as e:
            logger.error(f"Error converting angles to vector: {str(e)}")
            return tf.constant([1.0, 0.0, 0.0])

    def optimize_beam_direction(self, channel_data, path_manager, obstacle_positions):
        """
        Optimize beam directions for all AGVs using vectorized operations.
        
        Args:
            channel_data (dict): Channel data with SNR metrics and path info.
            path_manager (AGVPathManager): Manager for AGV positions and statuses.
            obstacle_positions (tf.Tensor): Obstacle positions [num_obstacles, 3].
        
        Returns:
            tf.Tensor: Optimized beam directions [num_agvs, 2].
        """
        try:
            # Get AGV positions as a tensor
            agv_positions = tf.stack([
                tf.convert_to_tensor(path_manager.get_current_status(f'agv_{i}')['position'], dtype=tf.float32)
                if path_manager.get_current_status(f'agv_{i}')['position'] is not None
                else tf.constant([0.0, 0.0, 0.5], dtype=tf.float32)
                for i in range(self.config.num_agvs)
            ])  # [num_agvs, 3]
            logger.debug(f"AGV positions: {agv_positions}")

            # Detect blockages vectorized
            blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
            logger.debug(f"Blockage status: {blocked}")

            # Extract SNR tensor
            if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                snr_db = tf.convert_to_tensor(channel_data['beam_metrics']['snr_db'], dtype=tf.float32)
                if len(snr_db.shape) > 1:
                    current_snr = tf.reduce_mean(snr_db, axis=1)  # [num_agvs]
                else:
                    if tf.rank(snr_db) == 0:  # Scalar value
                        current_snr = tf.fill([self.config.num_agvs], snr_db)
                    else:
                        current_snr = tf.broadcast_to(snr_db, [self.config.num_agvs])
            else:
                current_snr = tf.zeros([self.config.num_agvs], dtype=tf.float32)
            logger.debug(f"Current SNR: {current_snr}")

            # Calculate direct beams vectorized
            bs_pos = tf.cast(self.config.bs_position, tf.float32)  # [3]
            direction_vectors = agv_positions - bs_pos  # [num_agvs, 3]
            azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
            horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1)
            elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
            azimuths = tf.where(azimuths < 0, azimuths + 360, azimuths)
            elevations = tf.clip_by_value(elevations, -30, 30)
            azimuths = tf.clip_by_value(azimuths, -self.config.beamforming['max_steering_angle'], 
                                    self.config.beamforming['max_steering_angle'])
            direct_beams = tf.stack([azimuths, elevations], axis=1)  # [num_agvs, 2]
            logger.debug(f"Direct beams: {direct_beams}")

            # Vectorized switch condition
            min_snr_threshold = self.config.beamforming['min_snr_threshold']
            snr_drop_threshold = self.config.beamforming['beam_switching']['switching_threshold']
            needs_switch = tf.logical_or(
                current_snr < min_snr_threshold,
                (current_snr - current_snr) > snr_drop_threshold  # Compare with current SNR
            )
            logger.debug(f"Needs switch: {needs_switch}")

            # Enhanced beam prediction with fallback checks
            def predict_beam(agv_idx):
                logger.debug(f"predict_beam: Processing for agv_idx={agv_idx}")
                
                # First check for valid path data
                if 'path_data' not in channel_data or 'path_directions' not in channel_data['path_data']:
                    logger.warning("No path data available, using direct beam")
                    return direct_beams[agv_idx]
                
                # Then check for valid power values
                path_powers = channel_data['path_data']['path_powers'][0, agv_idx]
                logger.debug(f"path_powers shape: {path_powers.shape}, value: {path_powers[:5]}")
                if tf.reduce_sum(path_powers) == 0:
                    logger.warning("All path powers are zero, using direct beam")
                    return direct_beams[agv_idx]
                
                try:
                    # درست گرفتن directions برای AGV مورد نظر
                    directions = channel_data['path_data']['path_directions'][agv_idx]  # [theta, phi] برای AGV
                    powers_tensor = channel_data['path_data']['path_powers'][0, agv_idx]
                    logger.debug(f"directions shape: {directions.shape}, value: {directions}")
                    logger.debug(f"powers_tensor shape: {powers_tensor.shape}, value: {powers_tensor[:5]}")
                    
                    # Dimension handling برای powers
                    if tf.rank(powers_tensor) > 1:  # اگه بیشتر از یه بعد داره
                        valid_dims = [i for i in range(tf.rank(powers_tensor)) if tf.shape(powers_tensor)[i] > 1]
                        powers = tf.reduce_mean(powers_tensor, axis=valid_dims) if valid_dims else powers_tensor
                    elif tf.rank(powers_tensor) == 1:  # اگه فقط یه بعده
                        powers = powers_tensor
                    else:  # اگه اسکالره
                        logger.warning("Powers is scalar, converting to single-element array")
                        powers = tf.expand_dims(powers_tensor, 0)
                    
                    logger.debug(f"powers after reduction: shape={powers.shape}, value={powers}")
                    
                    if tf.size(powers) == 0:
                        logger.warning("Empty powers tensor, using direct beam")
                        return direct_beams[agv_idx]
                    
                    # اگه powers آرایه‌ست، بهترین مسیر رو پیدا کن
                    best_idx = tf.argmax(powers)
                    logger.debug(f"best_idx: {best_idx}")
                    
                    # directions باید دو مقدار داشته باشه (theta و phi)
                    if tf.rank(directions) == 1 and tf.shape(directions)[0] == 2:
                        best_dir = directions  # مستقیم استفاده می‌کنیم چون فقط یه مسیر داریم
                    else:
                        logger.warning("Invalid directions shape, using direct beam")
                        return direct_beams[agv_idx]
                    
                    logger.debug(f"best_dir: {best_dir}")
                    
                    # Convert to degrees with safety checks
                    azimuth_deg = tf.clip_by_value(best_dir[0] * 180.0 / np.pi, 
                                                -self.config.beamforming['max_steering_angle'], 
                                                self.config.beamforming['max_steering_angle'])
                    elevation_deg = tf.clip_by_value(best_dir[1] * 180.0 / np.pi,
                                                    -self.config.beamforming['max_steering_angle']/2,
                                                    self.config.beamforming['max_steering_angle']/2)
                    
                    return tf.stack([azimuth_deg, elevation_deg])
                
                except Exception as e:
                    logger.error(f"Error processing AGV {agv_idx}: {str(e)}")
                    return direct_beams[agv_idx]

            # Process beams with fallback protection
            predicted_beams = []
            for idx in range(self.config.num_agvs):
                predicted_beam = predict_beam(idx)
                predicted_beams.append(predicted_beam)

            predicted_beams = tf.stack(predicted_beams, axis=0, name="stacked_predicted_beams")  # Shape: [2, 2]
            predicted_beams = tf.cast(predicted_beams, tf.float32)  # مطمئن شو که float32 باشه
            logger.debug(f"Stacked predicted beams: {predicted_beams}")

            # مطمئن شو که direct_beams هم float32 باشه
            direct_beams = tf.cast(direct_beams, tf.float32)

            # Final beam selection logic
            optimal_beams = tf.where(
                tf.expand_dims(needs_switch, 1),
                predicted_beams,
                direct_beams,
                name="optimal_beams_selection"
            )

            # Update beam state only if changed
            if not tf.reduce_all(tf.equal(self.current_beam, optimal_beams)):
                self.log_beam_switch(self.current_beam, optimal_beams)
                self.current_beam = optimal_beams

            self.beam_history.append(self.current_beam)

            return optimal_beams
        except Exception as e:
            logger.error(f"Error in beam optimization: {str(e)}")
            return self.current_beam if self.current_beam is not None else direct_beams
            
    def _predict_optimal_beam(self, channel_data, agv_index):
            """Predict optimal beam using path_directions directly as [azimuth, elevation]"""
            try:
                if 'path_data' not in channel_data or 'path_directions' not in channel_data['path_data']:
                    logger.warning(f"No path data for AGV {agv_index}, returning current beam")
                    return self.current_beam[agv_index] if self.current_beam is not None else tf.zeros([2])

                # path_directions: [1, num_agvs, num_paths, 2] (azimuth, elevation in radians)
                directions = channel_data['path_data']['path_directions'][0, agv_index]  # [num_paths, 2]
                powers = tf.reduce_mean(channel_data['path_data']['path_powers'][0, agv_index], axis=[0, 1, 2])  # [num_paths]

                best_path_idx = tf.argmax(powers)  # Index of strongest path
                best_direction = directions[best_path_idx]  # [azimuth, elevation] in radians

                # Convert to degrees
                azimuth_deg = best_direction[0] * 180.0 / np.pi
                elevation_deg = best_direction[1] * 180.0 / np.pi

                predicted_beam = tf.stack([azimuth_deg, elevation_deg])
                logger.debug(f"Predicted beam for AGV {agv_index}: {predicted_beam}")

                return tf.clip_by_value(
                    predicted_beam,
                    [-self.config.beamforming['max_steering_angle'], -self.config.beamforming['max_steering_angle']/2],
                    [self.config.beamforming['max_steering_angle'], self.config.beamforming['max_steering_angle']/2]
                )

            except Exception as e:
                logger.error(f"Error in beam prediction: {str(e)}")
                return self.current_beam[agv_index] if self.current_beam is not None else tf.zeros([2])
    
    def _refine_beam(self, predicted_beam, channel_data, agv_index=None):
        """
        Refines the predicted beam direction based on channel conditions
        
        Args:
            predicted_beam: Initial beam prediction
            channel_data: Channel state information
            agv_index: Index of the AGV being processed
            
        Returns:
            tf.Tensor: Refined beam direction
        """
        try:
            if predicted_beam is None:
                return self.current_beam if self.current_beam is not None else tf.zeros([2])
                
            # Apply refinement based on channel quality
            if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                snr_data = channel_data['beam_metrics']['snr_db']
                if agv_index is not None and isinstance(snr_data, (np.ndarray, tf.Tensor)) and len(snr_data.shape) > 1:
                    current_snr = tf.reduce_mean(snr_data[agv_index])
                else:
                    current_snr = tf.reduce_mean(snr_data)
                
                # If SNR is good, make smaller adjustments
                if current_snr > self.config.beamforming['good_snr_threshold']:
                    refinement_factor = self.config.beamforming['refinement_factor_good']
                else:
                    refinement_factor = self.config.beamforming['refinement_factor_poor']
                    
                # Apply small adjustment based on SNR
                refined_beam = predicted_beam * (1 + refinement_factor * tf.random.normal(predicted_beam.shape, mean=0.0, stddev=0.1))
                
                # Ensure beam stays within valid ranges
                refined_beam = tf.clip_by_value(refined_beam, 
                                clip_value_min=[-self.config.beamforming['max_steering_angle'], -self.config.beamforming['max_steering_angle']/2], 
                                clip_value_max=[self.config.beamforming['max_steering_angle'], self.config.beamforming['max_steering_angle']/2])
                
                return refined_beam
            return predicted_beam
            
        except Exception as e:
            logger.error(f"Error in beam refinement: {str(e)}")
            return predicted_beam
        
    def _combine_multipath(self, refined_beam, channel_data, agv_index):
        """Combine multipath components for final beam direction"""
        try:
            return refined_beam  # For now, just return the refined beam
        except Exception as e:
            logger.error(f"Error in multipath combination: {str(e)}")
            return refined_beam

    