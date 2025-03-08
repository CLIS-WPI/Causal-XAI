# src/beam_manager.py
import mitsuba
from mitsuba import ScalarTransform4f, Bitmap
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
        self.current_beam = tf.zeros((config.num_agvs, 2), dtype=tf.float32)  # Initialized as tensor
        self.packet_stats = {
            'total': 0,
            'successful': 0,
            'failed_during_switch': 0
        }
        self.current_switch_start = None

        # Initialize beam codebook
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
            'old_beam': tf.identity(old_beam).numpy(),  # Tensor-safe copy
            'new_beam': tf.identity(new_beam).numpy()   # Tensor-safe copy
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
            'value': float(snr_value)  # Ensure scalar
        })

    def get_performance_metrics(self):
        total = max(1, self.packet_stats['total'])
        return {
            'switch_times': self.switch_times,
            'packet_success_rate': self.packet_stats['successful'] / total,
            'switch_failure_rate': self.packet_stats['failed_during_switch'] / total,
            'snr_history': self.snr_history
        }  
        
    def _initialize_beam_codebook(self):
        total_beams = self.config.beamforming['num_beams']  # e.g., 512
        num_rows = self.config.bs_array['num_rows']  # e.g., 32
        num_cols = self.config.bs_array['num_cols']  # e.g., 32
        max_steering_angle = float(self.config.beamforming['max_steering_angle'])  # Ensure float, e.g., 60.0
        
        num_azimuth = num_rows  # 32
        num_elevation = total_beams // num_azimuth  # 16
        
        # Cast inputs to float32 explicitly
        start_az = tf.cast(-max_steering_angle, tf.float32)
        stop_az = tf.cast(max_steering_angle, tf.float32)
        start_el = tf.cast(-max_steering_angle / 2, tf.float32)
        stop_el = tf.cast(max_steering_angle / 2, tf.float32)
        
        azimuth_angles = tf.linspace(start_az, stop_az, num_azimuth)  # No dtype arg
        elevation_angles = tf.linspace(start_el, stop_el, num_elevation)  # No dtype arg
        
        azimuth_grid, elevation_grid = tf.meshgrid(azimuth_angles, elevation_angles)
        self.beam_codebook = tf.stack([azimuth_grid, elevation_grid], axis=-1)
        self.beam_codebook = tf.reshape(self.beam_codebook, [-1, 2])  # Shape: (512, 2)
        
        assert len(self.beam_codebook) == total_beams, f"Beam codebook size {len(self.beam_codebook)} does not match {total_beams}"
        logger.info(f"DFT Beam codebook initialized with {len(self.beam_codebook)} beams, reflecting {num_rows}x{num_cols} array")
    
    def get_current_beams(self):
        return self.current_beam  # Returns tensor
    
    def should_switch_beam(self, current_snr, proposed_snr, agv_position):
        MIN_SNR_THRESHOLD = self.config.beamforming['min_snr_threshold']
        SNR_DROP_THRESHOLD = self.config.beamforming['beam_switching']['switching_threshold']
        DISTANCE_THRESHOLD = 10.0
        
        bs_pos = tf.cast(self.config.bs_position, tf.float32)
        agv_pos = tf.cast(agv_position, tf.float32)
        distance = tf.norm(agv_pos - bs_pos)
        
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
        try:
            return [tf.identity(beam).numpy() for beam in self.beam_history]  # Tensor-safe copies
        except Exception as e:
            logger.error(f"Error getting beam history: {str(e)}")
            return []
        
    def update_beam(self, new_beam, success=True, channel_data=None):
        """Update beam state and log metrics"""
        try:
            if self.current_beam is None:
                self.current_beam = tf.zeros([self.config.num_agvs, 2], dtype=tf.float32)
            
            old_beam = tf.identity(self.current_beam)
            new_beam = tf.convert_to_tensor(new_beam, dtype=tf.float32)
            
            if new_beam.shape != [self.config.num_agvs, 2]:
                raise ValueError(f"new_beam shape {new_beam.shape} does not match expected ({self.config.num_agvs}, 2)")
            
            if not tf.reduce_all(tf.equal(new_beam, old_beam)):
                self.log_beam_switch(old_beam, new_beam)
                self.current_beam = new_beam
                logger.info(f"Beam updated for AGVs: {new_beam.numpy()}")
            else:
                logger.debug("No beam change needed")
            
            self.beam_history.append(self.current_beam.numpy())
            
            if channel_data is not None:
                self.current_channel_state = channel_data  # Update channel state
            
            if success is not None:
                self.update_packet_stats(success, during_switch=self.has_switch_occurred())
                snr_value = self.calculate_snr(self.current_beam, channel_data)
                self.snr_history.append({
                    'timestamp': time.time(),
                    'value': float(snr_value),
                    'success': success
                })
            
            self.beam_metrics = {
                'beam_directions': self.current_beam,
                'update_time': time.time()
            }
        except Exception as e:
            logger.error(f"Error updating beam: {str(e)}")
            raise
        
    def has_switch_occurred(self):
        if len(self.beam_history) < 1:
            return False
        current = self.current_beam
        previous = tf.convert_to_tensor(self.beam_history[-1], dtype=tf.float32)
        return not tf.reduce_all(tf.abs(current - previous) < 1e-5)

    def detect_blockage(self, channel_data, agv_positions, obstacle_positions):
        try:
            agv_positions = tf.ensure_shape(tf.convert_to_tensor(agv_positions, dtype=tf.float32), [self.config.num_agvs, 3])
            obstacle_positions = tf.convert_to_tensor(obstacle_positions, dtype=tf.float32)
            bs_pos = tf.cast(self.config.bs_position, tf.float32)

            def check_ray_intersection(agv_pos):
                ray_start = bs_pos[tf.newaxis, :]
                ray_end = agv_pos[tf.newaxis, :]
                ray_dir = ray_end - ray_start
                ray_length = tf.norm(ray_dir)
                ray_dir = ray_dir / (ray_length + tf.keras.backend.epsilon())
                
                to_obstacles = obstacle_positions - ray_start
                projections = tf.reduce_sum(to_obstacles * ray_dir, axis=1)
                closest_points = ray_start + projections[:, tf.newaxis] * ray_dir
                distances = tf.norm(obstacle_positions - closest_points, axis=1)
                is_between = tf.logical_and(projections >= 0, projections <= ray_length)
                intersects = tf.logical_and(distances < self.config.beamforming['obstacle_radius'], is_between)
                return tf.reduce_any(intersects)

            direct_blocked = tf.map_fn(check_ray_intersection, agv_positions, dtype=tf.bool)
            logger.debug(f"Direct blockage status: {direct_blocked.numpy()}")

            snr_blocked = tf.ones([self.config.num_agvs], dtype=tf.bool)
            if channel_data and 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                snr_db = tf.reshape(tf.convert_to_tensor(channel_data['beam_metrics']['snr_db'], dtype=tf.float32), [-1])
                snr_db = tf.ensure_shape(snr_db[:self.config.num_agvs], [self.config.num_agvs])
                snr_blocked = snr_db < self.config.beamforming['min_snr_threshold']
                logger.debug(f"SNR blocked: {snr_blocked.numpy()}")

            los_blocked = tf.logical_or(direct_blocked, snr_blocked)
            logger.debug(f"Final LOS blocked status: {los_blocked.numpy()}")
            return los_blocked
        except Exception as e:
            logger.error(f"Error in blockage detection: {str(e)}", exc_info=True)
            return tf.ones([self.config.num_agvs], dtype=tf.bool)

    def optimize_beam_direction(self, channel_data, path_manager, obstacle_positions):
        try:
            # Get AGV positions from path_manager
            agv_positions = tf.stack([
                tf.convert_to_tensor(path_manager.get_current_status(f'agv_{i}')['position'], dtype=tf.float32)
                if path_manager.get_current_status(f'agv_{i}') is not None and path_manager.get_current_status(f'agv_{i}').get('position') is not None
                else tf.constant([0.0, 0.0, 0.5], dtype=tf.float32)
                for i in range(self.config.num_agvs)
            ])
            logger.debug(f"AGV positions: {agv_positions}")

            # Detect blockages
            blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
            logger.debug(f"Blockage status: {blocked}")

            # Get current SNR from channel_data
            current_snr = tf.zeros([self.config.num_agvs], dtype=tf.float32)
            if channel_data and 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                snr_db = tf.convert_to_tensor(channel_data['beam_metrics']['snr_db'], dtype=tf.float32)
                current_snr = tf.reshape(snr_db, [-1])[:self.config.num_agvs]
            logger.debug(f"Current SNR: {current_snr}")

            # Calculate direct beams (geometric direction to AGVs)
            bs_pos = tf.cast(self.config.bs_position, tf.float32)
            direction_vectors = agv_positions - bs_pos
            horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1)
            azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
            elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
            azimuths = tf.where(azimuths < 0, azimuths + 360, azimuths)
            elevations = tf.clip_by_value(elevations, -30, 30)
            azimuths = tf.clip_by_value(azimuths, -self.config.beamforming['max_steering_angle'], 
                                        self.config.beamforming['max_steering_angle'])
            direct_beams = tf.stack([azimuths, elevations], axis=1)
            logger.debug(f"Direct beams: {direct_beams}")

            # Determine if a beam switch is needed
            min_snr_threshold = self.config.beamforming['min_snr_threshold']
            snr_drop_threshold = self.config.beamforming['beam_switching']['switching_threshold']
            last_snr = self.snr_history[-1]['value'] if self.snr_history else 0.0
            snr_drop = tf.where(last_snr > 0, last_snr - current_snr, tf.zeros_like(current_snr))
            needs_switch = tf.logical_or(
                current_snr < min_snr_threshold,
                snr_drop > snr_drop_threshold
            )
            needs_switch = tf.logical_or(needs_switch, blocked)
            logger.debug(f"Needs switch updated: {needs_switch}")

            # Beam prediction function using SNR across codebook
            def predict_beam(agv_idx):
                try:
                    if not needs_switch[agv_idx]:
                        return direct_beams[agv_idx]  # Use direct beam if no switch needed

                    if not (channel_data and 'channel_matrices' in channel_data):
                        logger.warning(f"AGV {agv_idx}: No valid channel data, falling back to direct beam")
                        return direct_beams[agv_idx]

                    h_freq = tf.convert_to_tensor(channel_data['channel_matrices'][agv_idx], dtype=tf.complex64)  # Shape: (num_subcarriers,)
                    if tf.size(h_freq) == 0:
                        logger.warning(f"AGV {agv_idx}: Empty channel matrix, using direct beam")
                        return direct_beams[agv_idx]

                    # Compute SNR for each beam in the codebook
                    snr_values = []
                    for beam in self.beam_codebook:
                        beam_tiled = tf.tile(beam[tf.newaxis, :], [1, 2])  # Shape: (1, 2)
                        snr = self.calculate_snr(beam_tiled, channel_data)
                        snr_values.append(snr)
                    snr_values = tf.stack(snr_values, axis=0)  # Shape: (num_beams,)
                    
                    if tf.size(snr_values) == 0 or tf.reduce_all(tf.math.is_nan(snr_values)):
                        logger.warning(f"AGV {agv_idx}: Invalid SNR values, using direct beam")
                        return direct_beams[agv_idx]

                    best_idx = tf.argmax(snr_values, axis=0)  # Scalar index
                    best_beam = self.beam_codebook[best_idx]
                    return best_beam

                except Exception as e:
                    logger.error(f"Predict beam failed for AGV {agv_idx}: {str(e)}")
                    return direct_beams[agv_idx]

            # Beam selection and update
            predicted_beams = tf.stack([predict_beam(idx) for idx in range(self.config.num_agvs)], axis=0)
            predicted_beams = tf.cast(predicted_beams, tf.float32)
            logger.debug(f"Stacked predicted beams: {predicted_beams}")

            direct_beams = tf.cast(direct_beams, tf.float32)
            needs_switch_expanded = tf.expand_dims(needs_switch, 1)
            optimal_beams = tf.where(needs_switch_expanded, predicted_beams, direct_beams)

            try:
                optimal_beams = tf.ensure_shape(optimal_beams, [self.config.num_agvs, 2])
                if not tf.reduce_all(tf.equal(self.current_beam, optimal_beams)):
                    self.log_beam_switch(self.current_beam, optimal_beams)
                    self.current_beam = optimal_beams
                self.beam_history.append(self.current_beam.numpy())
            except Exception as e:
                logger.error(f"Error updating beam state: {str(e)}")

            return optimal_beams

        except Exception as e:
            logger.error(f"Error in beam optimization: {str(e)}")
            return self.current_beam  # Assumes current_beam is initialized in __init__


    def calculate_snr(self, beam, channel_data=None):
        """
        Calculate SNR for the given beam direction based on channel data.
        
        Args:
            beam: Tensor of shape [num_agvs, 2] with azimuth and elevation angles (degrees)
            channel_data: Optional channel data dictionary from SmartFactoryChannel
        
        Returns:
            tf.Tensor: Scalar SNR value (average across AGVs)
        """
        try:
            if channel_data is None:
                channel_data = self.current_channel_state
            if channel_data is None or 'channel_matrices' not in channel_data:
                logger.warning("No channel data available, returning default SNR")
                return tf.reduce_mean(tf.ones_like(beam[:, 0]) * 15.0)

            h_freq = tf.convert_to_tensor(channel_data['channel_matrices'], dtype=tf.complex64)
            path_losses = tf.convert_to_tensor(channel_data.get('path_losses', tf.ones([self.config.num_agvs], dtype=tf.float32) * 70.0), dtype=tf.float32)
            tx_power_dbm = self.config.tx_power
            tx_antenna_gain_db = self.config.bs_array['antenna_gain_db']
            rx_antenna_gain_db = self.config.agv_array['antenna_gain_db']
            noise_power = self.config.simulation['noise_power']

            tx_power = tf.pow(10.0, (tx_power_dbm - 30) / 10.0)
            tx_gain = tf.pow(10.0, tx_antenna_gain_db / 10.0)
            rx_gain = tf.pow(10.0, rx_antenna_gain_db / 10.0)
            path_loss_linear = tf.pow(10.0, -path_losses / 10.0)

            channel_power = tf.reduce_mean(tf.abs(h_freq) ** 2, axis=1)

            azimuth_rad = beam[:, 0] * np.pi / 180.0
            elevation_rad = beam[:, 1] * np.pi / 180.0
            beam_gain = tf.abs(tf.cos(azimuth_rad) * tf.cos(elevation_rad))  # Ensure positive, realistic gain
            beam_gain = tf.clip_by_value(beam_gain, 0.1, 1.0)  # Limit gain range

            signal_power = channel_power * tx_power * tx_gain * rx_gain * path_loss_linear * beam_gain
            signal_power = tf.maximum(signal_power, 1e-10)
            noise_power = tf.maximum(noise_power, 1e-20)

            snr_linear = signal_power / noise_power
            snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)
            snr_db = tf.clip_by_value(snr_db, -10.0, 50.0)  # Increase upper limit to see variation

            avg_snr = tf.reduce_mean(snr_db)
            logger.debug(f"Calculated SNR: {avg_snr.numpy()} dB, per AGV: {snr_db.numpy()}")
            return avg_snr
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            return tf.reduce_mean(tf.ones_like(beam[:, 0]) * 15.0)