import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import numpy as np
import time
from typing import Any, Dict, Optional

# --- NEW IMPORT ---
# Adjust path based on your Sionna version (phy.mimo or just mimo?)
try:
    # Try Sionna 1.0+ style import
    from sionna.phy.mimo import grid_of_beams_dft
    logger.debug("Imported grid_of_beams_dft from sionna.phy.mimo")
except ImportError:
    try:
        from sionna.mimo import grid_of_beams_dft
        logger.debug("Imported grid_of_beams_dft from sionna.mimo")
    except ImportError:
        logger.error("Could not import grid_of_beams_dft from Sionna. Check your installation/version.")
        def grid_of_beams_dft(**kwargs: Any) -> tf.Tensor:
            logger.warning("Using dummy grid_of_beams_dft!")
            num_tx = kwargs.get('num_ant_v', 1) * kwargs.get('num_ant_h', 1)
            num_beams = num_tx  # Simple fallback
            vecs = tf.complex(tf.random.normal([num_beams, num_tx]), tf.random.normal([num_beams, num_tx]))
            return vecs / tf.norm(vecs, axis=1, keepdims=True)
# --- END NEW IMPORT ---


class BeamManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the beam manager.

        Args:
            config: Configuration dictionary containing beamforming parameters.
        """
        self.config: Dict[str, Any] = config
        self.last_switch_time: Optional[float] = None
        self.beam_history: list = []  # Will store angles as numpy arrays
        self.snr_history: list = []  # Will store average SNR dB
        self.switch_times: list = []
        self.current_channel_state: Optional[Dict[str, Any]] = None
        self.channel_state_history: list = []
        # Stores current beam angles [azimuth, elevation] per AGV
        self.current_beam_angles: tf.Tensor = tf.zeros((config["num_agvs"], 2), dtype=tf.float32)
        self.packet_stats: Dict[str, int] = {
            'total': 0,
            'successful': 0,
            'failed_during_switch': 0
        }
        self.current_switch_start: Optional[float] = None

        # Validate configuration
        self.validate_config()

        # --- Initialize beam codebook (vectors and angles) ---
        self.beam_vector_codebook: Optional[tf.Tensor] = None  # Holds complex vectors (w)
        self.angle_codebook: Optional[tf.Tensor] = None  # Holds corresponding angles [az, el]
        self.codebook_size: int = 0
        self._initialize_beam_codebook()
        # --- END INITIALIZATION ---

    def validate_config(self) -> None:
        """Validate the beamforming configuration parameters."""
        required = ['num_beams', 'max_steering_angle', 'min_snr_threshold']
        if not all(k in self.config.get("beamforming", {}) for k in required):
            raise ValueError(f"Missing required beamforming config keys: {required}")
        if "bs_array" not in self.config or "num_agvs" not in self.config:
            raise ValueError("Configuration must include 'bs_array' and 'num_agvs'")
        logger.debug("BeamManager configuration validated.")

    def _initialize_beam_codebook(self) -> None:
        """Initializes the beamforming vector codebook and corresponding angle codebook."""
        try:
            logger.info("Initializing Beamforming Vector & Angle Codebooks...")
            # Cast bs_array dimensions to integers and assert positive values
            num_rows = tf.cast(self.config["bs_array"]['num_rows'], tf.int32)
            num_cols = tf.cast(self.config["bs_array"]['num_cols'], tf.int32)
            tf.debugging.assert_positive(num_rows, message="Number of rows must be positive")
            tf.debugging.assert_positive(num_cols, message="Number of columns must be positive")
            num_tx_ant = int(num_rows.numpy()) * int(num_cols.numpy())
            num_beams_config = self.config["beamforming"]['num_beams']

            # --- Generate Beamforming Vectors (w) using Sionna function ---
            self.beam_vector_codebook = grid_of_beams_dft(
                num_ant_v=int(num_rows.numpy()),
                num_ant_h=int(num_cols.numpy()),
                normalize=True
            )
            self.beam_vector_codebook = tf.cast(self.beam_vector_codebook, dtype=tf.complex64)
            self.codebook_size = int(self.beam_vector_codebook.shape[0])
            logger.info(f"Generated beam vector codebook with shape: {self.beam_vector_codebook.shape}")

            # --- Generate Corresponding Angle Codebook (azimuth, elevation) ---
            max_steering_angle = float(self.config["beamforming"]['max_steering_angle'])
            num_codebook_beams = self.codebook_size
            # Assume a roughly square grid for angles
            num_az_approx = int(np.ceil(np.sqrt(num_codebook_beams)))
            num_el_approx = int(np.ceil(num_codebook_beams / num_az_approx))
            azimuth_angles = tf.linspace(-max_steering_angle, max_steering_angle, num_az_approx)
            # Typical UPA: elevation range is half the azimuth range
            elevation_angles = tf.linspace(-max_steering_angle / 2.0, max_steering_angle / 2.0, num_el_approx)
            az_grid, el_grid = tf.meshgrid(azimuth_angles, elevation_angles)
            angle_codebook_flat = tf.stack([tf.reshape(az_grid, [-1]), tf.reshape(el_grid, [-1])], axis=-1)
            self.angle_codebook = angle_codebook_flat[:self.codebook_size, :]
            self.angle_codebook = tf.cast(self.angle_codebook, tf.float32)
            if self.angle_codebook.shape[0] != self.codebook_size:
                logger.warning(f"Angle codebook size ({self.angle_codebook.shape[0]}) mismatch with vector codebook size ({self.codebook_size}).")
                self.angle_codebook = tf.zeros([self.codebook_size, 2], dtype=tf.float32)
            logger.info(f"Generated angle codebook with shape: {self.angle_codebook.shape}")

        except ImportError:
            logger.error("Failed to import grid_of_beams_dft. Using random fallback.")
            num_tx_ant = self.config["bs_array"]['num_rows'] * self.config["bs_array"]['num_cols']
            num_beams = self.config["beamforming"]['num_beams']
            self.beam_vector_codebook = tf.complex(
                tf.random.normal([num_beams, num_tx_ant]),
                tf.random.normal([num_beams, num_tx_ant])
            )
            self.beam_vector_codebook /= tf.norm(self.beam_vector_codebook, axis=1, keepdims=True)
            self.angle_codebook = tf.random.uniform([num_beams, 2], minval=-60, maxval=60, dtype=tf.float32)
            self.codebook_size = num_beams
            logger.warning("Using random fallback for beam vector and angle codebooks.")
        except Exception as e:
            logger.error(f"Error initializing beam codebooks: {str(e)}", exc_info=True)
            raise

    def log_beam_switch(self, old_beam_angles: tf.Tensor, new_beam_angles: tf.Tensor) -> None:
        """Log the event of switching beams."""
        switch_time = time.time()
        switch_duration = 0.0
        if self.current_switch_start is not None:
            switch_duration = switch_time - self.current_switch_start
            self.current_switch_start = None
        switch_data = {
            'timestamp': switch_time,
            'duration': switch_duration,
            'old_beam': tf.identity(old_beam_angles).numpy(),
            'new_beam': tf.identity(new_beam_angles).numpy()
        }
        self.switch_times.append(switch_data)
        logger.info(f"Beam switch logged: {old_beam_angles.numpy()} -> {new_beam_angles.numpy()}")

    def update_packet_stats(self, success: bool, during_switch: bool = False) -> None:
        self.packet_stats['total'] += 1
        if success:
            self.packet_stats['successful'] += 1
        elif during_switch:
            self.packet_stats['failed_during_switch'] += 1

    def log_snr(self, snr_value_db: float) -> None:
        """Log the SNR value along with a timestamp."""
        self.snr_history.append({
            'timestamp': time.time(),
            'value': snr_value_db
        })

    def get_performance_metrics(self) -> Dict[str, Any]:
        total = max(1, self.packet_stats['total'])
        return {
            'switch_times': self.switch_times,
            'packet_success_rate': self.packet_stats['successful'] / total,
            'switch_failure_rate': self.packet_stats['failed_during_switch'] / total,
            'snr_history': self.snr_history
        }

    @tf.function
    def calculate_snr(self, h_mimo: tf.Tensor, beamforming_vector_w: tf.Tensor,
                      tx_power_linear: tf.Tensor, noise_power_linear: tf.Tensor) -> tf.Tensor:
        """
        Calculates the SNR for a given beamforming vector w and MIMO channel H.
        
        Args:
            h_mimo: MIMO channel matrix [num_rx_ant, num_tx_ant, num_subcarriers].
            beamforming_vector_w: Beamforming vector [num_tx_ant] or [num_tx_ant, 1].
            tx_power_linear: Transmit power in linear scale.
            noise_power_linear: Noise power in linear scale.
            
        Returns:
            A scalar SNR value in dB (averaged over subcarriers).
        """
        try:
            if h_mimo is None or tf.size(h_mimo) == 0 or beamforming_vector_w is None or tf.size(beamforming_vector_w) == 0:
                logger.warning("calculate_snr: Invalid h_mimo or beamforming_vector_w input.")
                return tf.constant(-10.0, dtype=tf.float32)

            h_mimo = tf.cast(h_mimo, tf.complex64)
            beamforming_vector_w = tf.cast(beamforming_vector_w, tf.complex64)

            # Ensure beamforming vector is a column vector [num_tx_ant, 1]
            if len(beamforming_vector_w.shape) == 1:
                w_col = tf.expand_dims(beamforming_vector_w, axis=-1)
            elif beamforming_vector_w.shape[-1] == 1:
                w_col = beamforming_vector_w
            else:
                if beamforming_vector_w.shape[0] == 1 and len(beamforming_vector_w.shape) == 2:
                    w_col = tf.transpose(beamforming_vector_w)
                else:
                    logger.error(f"calculate_snr: beamforming vector has unexpected shape {beamforming_vector_w.shape}")
                    return tf.constant(-10.0, dtype=tf.float32)

            # Verify dimensions for matrix multiplication
            num_tx_ant_h = tf.shape(h_mimo)[-2]
            num_tx_ant_w = tf.shape(w_col)[0]
            if num_tx_ant_h != num_tx_ant_w:
                logger.error(f"Dimension mismatch: H's tx_ant ({num_tx_ant_h}) != w's first dim ({num_tx_ant_w})")
                return tf.constant(-10.0, dtype=tf.float32)

            # Calculate effective channel gain: || H * w ||^2
            effective_channel = tf.matmul(h_mimo, w_col)
            channel_gain_per_sc = tf.reduce_sum(tf.abs(effective_channel)**2, axis=0)
            channel_gain_linear = tf.reduce_mean(channel_gain_per_sc, axis=-1)
            channel_gain_linear = tf.squeeze(channel_gain_linear)

            signal_power = tf.cast(tx_power_linear, tf.float32) * tf.cast(channel_gain_linear, tf.float32)
            noise = tf.maximum(tf.cast(noise_power_linear, tf.float32), 1e-20)
            snr_linear = signal_power / noise
            snr_linear = tf.maximum(snr_linear, 1e-10)
            snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)
            snr_db_clipped = tf.clip_by_value(snr_db, -10.0, 50.0)
            return snr_db_clipped

        except Exception as e:
            logger.error(f"Error in calculate_snr: {str(e)}", exc_info=True)
            return tf.constant(-10.0, dtype=tf.float32)

    def get_current_beam_angles(self) -> tf.Tensor:
        """Returns the current beam angles [azimuth, elevation] per AGV."""
        return self.current_beam_angles

    def should_switch_beam(self, current_avg_snr_db: float, agv_positions: tf.Tensor) -> tf.Tensor:
        """
        Determine if a switch is needed based on average SNR and positions.
        
        Returns a Boolean tensor indicating for each AGV if a switch should occur.
        """
        MIN_SNR_THRESHOLD = self.config["beamforming"]['min_snr_threshold']
        needs_switch = tf.zeros([self.config["num_agvs"]], dtype=tf.bool)
        if current_avg_snr_db < MIN_SNR_THRESHOLD:
            needs_switch = tf.ones([self.config["num_agvs"]], dtype=tf.bool)
            logger.debug(f"Avg SNR below threshold: {current_avg_snr_db:.2f} dB, switching beams.")
        logger.debug(f"Switch check: {needs_switch.numpy()}")
        return needs_switch

    def get_beam_history(self) -> list:
        """Returns the history of beam angles."""
        return self.beam_history

    def update_beam(self, new_beam_angles: Any, success: bool = True, channel_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update beam state and log metrics.
        
        Args:
            new_beam_angles: New beam angles to update (should be convertible to Tensor of shape [num_agvs, 2]).
            success: Whether the beam update is considered successful.
            channel_data: Optional channel data to store.
        """
        try:
            if self.current_beam_angles is None:
                self.current_beam_angles = tf.zeros([self.config["num_agvs"], 2], dtype=tf.float32)
            old_beam_angles = tf.identity(self.current_beam_angles)
            new_beam_angles_tensor = tf.convert_to_tensor(new_beam_angles, dtype=tf.float32)
            if new_beam_angles_tensor.shape != (self.config["num_agvs"], 2):
                raise ValueError(f"new_beam_angles shape {new_beam_angles_tensor.shape} does not match expected ({self.config['num_agvs']}, 2)")
            if not tf.reduce_all(tf.abs(new_beam_angles_tensor - old_beam_angles) < 1e-4):
                self.log_beam_switch(old_beam_angles, new_beam_angles_tensor)
                self.current_beam_angles = new_beam_angles_tensor
            else:
                logger.debug("No significant change in beam angles.")
            self.beam_history.append(self.current_beam_angles.numpy())
            if channel_data is not None:
                self.current_channel_state = channel_data
            self.update_packet_stats(success, during_switch=self.has_switch_occurred())
        except Exception as e:
            logger.error(f"Error updating beam angles: {str(e)}", exc_info=True)
            raise

    def has_switch_occurred(self) -> bool:
        """Return whether a beam switch occurred compared to the previous state."""
        if len(self.beam_history) < 2:
            return False
        current_angles = self.current_beam_angles.numpy()
        previous_angles = self.beam_history[-2]
        return not np.allclose(current_angles, previous_angles, atol=1e-4)

    def detect_blockage(self, channel_data: Dict[str, Any], agv_positions: tf.Tensor,
                          obstacle_positions: tf.Tensor) -> tf.Tensor:
        """
        Detect blockage based on geometric ray intersection and LOS conditions from channel_data.
        
        Returns a Boolean tensor for each AGV indicating blockage.
        """
        try:
            agv_positions = tf.ensure_shape(tf.convert_to_tensor(agv_positions, dtype=tf.float32),
                                             [self.config["num_agvs"], 3])
            obstacle_positions = tf.convert_to_tensor(obstacle_positions, dtype=tf.float32)
            bs_pos = tf.cast(self.config["bs_position"], tf.float32)

            def check_ray_intersection(agv_pos: tf.Tensor) -> tf.Tensor:
                ray_start = bs_pos[tf.newaxis, :]
                ray_end = agv_pos[tf.newaxis, :]
                ray_dir = ray_end - ray_start
                ray_length = tf.norm(ray_dir)
                ray_dir = ray_dir / (ray_length + tf.keras.backend.epsilon())
                to_obstacles = obstacle_positions - ray_start
                projections = tf.reduce_sum(to_obstacles * ray_dir, axis=1)
                closest_points = ray_start + projections[:, tf.newaxis] * ray_dir
                distances = tf.norm(obstacle_positions - closest_points, axis=1)
                obstacle_radius = self.config["beamforming"].get('obstacle_radius', 0.5)
                is_between = tf.logical_and(projections >= 0, projections <= ray_length)
                intersects = tf.logical_and(distances < obstacle_radius, is_between)
                return tf.reduce_any(intersects)

            direct_blocked = tf.map_fn(check_ray_intersection, agv_positions, dtype=tf.bool)
            logger.debug(f"Geometric blockage check: {direct_blocked.numpy()}")

            los_conditions = channel_data.get('los_conditions')
            if los_conditions is not None:
                los_conditions = tf.cast(los_conditions, tf.bool)
                rt_blocked = tf.logical_not(los_conditions)
                logger.debug(f"Ray Tracing LOS blocked: {rt_blocked.numpy()}")
                final_blocked = rt_blocked
            else:
                logger.warning("LOS conditions not found in channel data, using geometric check.")
                final_blocked = direct_blocked

            logger.debug(f"Final blocked status: {final_blocked.numpy()}")
            return final_blocked
        except Exception as e:
            logger.error(f"Error in blockage detection: {str(e)}", exc_info=True)
            return tf.ones([self.config["num_agvs"]], dtype=tf.bool)

    def optimize_beam_direction(self, channel_data: Dict[str, Any], path_manager: Any,
                                  obstacle_positions: tf.Tensor) -> tf.Tensor:
        """
        Optimize beam direction for each AGV based on accurate SNR calculations.
        Returns the angles corresponding to the optimal beam vectors.
        """
        try:
            agv_positions = tf.stack([
                tf.convert_to_tensor(path_manager.get_current_status(f'agv_{i}')['position'], dtype=tf.float32)
                if path_manager.get_current_status(f'agv_{i}') is not None and 'position' in path_manager.get_current_status(f'agv_{i}')
                else tf.constant([0.0, 0.0, 0.5], dtype=tf.float32)
                for i in range(self.config["num_agvs"])
            ])
            logger.debug(f"AGV positions: {agv_positions.numpy()}")
            blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
            logger.debug(f"Blockage status: {blocked.numpy()}")
            bs_pos = tf.cast(self.config["bs_position"], tf.float32)
            direction_vectors = agv_positions - bs_pos
            horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1) + 1e-9
            azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
            elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
            azimuths = tf.clip_by_value(azimuths, -self.config["beamforming"]['max_steering_angle'], self.config["beamforming"]['max_steering_angle'])
            elevations = tf.clip_by_value(elevations, -self.config["beamforming"]['max_steering_angle']/2.0, self.config["beamforming"]['max_steering_angle']/2.0)
            direct_beams_angles = tf.stack([azimuths, elevations], axis=1)
            logger.debug(f"Direct geometric beam angles: {direct_beams_angles.numpy()}")
            needs_switch = blocked
            logger.debug(f"Needs switch status: {needs_switch.numpy()}")

            if not (channel_data and 'channel_matrices' in channel_data and channel_data['channel_matrices'] is not None):
                logger.error("Full MIMO channel matrix 'channel_matrices' not found. Returning current beam angles.")
                return self.current_beam_angles

            h_mimo_full = channel_data['channel_matrices']
            if tf.rank(h_mimo_full) < 4 or tf.shape(h_mimo_full)[0] != self.config["num_agvs"]:
                logger.error(f"Unexpected shape for h_mimo_full: {h_mimo_full.shape}.")
                return self.current_beam_angles

            tx_power_linear = tf.pow(10.0, (self.config["tx_power"] - 30)/10.0)
            noise_power_linear = tf.maximum(self.config["simulation"]['noise_power'], 1e-20)
            optimal_beam_angles_list = []
            calculated_snrs_per_agv = []
            for agv_idx in range(self.config["num_agvs"]):
                if not needs_switch[agv_idx]:
                    optimal_beam_angles_list.append(direct_beams_angles[agv_idx])
                    calculated_snrs_per_agv.append(-99.0)
                    logger.debug(f"AGV {agv_idx}: No switch needed, using direct geometric beam angles.")
                else:
                    logger.debug(f"AGV {agv_idx}: Switch needed. Evaluating {self.codebook_size} beams...")
                    best_snr_for_agv = tf.constant(-np.inf, dtype=tf.float32)
                    best_beam_index = 0
                    h_agv = h_mimo_full[agv_idx]
                    for beam_idx, w_vector in enumerate(tf.unstack(self.beam_vector_codebook, axis=0)):
                        snr_db = self.calculate_snr(h_agv, w_vector, tx_power_linear, noise_power_linear)
                        if snr_db > best_snr_for_agv:
                            best_snr_for_agv = snr_db
                            best_beam_index = beam_idx
                    if self.angle_codebook is not None and best_beam_index < self.angle_codebook.shape[0]:
                        optimal_beam_angles_list.append(self.angle_codebook[best_beam_index])
                        logger.debug(f"AGV {agv_idx}: Best vector {best_beam_index} with SNR {best_snr_for_agv:.2f} dB, angles: {self.angle_codebook[best_beam_index].numpy()}")
                    else:
                        logger.warning(f"AGV {agv_idx}: Angle codebook issue for index {best_beam_index}. Using direct angles.")
                        optimal_beam_angles_list.append(direct_beams_angles[agv_idx])
                    calculated_snrs_per_agv.append(float(best_snr_for_agv.numpy()))
            optimal_beams_angles = tf.stack(optimal_beam_angles_list, axis=0)
            optimal_beams_angles = tf.cast(optimal_beams_angles, tf.float32)
            logger.debug(f"Final optimal beam angles: {optimal_beams_angles.numpy()}")
            avg_optimal_snr = np.mean(calculated_snrs_per_agv) if calculated_snrs_per_agv else -10.0
            success = avg_optimal_snr > self.config["beamforming"]['min_snr_threshold']
            self.update_beam(optimal_beams_angles, success=success, channel_data=channel_data)
            self.log_snr(avg_optimal_snr)
            return optimal_beams_angles
        except Exception as e:
            logger.error(f"Error in beam optimization: {str(e)}", exc_info=True)
            return self.current_beam_angles

# --- END OF BeamManager CLASS ---
