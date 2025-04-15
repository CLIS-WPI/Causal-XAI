# src/beam_manager.py
# import mitsuba # Not directly used here? Remove if not needed.
# from mitsuba import ScalarTransform4f, Bitmap # Remove if not needed
import tensorflow as tf
import numpy as np
import logging
import time

# --- NEW IMPORT ---
# Adjust path based on your Sionna version (phy.mimo or just mimo?)
try:
    # Try Sionna 1.0+ style import
    from sionna.phy.mimo import grid_of_beams_dft
    logger.debug("Imported grid_of_beams_dft from sionna.phy.mimo")
except ImportError:
    try:
       # Try older Sionna style import
       from sionna.mimo import grid_of_beams_dft
       logger.debug("Imported grid_of_beams_dft from sionna.mimo")
    except ImportError:
       logger.error("Could not import grid_of_beams_dft from Sionna. Check your installation/version.")
       # Define a dummy function or raise error if critical
       def grid_of_beams_dft(**kwargs):
           logger.warning("Using dummy grid_of_beams_dft!")
           num_tx = kwargs.get('num_ant_v', 1) * kwargs.get('num_ant_h', 1)
           num_beams = num_tx # Simple fallback
           vecs = tf.complex(tf.random.normal([num_beams, num_tx]), tf.random.normal([num_beams, num_tx]))
           return vecs / tf.norm(vecs, axis=1, keepdims=True)
# --- END NEW IMPORT ---

logger = logging.getLogger(__name__)

class BeamManager:
    def __init__(self, config):
        self.config = config
        self.last_switch_time = None
        self.beam_history = [] # Will store angles
        self.snr_history = [] # Will store avg SNR dB calculated accurately
        self.switch_times = []
        self.current_channel_state = None
        self.channel_state_history = []
        # Stores current beam ANGLES [az, el] per AGV
        self.current_beam_angles = tf.zeros((config.num_agvs, 2), dtype=tf.float32)
        self.packet_stats = {
            'total': 0,
            'successful': 0,
            'failed_during_switch': 0
        }
        self.current_switch_start = None

        # --- MODIFIED: Initialize beam codebook (vectors and angles) ---
        self.beam_vector_codebook = None # Holds complex vectors w
        self.angle_codebook = None # Holds corresponding angles [az, el]
        self.codebook_size = 0
        self._initialize_beam_codebook()
        # --- END MODIFICATION ---

    def log_beam_switch(self, old_beam_angles, new_beam_angles):
        # Log based on angles
        switch_time = time.time()
        switch_duration = 0
        if self.current_switch_start is not None:
            switch_duration = switch_time - self.current_switch_start
            self.current_switch_start = None
        # else: # Resetting start time perhaps only when switch *ends*? Or start on trigger?
        #     self.current_switch_start = switch_time # Let's reset it here for now

        switch_data = {
            'timestamp': switch_time,
            'duration': switch_duration, # Duration calculation might need refinement
            'old_beam': tf.identity(old_beam_angles).numpy(),
            'new_beam': tf.identity(new_beam_angles).numpy()
        }
        # Add reason later in should_switch_beam or optimize_beam_direction
        self.switch_times.append(switch_data)
        logger.info(f"Beam switch logged: {old_beam_angles.numpy()} -> {new_beam_angles.numpy()}")


    def update_packet_stats(self, success, during_switch=False):
        self.packet_stats['total'] += 1
        if success:
            self.packet_stats['successful'] += 1
        elif during_switch:
            self.packet_stats['failed_during_switch'] += 1

    def log_snr(self, snr_value_db):
        # Logs the average SNR across AGVs for this time step
        self.snr_history.append({
            'timestamp': time.time(),
            'value': float(snr_value_db) # Ensure scalar
        })

    def get_performance_metrics(self):
        # (Keep as is)
        total = max(1, self.packet_stats['total'])
        return {
            'switch_times': self.switch_times,
            'packet_success_rate': self.packet_stats['successful'] / total,
            'switch_failure_rate': self.packet_stats['failed_during_switch'] / total,
            'snr_history': self.snr_history
        }

    # --- MODIFIED: Initialize both vector and angle codebooks ---
    def _initialize_beam_codebook(self):
        """Initializes beamforming vector codebook and corresponding angle codebook."""
        try:
            logger.info("Initializing Beamforming Vector & Angle Codebooks...")
            num_rows = self.config.bs_array['num_rows']
            num_cols = self.config.bs_array['num_cols']
            num_tx_ant = num_rows * num_cols
            num_beams_config = self.config.beamforming['num_beams']

            # --- Generate Beamforming Vectors (w) ---
            # Using Sionna function based on guidance
            self.beam_vector_codebook = grid_of_beams_dft(
                num_ant_v=num_rows,
                num_ant_h=num_cols,
                normalize=True
                # Add oversampling if needed/available and defined in config
                # oversampling_factor_v=self.config.beamforming['codebook'].get('oversampling_v', 1),
                # oversampling_factor_h=self.config.beamforming['codebook'].get('oversampling_h', 1),
            )
            self.beam_vector_codebook = tf.cast(self.beam_vector_codebook, dtype=tf.complex64)
            self.codebook_size = self.beam_vector_codebook.shape[0]
            logger.info(f"Generated beam vector codebook with shape: {self.beam_vector_codebook.shape}")

            # --- Generate Corresponding Angle Codebook (az, el) ---
            # This part generates angles roughly matching the DFT grid for logging/state tracking.
            # Note: grid_of_beams_dft doesn't directly return angles. This is an approximation.
            # We might need to adjust this based on how grid_of_beams_dft spaces beams.
            max_steering_angle = float(self.config.beamforming['max_steering_angle'])
            # Simple linspace approach (might not perfectly match DFT beam angles)
            # Adjust num_azimuth/num_elevation if grid_of_beams_dft uses different logic
            num_codebook_beams = self.codebook_size
            # Try to infer grid size, e.g., assume roughly square grid in angles
            num_az_approx = int(np.ceil(np.sqrt(num_codebook_beams)))
            num_el_approx = int(np.ceil(num_codebook_beams / num_az_approx))

            azimuth_angles = tf.linspace(-max_steering_angle, max_steering_angle, num_az_approx)
            # Reduce elevation range slightly based on typical UPA patterns
            elevation_angles = tf.linspace(-max_steering_angle / 2.0, max_steering_angle / 2.0, num_el_approx)

            az_grid, el_grid = tf.meshgrid(azimuth_angles, elevation_angles)
            angle_codebook_flat = tf.stack([tf.reshape(az_grid, [-1]), tf.reshape(el_grid, [-1])], axis=-1)

            # Trim if meshgrid produced more pairs than beams in vector codebook
            self.angle_codebook = angle_codebook_flat[:self.codebook_size, :]
            self.angle_codebook = tf.cast(self.angle_codebook, tf.float32)

            if self.angle_codebook.shape[0] != self.codebook_size:
                 logger.warning(f"Angle codebook size ({self.angle_codebook.shape[0]}) mismatch with vector codebook size ({self.codebook_size}). Angle mapping might be inaccurate.")
                 # Fallback: Create dummy angles
                 self.angle_codebook = tf.zeros([self.codebook_size, 2], dtype=tf.float32)


            logger.info(f"Generated angle codebook (for reference) with shape: {self.angle_codebook.shape}")

        except ImportError:
             logger.error("Failed to import grid_of_beams_dft. Using random fallback.")
             num_tx_ant = self.config.bs_array['num_rows'] * self.config.bs_array['num_cols']
             num_beams = self.config.beamforming['num_beams']
             # Vector codebook
             self.beam_vector_codebook = tf.complex(
                 tf.random.normal([num_beams, num_tx_ant]), tf.random.normal([num_beams, num_tx_ant])
             )
             self.beam_vector_codebook /= tf.norm(self.beam_vector_codebook, axis=1, keepdims=True)
             # Angle codebook (dummy)
             self.angle_codebook = tf.random.uniform([num_beams, 2], minval=-60, maxval=60, dtype=tf.float32)
             self.codebook_size = num_beams
             logger.warning("Using random fallback beam vector and angle codebooks.")
        except Exception as e:
             logger.error(f"Error initializing beam codebooks: {e}", exc_info=True)
             raise

    # --- Renamed for clarity ---
    def get_current_beam_angles(self):
        """Returns the current beam angles [az, el] per AGV."""
        return self.current_beam_angles

    # --- Should switch logic can remain similar, using average SNR maybe ---
    def should_switch_beam(self, current_avg_snr_db, agv_positions):
        """Determine if a switch might be needed based on average SNR and position."""
        # Note: This is a heuristic. The actual decision happens in optimize_beam_direction.
        MIN_SNR_THRESHOLD = self.config.beamforming['min_snr_threshold']
        # SNR_DROP_THRESHOLD = self.config.beamforming['beam_switching']['switching_threshold'] # Drop check might be complex with avg SNR
        DISTANCE_THRESHOLD = 50.0 # Increased distance threshold example

        needs_switch = tf.zeros([self.config.num_agvs], dtype=tf.bool)
        reasons = [None] * self.config.num_agvs

        # Check based on average SNR
        if current_avg_snr_db < MIN_SNR_THRESHOLD:
            needs_switch = tf.ones([self.config.num_agvs], dtype=tf.bool) # Switch all if avg is too low
            reasons = [f"Avg SNR below threshold: {current_avg_snr_db:.2f} < {MIN_SNR_THRESHOLD}"] * self.config.num_agvs
            logger.debug(reasons[0])
            # Store reason temporarily for logging in log_beam_switch if needed
            # This logic might need refinement - maybe check per-AGV SNR if available?
            # If self.switch_times: self.switch_times[-1]['reason'] = reasons[0] # Example
            return needs_switch # Return boolean tensor [num_agvs]

        # Check distance (example logic, might not be robust)
        # bs_pos = tf.cast(self.config.bs_position, tf.float32)
        # distances = tf.norm(tf.cast(agv_positions, tf.float32) - bs_pos, axis=1)
        # dist_exceeded = distances > DISTANCE_THRESHOLD
        # needs_switch = tf.logical_or(needs_switch, dist_exceeded)
        # for i in range(self.config.num_agvs):
        #     if dist_exceeded[i] and reasons[i] is None:
        #         reasons[i] = f"Distance exceeded: {distances[i]:.2f} > {DISTANCE_THRESHOLD}"
        # logger.debug(f"Distance check: {distances.numpy()}, Exceeded: {dist_exceeded.numpy()}")


        logger.debug(f"Switch check: Needs switch flags: {needs_switch.numpy()}")
        return needs_switch # Boolean tensor [num_agvs]

    def get_beam_history(self):
         # Returns history of angles
        try:
            # beam_history stores numpy arrays, return directly
            return self.beam_history
        except Exception as e:
            logger.error(f"Error getting beam history: {str(e)}")
            return []

    def update_beam(self, new_beam_angles, success=True, channel_data=None):
        """Update beam state (angles) and log metrics"""
        # Note: SNR logging might happen elsewhere based on more detailed calculations
        try:
            if self.current_beam_angles is None:
                 # Initialize if first call
                 self.current_beam_angles = tf.zeros([self.config.num_agvs, 2], dtype=tf.float32)

            old_beam_angles = tf.identity(self.current_beam_angles)
            new_beam_angles = tf.convert_to_tensor(new_beam_angles, dtype=tf.float32)

            if new_beam_angles.shape != [self.config.num_agvs, 2]:
                raise ValueError(f"new_beam_angles shape {new_beam_angles.shape} does not match expected ({self.config.num_agvs}, 2)")

            # Check if angles have changed significantly
            if not tf.reduce_all(tf.abs(new_beam_angles - old_beam_angles) < 1e-4):
                self.log_beam_switch(old_beam_angles, new_beam_angles)
                self.current_beam_angles = new_beam_angles
                # logger.info(f"Beam angles updated for AGVs: {new_beam_angles.numpy()}") # Logged in log_beam_switch
            else:
                 logger.debug("No significant beam angle change needed")

            # Store angles in history
            self.beam_history.append(self.current_beam_angles.numpy())

            if channel_data is not None:
                self.current_channel_state = channel_data # Store latest channel data

            # Update packet stats based on overall success for this step
            # SNR logging might be better handled in main loop where per-beam SNR is calculated
            if success is not None:
                 self.update_packet_stats(success, during_switch=self.has_switch_occurred())
                 # Maybe log the *average* SNR obtained with the new beam?
                 # avg_snr = channel_data.get('beam_metrics', {}).get('average_snr', 0.0) # Example
                 # self.log_snr(avg_snr) # Log average SNR for the step

        except Exception as e:
            logger.error(f"Error updating beam angles: {str(e)}")
            raise

    def has_switch_occurred(self):
        # Compares current angles to last angles in history
        if len(self.beam_history) < 2: # Need at least two entries to compare
            return False
        current_angles = self.current_beam_angles.numpy()
        previous_angles = self.beam_history[-2] # Compare with the one before the latest entry
        return not np.allclose(current_angles, previous_angles, atol=1e-4)


    def detect_blockage(self, channel_data, agv_positions, obstacle_positions):
        # (Keep existing geometric check, but use LOS from channel_data)
        try:
            agv_positions = tf.ensure_shape(tf.convert_to_tensor(agv_positions, dtype=tf.float32), [self.config.num_agvs, 3])
            obstacle_positions = tf.convert_to_tensor(obstacle_positions, dtype=tf.float32)
            bs_pos = tf.cast(self.config.bs_position, tf.float32)

            # Geometric check (same as before)
            def check_ray_intersection(agv_pos):
                # ... (existing intersection logic) ...
                ray_start = bs_pos[tf.newaxis, :]
                ray_end = agv_pos[tf.newaxis, :]
                ray_dir = ray_end - ray_start
                ray_length = tf.norm(ray_dir)
                ray_dir = ray_dir / (ray_length + tf.keras.backend.epsilon())
                to_obstacles = obstacle_positions - ray_start
                projections = tf.reduce_sum(to_obstacles * ray_dir, axis=1)
                closest_points = ray_start + projections[:, tf.newaxis] * ray_dir
                distances = tf.norm(obstacle_positions - closest_points, axis=1)
                # Use obstacle dimensions for a more accurate check if available, else use radius
                obstacle_radius = self.config.beamforming.get('obstacle_radius', 0.5) # Safer default
                is_between = tf.logical_and(projections >= 0, projections <= ray_length)
                intersects = tf.logical_and(distances < obstacle_radius, is_between)
                return tf.reduce_any(intersects)

            direct_blocked = tf.map_fn(check_ray_intersection, agv_positions, dtype=tf.bool)
            logger.debug(f"Geometric blockage check: {direct_blocked.numpy()}")

            # Use LOS condition from channel data (more reliable than SNR threshold)
            los_conditions = channel_data.get('los_conditions') # From Paths.LOS
            if los_conditions is not None:
                 los_conditions = tf.cast(los_conditions, tf.bool)
                 # Blocked if NOT Line-of-Sight
                 rt_blocked = tf.logical_not(los_conditions)
                 logger.debug(f"Ray Tracing LOS blocked: {rt_blocked.numpy()}")
                 # Combine geometric check with RT LOS check (e.g., blocked if RT says NLOS)
                 # Or prioritize RT LOS:
                 final_blocked = rt_blocked
            else:
                 logger.warning("LOS conditions not found in channel data, relying on geometric check.")
                 final_blocked = direct_blocked # Fallback to geometric if LOS unavailable

            logger.debug(f"Final blocked status: {final_blocked.numpy()}")
            return final_blocked
        except Exception as e:
            logger.error(f"Error in blockage detection: {str(e)}", exc_info=True)
            return tf.ones([self.config.num_agvs], dtype=tf.bool) # Assume blocked on error

    # --- REWRITTEN: Optimize using accurate SNR, return angles ---
    def optimize_beam_direction(self, channel_data, path_manager, obstacle_positions):
        """
        Optimizes beam direction for each AGV based on accurate SNR calculations
        across the beam vector codebook. Still uses exhaustive search for now.
        Returns the *angles* corresponding to the best beams.
        """
        try:
            # --- Get AGV positions, detect blockage (using updated detect_blockage) ---
            agv_positions = tf.stack([
                tf.convert_to_tensor(path_manager.get_current_status(f'agv_{i}')['position'], dtype=tf.float32)
                if path_manager.get_current_status(f'agv_{i}') is not None and 'position' in path_manager.get_current_status(f'agv_{i}')
                else tf.constant([0.0, 0.0, 0.5], dtype=tf.float32) # Fallback
                for i in range(self.config.num_agvs)
            ])
            logger.debug(f"AGV positions: {agv_positions.numpy()}")

            blocked = self.detect_blockage(channel_data, agv_positions, obstacle_positions)
            logger.debug(f"Blockage status: {blocked.numpy()}")

            # --- Calculate direct geometric beam angles (Keep as fallback/initial) ---
            bs_pos = tf.cast(self.config.bs_position, tf.float32)
            direction_vectors = agv_positions - bs_pos
            horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1) + 1e-9 # Avoid division by zero
            azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
            elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
            # Normalize/clip angles if necessary based on system constraints
            azimuths = tf.clip_by_value(azimuths, -self.config.beamforming['max_steering_angle'], self.config.beamforming['max_steering_angle'])
            elevations = tf.clip_by_value(elevations, -self.config.beamforming['max_steering_angle']/2.0, self.config.beamforming['max_steering_angle']/2.0) # Example range
            direct_beams_angles = tf.stack([azimuths, elevations], axis=1)
            logger.debug(f"Direct geometric beam angles: {direct_beams_angles.numpy()}")

            # --- Determine if switch is needed (Simplified heuristic - maybe refine) ---
            # Use LOS status primarily, maybe basic SNR threshold as secondary check
            needs_switch = blocked # Switch if blocked (NLOS according to RT)
            # Add other conditions if needed, e.g., very low SNR even if LOS?
            # current_snr_db = channel_data.get('beam_metrics', {}).get('snr_db', tf.ones(self.config.num_agvs)*-10.0)
            # needs_switch = tf.logical_or(needs_switch, current_snr_db < self.config.beamforming['min_snr_threshold'])
            logger.debug(f"Needs switch status (based on blockage primarily): {needs_switch.numpy()}")


            # --- Get Full MIMO Channel Matrix ---
            if not (channel_data and 'channel_matrices' in channel_data and channel_data['channel_matrices'] is not None):
                 logger.error("Full MIMO channel matrix 'channel_matrices' not found in channel_data. Cannot optimize.")
                 return self.current_beam_angles # Return current angles as fallback

            h_mimo_full = channel_data['channel_matrices'] # Shape: [num_agvs, num_rx_ant, num_tx_ant, num_subcarriers] (Expected)
            # Basic shape validation
            # Add more robust checks if needed
            if tf.rank(h_mimo_full) < 4 or tf.shape(h_mimo_full)[0] != self.config.num_agvs:
                logger.error(f"Unexpected shape for h_mimo_full: {h_mimo_full.shape}. Cannot optimize.")
                return self.current_beam_angles


            # --- Get Tx Power and Noise Power ---
            tx_power_linear = tf.pow(10.0, (self.config.tx_power - 30) / 10.0)
            noise_power_linear = tf.maximum(self.config.simulation['noise_power'], 1e-20)

            # --- Optimize Beam per AGV (Exhaustive Search with Accurate SNR) ---
            optimal_beam_angles_list = []
            calculated_snrs_per_agv = [] # To store the best SNR found for each AGV

            for agv_idx in range(self.config.num_agvs):
                if not needs_switch[agv_idx]:
                    # Use direct geometric beam angles if no switch indicated
                    optimal_beam_angles_list.append(direct_beams_angles[agv_idx])
                    # Calculate SNR for this direct beam for logging? - Optional
                    # Need to find corresponding 'w' for direct angles or use approx SNR
                    snr_for_direct_beam = -99.0 # Placeholder
                    calculated_snrs_per_agv.append(snr_for_direct_beam)
                    logger.debug(f"AGV {agv_idx}: No switch needed, using direct geometric beam angles.")
                else:
                    # Switch needed: Perform exhaustive search using the *vector* codebook
                    logger.debug(f"AGV {agv_idx}: Switch needed. Evaluating {self.codebook_size} beams...")
                    best_snr_for_agv = tf.constant(-np.inf, dtype=tf.float32) # Initialize with very low value
                    best_beam_index = 0

                    # Extract channel for this specific AGV
                    # h_agv shape: [num_rx_ant, num_tx_ant, num_subcarriers]
                    h_agv = h_mimo_full[agv_idx]

                    # Iterate through beam *vectors*
                    for beam_idx, w_vector in enumerate(self.beam_vector_codebook):
                        # Calculate SNR using the new accurate method
                        # calculate_snr returns scalar avg SNR dB over subcarriers
                        snr_db = self.calculate_snr(h_agv, w_vector, tx_power_linear, noise_power_linear)

                        # Find beam with max SNR
                        if snr_db > best_snr_for_agv:
                            best_snr_for_agv = snr_db
                            best_beam_index = beam_idx

                    # Store the angles corresponding to the best vector index
                    if self.angle_codebook is not None and best_beam_index < self.angle_codebook.shape[0]:
                        optimal_beam_angles_list.append(self.angle_codebook[best_beam_index])
                        logger.debug(f"AGV {agv_idx}: Best vector index {best_beam_index} found with SNR {best_snr_for_agv:.2f} dB. Angles: {self.angle_codebook[best_beam_index].numpy()}")
                    else:
                        logger.warning(f"AGV {agv_idx}: Angle codebook issue for index {best_beam_index}. Using direct beam angles as fallback.")
                        optimal_beam_angles_list.append(direct_beams_angles[agv_idx])

                    # Store the best SNR found for this AGV
                    calculated_snrs_per_agv.append(float(best_snr_for_agv.numpy()))


            # --- Combine optimal beam angles ---
            optimal_beams_angles = tf.stack(optimal_beam_angles_list, axis=0)
            optimal_beams_angles = tf.cast(optimal_beams_angles, tf.float32)
            logger.debug(f"Final optimal beam angles determined: {optimal_beams_angles.numpy()}")

            # --- Update internal state (call update_beam to handle logging/history) ---
            # Determine success based on average SNR achieved with the new beams
            avg_optimal_snr = np.mean(calculated_snrs_per_agv) if calculated_snrs_per_agv else -10.0
            success = avg_optimal_snr > self.config.beamforming['min_snr_threshold']
            self.update_beam(optimal_beams_angles, success=success, channel_data=channel_data)
            # Log the average SNR achieved with the selected beams
            self.log_snr(avg_optimal_snr)


            return optimal_beams_angles # Return the selected angles

        except Exception as e:
            logger.error(f"Error in beam optimization: {str(e)}", exc_info=True)
            # On error, return the last known beam angles to avoid crashing
            return self.current_beam_angles

    # --- REWRITTEN: Calculate SNR accurately using H and w ---
    def calculate_snr(self, h_mimo, beamforming_vector_w, tx_power_linear, noise_power_linear):
        """
        Calculates the SNR for a given beamforming vector w and MIMO channel H.
        Implements SNR = (TxPower * || H * w ||^2) / NoisePower averaged over subcarriers.

        Args:
            h_mimo: The MIMO channel matrix for a single link.
                    Expected shape: [num_rx_ant, num_tx_ant, num_subcarriers].
            beamforming_vector_w: The complex beamforming vector for the Tx array.
                                 Expected shape: [num_tx_ant] or [num_tx_ant, 1].
            tx_power_linear: Transmit power in linear scale (Watts).
            noise_power_linear: Noise power in linear scale (Watts).

        Returns:
            tf.Tensor: Scalar SNR value in dB (averaged over subcarriers).
                       Returns a low default value (e.g., -10 dB) on error.
        """
        try:
            # --- Input Validation ---
            if h_mimo is None or tf.size(h_mimo) == 0 or beamforming_vector_w is None or tf.size(beamforming_vector_w) == 0:
                logger.warning("calculate_snr: Invalid H or w input.")
                return tf.constant(-10.0, dtype=tf.float32)

            h_mimo = tf.cast(h_mimo, tf.complex64)
            beamforming_vector_w = tf.cast(beamforming_vector_w, tf.complex64)

            # Ensure w is column vector [num_tx_ant, 1]
            if len(beamforming_vector_w.shape) == 1:
                w_col = tf.expand_dims(beamforming_vector_w, axis=-1)
            elif beamforming_vector_w.shape[-1] == 1:
                w_col = beamforming_vector_w
            else: # Attempt transpose if shape seems like [1, num_tx_ant]
                 if beamforming_vector_w.shape[0] == 1 and len(beamforming_vector_w.shape) == 2:
                      w_col = tf.transpose(beamforming_vector_w)
                 else:
                      logger.error(f"w has unexpected shape {beamforming_vector_w.shape}")
                      return tf.constant(-10.0, dtype=tf.float32)

            # Verify dimensions for matmul: H[..., num_rx, num_tx, sc] @ w[num_tx, 1]
            num_rx_ant = tf.shape(h_mimo)[-3] # Assuming Rx Ant is 3rd last dim
            num_tx_ant_h = tf.shape(h_mimo)[-2] # Assuming Tx Ant is 2nd last dim
            num_sc = tf.shape(h_mimo)[-1]      # Assuming SC is last dim
            num_tx_ant_w = tf.shape(w_col)[0]

            if num_tx_ant_h != num_tx_ant_w:
                logger.error(f"Dimension mismatch for matmul: H tx_ant ({num_tx_ant_h}) != w dim 0 ({num_tx_ant_w})")
                return tf.constant(-10.0, dtype=tf.float32)

            # --- Calculate Effective Channel Gain: || H * w ||^2 ---
            # H shape: [num_rx, num_tx, num_sc], w shape: [num_tx, 1]
            # effective_channel shape: [num_rx, 1, num_sc]
            effective_channel = tf.matmul(h_mimo, w_col)

            # Sum power over receive antennas dimension (axis=-3 relative to H, now axis=0 of effective_channel)
            # channel_gain_per_sc shape: [1, num_sc] or [num_sc]
            channel_gain_per_sc = tf.reduce_sum(tf.abs(effective_channel)**2, axis=0)

            # Average gain over subcarriers
            # channel_gain_linear shape: scalar or [1]
            channel_gain_linear = tf.reduce_mean(channel_gain_per_sc, axis=-1) # Average over last dim (subcarriers)
            # Remove the dummy dimension '1' if needed
            if len(channel_gain_linear.shape) > 0: # If not scalar
                 channel_gain_linear = tf.squeeze(channel_gain_linear, axis=0)


            # --- Calculate SNR ---
            signal_power = tf.cast(tx_power_linear, tf.float32) * tf.cast(channel_gain_linear, tf.float32)
            noise = tf.maximum(tf.cast(noise_power_linear, tf.float32), 1e-20)

            snr_linear = signal_power / noise
            snr_linear = tf.maximum(snr_linear, 1e-10) # Avoid log(0)
            snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)

            # Clip to a realistic range
            snr_db_clipped = tf.clip_by_value(snr_db, -10.0, 50.0) # Increased upper range

            # logger.debug(f"Calculated SNR (dB) for beam: {snr_db_clipped.numpy()}") # Log inside loop might be too verbose
            return snr_db_clipped

        except Exception as e:
            logger.error(f"Error calculating SNR for beam: {str(e)}", exc_info=True)
            return tf.constant(-10.0, dtype=tf.float32) # Low default SNR on error

# --- END OF BeamManager CLASS ---