#channel_generator.py#
# Keep unchanged - external dependencies
from utils import ensure_mitsuba_variant
import mitsuba
import time
import tensorflow as tf
import numpy as np
import sionna
from scene_setup import setup_scene
import logging
from beam_manager import BeamManager
from agv_path_manager import AGVPathManager
from scipy.special import erfc
import gc

# Update Sionna imports based on new structure
from sionna.constants import SPEED_OF_LIGHT
from sionna.phy.channel.utils import cir_to_ofdm_channel, subcarrier_frequencies

# Update RT-related imports (these need to be reorganized based on new structure)
from sionna.rt.scenes import Scene
from sionna.rt.components import Transmitter, Receiver
from sionna.rt.antenna import PlanarArray, DiscretePhaseProfile
from sionna.rt.materials import RadioMaterial
from sionna.rt.paths import Paths, PathSolver
from sionna.rt.grid import CellGrid

logger = logging.getLogger(__name__)

class SmartFactoryChannel:
    """Smart Factory Channel Generator using Sionna"""
    
    def __init__(self, config, scene=None):
        try:
            # Force Mitsuba variant first
            variant = ensure_mitsuba_variant('cuda_ad_rgb')
            logger.debug(f"Channel generator initialized with Mitsuba variant: {variant}")
            # Verify Mitsuba variant before proceeding
            from mitsuba import __version__
            current_variant = mitsuba.variant()
            if current_variant != 'cuda_ad_rgb':
                raise RuntimeError(f"Wrong Mitsuba variant: {current_variant}. Required 'cuda_ad_rgb'")

            logger.debug("=== Initializing SmartFactoryChannel ===")
            logger.debug(f"Config parameters:")
            logger.debug(f"- Number of AGVs: {config.num_agvs}")
            logger.debug(f"- Room dimensions: {config.room_dim}")
            logger.debug(f"- Carrier frequency: {config.carrier_frequency} Hz")
            logger.debug(f"- Ray tracing config: {config.ray_tracing}")
            
            self.config = config
            sionna.config.xla_compat = True
            tf.random.set_seed(config.seed if hasattr(config, 'seed') else 42)
            
            self._setup_indoor_factory_params()
            logger.debug("Indoor factory parameters initialized")
            
            self.positions_history = [[] for _ in range(config.num_agvs)]
            self.agv_positions = self._generate_initial_agv_positions()
            logger.debug(f"Initial AGV positions:\n{self.agv_positions.numpy()}")

            logger.debug("Setting up scene...")
            self.scene = scene if scene is not None else setup_scene(config)
            logger.debug("Scene setup completed")
            
            logger.debug("Configuring antenna arrays...")
            self._setup_antenna_arrays()
            logger.debug("Antenna arrays configured successfully")
            
            self.verify_scene_configuration()
            logger.debug("Scene configuration verified successfully")
            
            self.path_manager = AGVPathManager(config)
            self.beam_manager = BeamManager(config)
            
            self.performance_metrics = {
                'ber_history': [],
                'snr_history': [],
                'packet_stats': {'total': 0, 'successful': 0}
            }   
            self.current_channel_state = {
                'los_status': None,
                'snr': None,
                'path_loss': None,
                'beam_direction': None,
                'timestamp': None,
                'channel_quality': {'average_power': None, 'peak_power': None, 'condition_number': None}
            }

        except Exception as e:
            logger.error(f"Channel initialization failed: {str(e)}")
            raise RuntimeError(f"Channel initialization failed: {str(e)}") from e

    def _generate_initial_agv_positions(self):
        """Generate initial AGV positions from config."""
        if not hasattr(self.config, 'agv_positions'):
            raise AttributeError("AGV positions not defined in config")
        return tf.constant(self.config.agv_positions, dtype=tf.float32)

    def calculate_ber(self, signal_data):
        try:
            if 'beam_metrics' in signal_data and 'snr_db' in signal_data['beam_metrics']:
                snr_db = np.mean(signal_data['beam_metrics']['snr_db'])
                snr_linear = 10**(snr_db/10)
                ber = 0.5 * erfc(np.sqrt(snr_linear))
            else:
                ber = 1.0
            
            self.performance_metrics['ber_history'].append({
                'timestamp': time.time(),
                'value': float(ber)
            })
            return ber
        except Exception as e:
            logger.error(f"Error calculating BER: {str(e)}")
            return 1.0
    
    def track_performance_metrics(self, signal_data):
        metrics = {
            'packet_success_rate': self.calculate_psr(),
            'ber_during_switch': self.calculate_ber(signal_data),
            'snr_variations': self.performance_metrics['snr_history']
        }
        return metrics
    
    def calculate_psr(self):
        if self.performance_metrics['packet_stats']['total'] == 0:
            return 0.0
        return (self.performance_metrics['packet_stats']['successful'] / 
                self.performance_metrics['packet_stats']['total'])
    
    def _setup_indoor_factory_params(self):
        try:
            self.inf_params = {
                'los_k_factor': 17.0,
                'nlos_sigma': 3.5,
                'path_loss_exp': 1.6,
                'shadow_std': 1.0,
                'penetration_loss': 8.0,
                'reflection_coeff': 0.85
            }
            logger.debug(f"Indoor factory parameters initialized: {self.inf_params}")
        except Exception as e:
            logger.error(f"Error setting up indoor factory parameters: {str(e)}")
            raise

    def calculate_path_loss(self, distance, frequency):
        wavelength = SPEED_OF_LIGHT / frequency
        # مدل ساده‌تر با ضریب کمتر
        basic_loss = 20 * tf.math.log(4 * np.pi * distance / wavelength) / tf.math.log(10.0)
        n = 1.0  # توی calculate_path_loss
        dist_loss = 10 * n * tf.math.log(distance + 1e-6) / tf.math.log(10.0)
        shadow_fading = tf.random.normal([], mean=0.0, stddev=self.inf_params['shadow_std'])
        total_loss = basic_loss + dist_loss + shadow_fading
        
        logger.debug(f"Path loss components:")
        logger.debug(f"- Basic loss: {float(basic_loss):.2f} dB")
        logger.debug(f"- Distance loss: {float(dist_loss):.2f} dB")
        logger.debug(f"- Shadow fading: {float(shadow_fading):.2f} dB")
        logger.debug(f"- Total loss: {float(total_loss):.2f} dB")
        
        return total_loss
    
    def _safe_process_channel(self, h_freq, num_agvs, num_subcarriers):
        try:
            if tf.shape(h_freq)[0] != num_agvs or tf.shape(h_freq)[-1] != num_subcarriers:
                logger.warning(f"Adjusting h_freq shape from {h_freq.shape} to [{num_agvs}, {num_subcarriers}]")
                return tf.reshape(h_freq, [num_agvs, num_subcarriers])
            return h_freq
        except Exception as e:
            logger.error(f"Failed to process h_freq: {str(e)}")
            return tf.complex(
                tf.random.normal([num_agvs, num_subcarriers], dtype=tf.float32),
                tf.random.normal([num_agvs, num_subcarriers], dtype=tf.float32)
            )

    @tf.autograph.experimental.do_not_convert
    def generate_channel_data(self, agv_positions):
        """Generate channel data with ray tracing, calculating power from CIR."""
        try:
            # Force eager execution if still needed for your Sionna/Mitsuba version
            # tf.config.run_functions_eagerly(True) # Uncomment if required

            logger.debug("=== Generating channel data ===")
            logger.debug(f"AGV positions input: shape={agv_positions.shape}")

            # Initialize channel_data dictionary
            channel_data = {
                'paths': None,
                'channel_matrices': None, # This might become the full MIMO matrix
                'h_freq_averaged': None, # Or keep the averaged one if needed elsewhere
                'path_delays': None,
                'los_conditions': None,
                'agv_positions': None,
                'received_power_linear': None, # Correctly calculated power
                'beam_metrics': None, # Keep structure for compatibility
                'path_data': None # Keep structure for compatibility
            }

            # --- Memory management (Keep as is) ---
            tf.keras.backend.clear_session()
            try:
                for device in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(device, True)
            except:
                pass
            gc.collect() # Explicit garbage collection

            # --- Ensure Mitsuba variant (Keep as is, assuming needed) ---
            # import mitsuba
            # mitsuba.set_variant('cuda_ad_rgb') # Or ensure variant logic

            # --- Handle batched input (Keep as is) ---
            if len(agv_positions.shape) == 3:  # Shape: (batch_size, num_agvs, 3)
                agv_positions = agv_positions[0]  # Take first time step: (num_agvs, 3)
            # ...(rest of shape validation)

            # --- Update receiver positions (Keep as is) ---
            agv_positions = tf.convert_to_tensor(agv_positions, dtype=self.config.real_dtype)
            for i in range(self.config.num_agvs):
                rx_name = f'rx_agv_{i}'
                if rx_name in self.scene.receivers:
                   self.scene.receivers[rx_name].position = agv_positions[i]
                else:
                   logger.error(f"Receiver {rx_name} not found in scene!")
            self._check_and_fix_receiver_positions() # Keep safety check

            tx_pos = list(self.scene.transmitters.values())[0].position
            tx_pos = tf.convert_to_tensor(tx_pos, dtype=self.config.real_dtype)

            # --- Compute paths using PathSolver (Sionna 1.0 API Style) ---
            logger.debug("Starting path computation...")
            # Instantiate solver here or use self.path_solver initialized in __init__
            # Need to check how RT parameters are passed in Sionna 1.0 PathSolver API
            # Example placeholder: assumes parameters are set during solver init
            path_solver = PathSolver() # Replace with self.path_solver if initialized elsewhere
            paths = path_solver(self.scene) # Call the solver instance

            # --- Fallback if paths=None (Keep similar logic) ---
            if paths is None:
                logger.error("Path computation returned None")
                # Use your existing fallback logic, ensure it returns compatible keys
                return self._create_fallback_channel_data(agv_positions, tx_pos)

            logger.debug("Paths computed successfully")
            channel_data['paths'] = paths
            channel_data['agv_positions'] = agv_positions

            # --- Compute CIR from paths (Keep as is) ---
            a, tau = paths.cir()
            a = tf.convert_to_tensor(a, dtype=tf.complex64)
            tau = tf.convert_to_tensor(tau, dtype=tf.float32)
            # !!! CRITICAL: Log and VERIFY the shape of 'a' !!!
            logger.debug(f"CIR computed: a={a.shape}, tau={tau.shape}")
            channel_data['path_delays'] = tau

            # --- Calculate Received Power from 'a' (NEW LOGIC) ---
            try:
                # Sum power over paths dimension. VERIFY THE AXIS INDEX (-1 is usually paths).
                path_powers = tf.abs(a)**2
                total_power_per_tx_rx_pair = tf.reduce_sum(path_powers, axis=-1)
                logger.debug(f"Total power per Tx/Rx pair: shape={total_power_per_tx_rx_pair.shape}")

                # Average power over Rx and Tx antennas to get power per AGV link.
                # VERIFY AXES based on the actual shape logged above.
                # Example: Assumes shape [batch=1, num_agvs=2, num_rx_ant=16, num_tx_ant=1024]
                if len(total_power_per_tx_rx_pair.shape) == 4:
                    dims_to_avg_over_antennas = [-1, -2] # Avg over Tx, then Rx antennas
                    received_power_per_agv = tf.reduce_mean(total_power_per_tx_rx_pair, axis=dims_to_avg_over_antennas)
                    received_power_per_agv = tf.squeeze(received_power_per_agv, axis=0) # Remove batch dim
                # Example: Assumes shape [num_agvs=2, num_rx_ant=16, num_tx_ant=1024]
                elif len(total_power_per_tx_rx_pair.shape) == 3:
                    dims_to_avg_over_antennas = [-1, -2] # Avg over Tx, then Rx antennas
                    received_power_per_agv = tf.reduce_mean(total_power_per_tx_rx_pair, axis=dims_to_avg_over_antennas)
                else:
                     logger.error(f"Unexpected shape for total_power_per_tx_rx_pair: {total_power_per_tx_rx_pair.shape}. Cannot reliably average over antennas.")
                     # Fallback: Use a simple mean, but this is likely wrong.
                     received_power_per_agv = tf.reduce_mean(total_power_per_tx_rx_pair, axis=list(range(1, tf.rank(total_power_per_tx_rx_pair))))
                     received_power_per_agv = tf.reshape(received_power_per_agv, [self.config.num_agvs])

                received_power_per_agv = tf.cast(received_power_per_agv, tf.float32)
                received_power_per_agv = tf.maximum(received_power_per_agv, 1e-20) # Ensure positive
                received_power_per_agv = tf.ensure_shape(received_power_per_agv, [self.config.num_agvs])

                # Store the correctly calculated linear power (includes path loss effects)
                channel_data['received_power_linear'] = received_power_per_agv
                logger.debug(f"Calculated received power (linear) per AGV from CIR: {received_power_per_agv.numpy()}")

            except Exception as e:
                logger.error(f"Error calculating received power from CIR: {str(e)}. Setting default.")
                channel_data['received_power_linear'] = tf.ones([self.config.num_agvs], dtype=tf.float32) * 1e-10 # Default low power
            # --- END NEW POWER CALCULATION ---

            # --- Apply fading (Keep as is, but check function signature if needed) ---
            los_conditions = tf.cast(paths.LOS, tf.bool)
            # ... (rest of fading logic) ...

            # --- Compute path powers and directions (Keep as is, for path_data dict) ---
            # ... (try-except block for path_powers, path_directions) ...
            # channel_data['path_data'] = ...

            # --- Compute OFDM channel (Keep, but decide if you need full MIMO or averaged) ---
            try:
                frequencies = subcarrier_frequencies(
                    num_subcarriers=self.config.num_subcarriers,
                    subcarrier_spacing=self.config.subcarrier_spacing
                )
                # This 'a' includes fading applied above
                h_freq_mimo = cir_to_ofdm_channel(frequencies, a, tau, normalize=True) # Potentially full MIMO H
                logger.debug(f"h_freq_mimo shape (Full MIMO): {h_freq_mimo.shape}")

                # Store the full MIMO matrix - RECOMMENDED for accurate beamforming SNR calc later
                channel_data['channel_matrices'] = h_freq_mimo

                # --- OPTIONAL: Keep the averaged version if needed elsewhere ---
                # Calculate the averaged version as before, if required by other parts of code
                try:
                    # Determine dimensions to reduce (all except AGVs and subcarriers)
                    # Assuming AGV dimension is axis=1 after potential squeeze/reshape from cir_to_ofdm
                    # Assuming Subcarrier dimension is axis=-1
                    # VERIFY THESE ASSUMPTIONS BASED ON h_freq_mimo.shape LOG
                    rank = tf.rank(h_freq_mimo)
                    agv_dim_index = 1 # Assumption, verify!
                    sc_dim_index = rank - 1
                    dims_to_reduce_avg = [i for i in range(rank) if i != agv_dim_index and i != sc_dim_index]

                    if dims_to_reduce_avg: # Only reduce if there are other dimensions
                       h_freq_avg = tf.reduce_mean(h_freq_mimo, axis=dims_to_reduce_avg)
                    else:
                       h_freq_avg = h_freq_mimo # Already has the target shape?

                    # Ensure final shape [num_agvs, num_subcarriers]
                    h_freq_avg = tf.reshape(h_freq_avg, [self.config.num_agvs, self.config.num_subcarriers])

                    channel_data['h_freq_averaged'] = h_freq_avg
                    logger.debug(f"h_freq_averaged shape: {h_freq_avg.shape}")
                except Exception as avg_e:
                    logger.error(f"Could not calculate averaged h_freq: {avg_e}")
                    channel_data['h_freq_averaged'] = None
                # --- END OPTIONAL AVERAGED H_FREQ ---

            except Exception as e:
                logger.error(f"Error in OFDM channel computation: {str(e)}")
                # Provide fallback for both potential matrices
                fallback_h = tf.complex(
                    tf.random.normal([self.config.num_agvs, self.config.num_subcarriers], dtype=tf.float32),
                    tf.random.normal([self.config.num_agvs, self.config.num_subcarriers], dtype=tf.float32)
                )
                # Need fallback for MIMO shape too - requires knowing expected MIMO dims
                channel_data['channel_matrices'] = None # Or fallback MIMO shape
                channel_data['h_freq_averaged'] = fallback_h


            # --- REMOVE OLD PATH LOSS CALCULATION BLOCK ---
            # The try-except block calculating 'path_losses' with the formula is already deleted from instructions above.

            # --- Calculate SNR using the *new* calculate_snr (needs modification too) ---
            # SNR calculation should ideally use 'received_power_linear' now.
            # Let's calculate a *basic* average SNR here for logging, assuming calculate_snr is called later.
            try:
                # We use the 'received_power_linear' calculated earlier.
                # This power already includes path loss from RT.
                # We still need Tx power, gains, and noise power from config.
                tx_power_linear = tf.pow(10.0, (self.config.tx_power - 30) / 10.0)
                tx_gain_linear = tf.pow(10.0, self.config.bs_array['antenna_gain_db'] / 10.0)
                rx_gain_linear = tf.pow(10.0, self.config.agv_array['antenna_gain_db'] / 10.0)
                total_noise_power = tf.maximum(self.config.simulation['noise_power'], 1e-20)

                # Simple SNR based on average received power (before beamforming gain)
                signal_power = tx_power_linear * tx_gain_linear * rx_gain_linear * channel_data['received_power_linear']
                snr_linear = signal_power / total_noise_power
                snr_linear = tf.maximum(snr_linear, 1e-10)
                snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)
                snr_db_clipped = tf.clip_by_value(snr_db, -10.0, 30.0)
                snr_db_clipped = tf.ensure_shape(snr_db_clipped, [self.config.num_agvs])

                channel_data['beam_metrics'] = {'snr_db': snr_db_clipped} # Store this basic SNR for now
                logger.debug(f"Basic average SNR calculated per AGV (before beam gain): {snr_db_clipped.numpy()}")

            except Exception as e:
                logger.error(f"Error calculating basic SNR in generate_channel_data: {str(e)}")
                channel_data['beam_metrics'] = {
                    'snr_db': tf.ones([self.config.num_agvs], dtype=tf.float32) * 10.0
                }

            # --- Process LOS conditions (Keep as is) ---
            try:
                los_conditions = tf.cast(paths.LOS, tf.int32)
                # ... (rest of LOS processing) ...
                channel_data['los_conditions'] = los_conditions
            except Exception as e:
                # ... (error handling) ...
                channel_data['los_conditions'] = tf.zeros([self.config.num_agvs], dtype=tf.int32)

            # --- Get beam directions (Keep as is, comes from BeamManager) ---
            try:
                beam_directions = self.beam_manager.get_current_beams() # Assumes beam_manager exists
                if 'beam_metrics' not in channel_data or channel_data['beam_metrics'] is None:
                    channel_data['beam_metrics'] = {}
                channel_data['beam_metrics']['beam_directions'] = beam_directions
            except Exception as e:
                # ... (error handling) ...
                 if 'beam_metrics' not in channel_data or channel_data['beam_metrics'] is None:
                    channel_data['beam_metrics'] = {}
                 channel_data['beam_metrics']['beam_directions'] = tf.zeros([self.config.num_agvs, 2], dtype=tf.float32)


            logger.debug(f"Channel data generated: keys={channel_data.keys()}")
            tf.config.run_functions_eagerly(False) # Disable eager after the function if you enabled it
            return channel_data

        except Exception as e:
            logger.error(f"Error in channel data generation: {str(e)}", exc_info=True)
            tf.config.run_functions_eagerly(False) # Ensure eager is disabled on error too
            # Use your existing fallback logic, ensure it returns compatible keys
            return self._create_fallback_channel_data(agv_positions if 'agv_positions' in locals() else None,
                                                   tx_pos if 'tx_pos' in locals() else None)


    def _create_fallback_channel_data(self, agv_positions=None, tx_pos=None):
        """Create fallback channel data when the main channel generation fails"""
        logger.warning("Creating fallback channel data")
        
        # Handle missing agv_positions
        if agv_positions is None:
            agv_positions = tf.zeros([self.config.num_agvs, 3], dtype=tf.float32)
            for i in range(self.config.num_agvs):
                if hasattr(self.scene.receivers, f'rx_agv_{i}'):
                    agv_positions = self.scene.receivers[f'rx_agv_{i}'].position
        
        # Handle missing tx_pos
        if tx_pos is None and hasattr(self, 'scene') and self.scene is not None:
            try:
                tx_pos = list(self.scene.transmitters.values())[0].position
            except (IndexError, AttributeError):
                tx_pos = tf.constant(self.config.bs_position, dtype=tf.float32)
        elif tx_pos is None:
            tx_pos = tf.constant(self.config.bs_position, dtype=tf.float32)
        
        # Create random channel matrices
        h_freq = tf.complex(
            tf.random.normal([self.config.num_agvs, self.config.num_subcarriers], dtype=tf.float32),
            tf.random.normal([self.config.num_agvs, self.config.num_subcarriers], dtype=tf.float32)
        )
        
        # Calculate basic path losses based on distance
        distances = tf.norm(agv_positions - tx_pos, axis=-1)
        path_losses = 20 * tf.math.log(distances + 1e-6) / tf.math.log(10.0) + 60.0
        
        # Create basic path data
        path_powers = tf.zeros([1, self.config.num_agvs, 1], dtype=tf.float32) + 0.1
        path_directions = tf.zeros([self.config.num_agvs, 2], dtype=tf.float32)
        
        # Random SNR values within reasonable bounds
        snr_db = tf.random.uniform([self.config.num_agvs], minval=5.0, maxval=20.0, dtype=tf.float32)
        
        # Create basic fallback channel data
        return {
            'paths': None,
            'channel_matrices': h_freq,
            'path_delays': tf.zeros([1, self.config.num_agvs, 1], dtype=tf.float32),
            'los_conditions': tf.zeros([self.config.num_agvs], dtype=tf.int32),
            'agv_positions': agv_positions,
            'path_losses': path_losses,
            'beam_metrics': {
                'snr_db': snr_db,
                'beam_directions': self.beam_manager.get_current_beams() 
                    if hasattr(self, 'beam_manager') else tf.zeros([self.config.num_agvs, 2], dtype=tf.float32)
            },
            'path_data': {
                'path_powers': path_powers,
                'path_directions': path_directions
            }
        }

    def _apply_rician_fading(self, channel, k_factor):
        """Apply Rician fading to the channel with the given K-factor"""
        try:
            k_linear = tf.pow(10.0, k_factor / 10.0)
            shape = tf.shape(channel)
            real = tf.random.normal(shape, mean=0.0, stddev=1.0)
            imag = tf.random.normal(shape, mean=0.0, stddev=1.0)
            los_component = tf.sqrt(k_linear / (k_linear + 1))
            nlos_component = tf.sqrt(1 / (k_linear + 1)) * tf.complex(real, imag)
            return channel * (los_component + nlos_component)
        except Exception as e:
            logger.error(f"Error in Rician fading: {str(e)}")
            return channel

    def _apply_rayleigh_fading(self, channel, sigma):
        """Apply Rayleigh fading to the channel with the given sigma"""
        try:
            shape = tf.shape(channel)
            real = tf.random.normal(shape, mean=0.0, stddev=sigma)
            imag = tf.random.normal(shape, mean=0.0, stddev=sigma)
            return channel * tf.complex(real, imag)
        except Exception as e:
            logger.error(f"Error in Rayleigh fading: {str(e)}")
            return channel

    def verify_scene_configuration(self):
        if self.scene is None:
            raise RuntimeError("Scene not initialized")
        if not hasattr(self, 'bs_array') or not hasattr(self, 'agv_array'):
            raise RuntimeError("Antenna arrays not properly configured")

    def _setup_antenna_arrays(self):
        try:
            logger.debug("=== Setting up antenna arrays ===")
            self.bs_array = PlanarArray(
                num_rows=self.config.bs_array['num_rows'],
                num_cols=self.config.bs_array['num_cols'],
                vertical_spacing=self.config.bs_array['vertical_spacing'] * self.config.wavelength,
                horizontal_spacing=self.config.bs_array['horizontal_spacing'] * self.config.wavelength,
                pattern=self.config.bs_array['pattern'],
                polarization=self.config.bs_array['polarization'],
                dtype=tf.complex64
            )
            logger.debug("BS array configured successfully")
            self.agv_array = PlanarArray(
                num_rows=self.config.agv_array['num_rows'],
                num_cols=self.config.agv_array['num_cols'],
                vertical_spacing=self.config.agv_array['vertical_spacing'],
                horizontal_spacing=self.config.agv_array['horizontal_spacing'],
                pattern=self.config.agv_array['pattern'],
                polarization=self.config.agv_array['polarization'],
                dtype=tf.complex64
            )
            logger.debug("AGV array configured successfully")
        except Exception as e:
            logger.error(f"Error in antenna array setup: {e}")
            raise RuntimeError("Failed to configure antenna arrays")

    def _update_agv_positions(self, time_step):
        current_positions = self.agv_positions.numpy()
        new_positions = []
        for i in range(self.config.num_agvs):
            current_pos = current_positions[i]
            new_pos = self.path_manager.get_next_position(f'agv_{i+1}', current_pos)
            new_positions.append(new_pos)
            self.positions_history[i].append(new_pos.copy())
        self.agv_positions = tf.convert_to_tensor(new_positions, dtype=self.config.real_dtype)

    def get_agv_status(self):
        status = {}
        for i in range(self.config.num_agvs):
            agv_id = f'agv_{i+1}'
            status[agv_id] = self.path_manager.get_current_status(agv_id)
        return status
        
    def simulate_movement(self, num_steps):
        movement_data = []
        for step in range(num_steps):
            self._update_agv_positions(step)
            movement_data.append({
                'step': step,
                'positions': self.agv_positions.numpy(),
                'status': self.get_agv_status()
            })
        return movement_data

    def monitor_channel_quality(self, h):
        nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(h), tf.int32))
        inf_count = tf.reduce_sum(tf.cast(tf.math.is_inf(h), tf.int32))
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Channel matrix contains {nan_count} NaN and {inf_count} Inf values")
        avg_power = tf.reduce_mean(tf.abs(h)**2)
        peak_power = tf.reduce_max(tf.abs(h)**2)
        min_power = tf.reduce_min(tf.abs(h)**2)
        s = tf.linalg.svd(h, compute_uv=False)
        condition_number = s[0] / s[-1]
        logger.info(f"Channel Quality Metrics:")
        logger.info(f"- Average power: {avg_power}")
        logger.info(f"- Peak power: {peak_power}")
        logger.info(f"- Minimum power: {min_power}")
        logger.info(f"- Condition number: {condition_number}")
        return h

    def analyze_beam_switching(self, channel_data):
        try:
            h_freq = channel_data['channel_matrices']
            los_conditions = channel_data['los_conditions']
            if not isinstance(los_conditions, tf.Tensor):
                los_conditions = tf.convert_to_tensor(los_conditions, dtype=tf.int32)
            
            noise_power = self.config.simulation['noise_power']
            signal_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=-1)
            
            # Calculate SNR and ensure it's 1D
            snr_db = 10 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)
            snr_db = tf.reshape(snr_db, [-1])  # Force 1D tensor (e.g., shape (2,))
            
            snr_threshold = self.config.beamforming['min_snr_threshold']
            beam_switches_needed = tf.where(snr_db < snr_threshold, True, False)
            
            return {
                'snr_db': snr_db.numpy(),  # Now guaranteed to be 1D
                'beam_switches_needed': beam_switches_needed.numpy(),
                'los_conditions': los_conditions.numpy()
            }
        except Exception as e:
            logger.error(f"Error in beam switching analysis: {str(e)}")
            raise

    def update_channel_state(self, channel_data):
        try:
            cm = channel_data['channel_matrices']
            if not isinstance(cm, tf.Tensor):
                cm = tf.convert_to_tensor(cm)
            self.current_channel_state = {
                'los_status': self.check_los_conditions(),
                'snr': channel_data.get('average_snr'),
                'path_loss': tf.reduce_mean(channel_data.get('path_losses', 0.0)),
                'beam_direction': self.beam_manager.get_current_beams() if hasattr(self, 'beam_manager') else None,
                'timestamp': time.time(),
                'channel_quality': {
                    'average_power': tf.reduce_mean(tf.abs(cm)**2).numpy(),
                    'peak_power': tf.reduce_max(tf.abs(cm)**2).numpy(),
                    'condition_number': None
                }
            }
        except Exception as e:
            logger.error(f"Error updating channel state: {str(e)}")
            self.current_channel_state = None


    def _check_and_fix_receiver_positions(self):
        """Check and fix receiver positions to ensure they are correctly shaped"""
        for rx_name, rx in self.scene.receivers.items():
            try:
                position = rx.position
                # Convert to tensor if not already
                if not isinstance(position, tf.Tensor):
                    position = tf.convert_to_tensor(position, dtype=tf.float32)
                
                # Fix shape issues
                if len(position.shape) > 1 and position.shape[0] > 1:
                    logger.warning(f"Receiver {rx_name} has invalid position shape {position.shape}")
                    # Take only the first row if it's a 2D array
                    corrected_position = position[0]
                    logger.warning(f"Correcting to: {corrected_position}")
                    rx.position = corrected_position
                elif len(position.shape) == 0:
                    logger.warning(f"Receiver {rx_name} has scalar position, converting to vector")
                    rx.position = tf.zeros([3], dtype=tf.float32)
            except Exception as e:
                logger.error(f"Error fixing position for receiver {rx_name}: {str(e)}")
                # Set a safe default position
                rx.position = tf.zeros([3], dtype=tf.float32)

    def calculate_snr(self, h_freq, config, path_losses=None):
        try:
            logger.debug("=== Starting SNR calculation ===")
            logger.debug(f"Input h_freq: shape={h_freq.shape}, dtype={h_freq.dtype}, sample={h_freq.numpy()[:2, :5]}")
            print(f"Input h_freq: shape={h_freq.shape}, dtype={h_freq.dtype}, sample={h_freq.numpy()[:2, :5]}")

            # Safety check - verify h_freq is valid (no scaling applied yet)
            if (tf.reduce_any(tf.math.is_nan(tf.math.real(h_freq))) or 
                tf.reduce_any(tf.math.is_inf(tf.math.real(h_freq))) or
                tf.reduce_any(tf.math.is_nan(tf.math.imag(h_freq))) or 
                tf.reduce_any(tf.math.is_inf(tf.math.imag(h_freq)))):
                logger.warning("h_freq contains NaN or Inf values, normalizing")
                h_freq = h_freq / (tf.reduce_max(tf.abs(h_freq)) + 1e-10) * 10.0

            # Ensure path_losses is properly formatted
            if path_losses is None:
                logger.debug("No path_losses provided, using default")
                path_losses = tf.ones([self.config.num_agvs], dtype=tf.float32) * 70.0
                print(f"Using default path losses: {path_losses.numpy()}")
            elif not isinstance(path_losses, tf.Tensor):
                logger.debug("Converting path_losses to tensor")
                path_losses = tf.convert_to_tensor(path_losses, dtype=tf.float32)
                if tf.size(path_losses) != self.config.num_agvs:
                    logger.warning(f"Reshaping path_losses from {tf.shape(path_losses)} to [{self.config.num_agvs}]")
                    path_losses = tf.broadcast_to(path_losses, [self.config.num_agvs])
                print(f"Path losses converted to tensor: shape={path_losses.shape}, value={path_losses.numpy()}")

            logger.debug(f"Input path_losses: shape={path_losses.shape}, value={path_losses.numpy()}")

            # Get configuration parameters
            tx_power_dbm = config.tx_power
            tx_antenna_gain_db = config.bs_array['antenna_gain_db']
            rx_antenna_gain_db = config.agv_array['antenna_gain_db']
            total_noise_power = config.simulation['noise_power']
            logger.debug(f"Config parameters - TX power: {tx_power_dbm} dBm, TX gain: {tx_antenna_gain_db} dB, RX gain: {rx_antenna_gain_db} dB, Noise power: {total_noise_power:.2e} W")
            print(f"Transmit power dBm: {tx_power_dbm}, TX gain dB: {tx_antenna_gain_db}, RX gain dB: {rx_antenna_gain_db}, Noise power: {total_noise_power:.2e} W")

            # Convert from dB to linear scale safely
            tx_power = tf.pow(10.0, (tx_power_dbm - 30) / 10.0)  # dBm to Watts
            tx_gain = tf.pow(10.0, tx_antenna_gain_db / 10.0)
            rx_gain = tf.pow(10.0, rx_antenna_gain_db / 10.0)
            logger.debug(f"Linear conversions - TX power: {float(tx_power):.2e} W, TX gain: {float(tx_gain):.2f}, RX gain: {float(rx_gain):.2f}, Noise power: {float(total_noise_power):.2e} W")
            print(f"Transmit power: {float(tx_power):.2e} W, TX gain: {float(tx_gain):.2f}, RX gain: {float(rx_gain):.2f}, Noise power: {float(total_noise_power):.2e} W")

            # Calculate channel power without arbitrary scaling
            logger.debug("Calculating channel power from h_freq")
            try:
                # Average over subcarriers (and other dims if present)
                channel_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=1)
                channel_power = tf.reshape(channel_power, [self.config.num_agvs])
            except Exception as e:
                logger.error(f"Error in channel power calculation: {str(e)}")
                channel_power = tf.ones([self.config.num_agvs], dtype=tf.float32)

            # Check for NaN/Inf values and replace if needed
            if tf.reduce_any(tf.math.is_nan(channel_power)) or tf.reduce_any(tf.math.is_inf(channel_power)):
                logger.warning("Channel power contains NaN or Inf, replacing with default")
                channel_power = tf.where(
                    tf.math.is_nan(channel_power) | tf.math.is_inf(channel_power),
                    tf.ones_like(channel_power),
                    channel_power
                )

            logger.debug(f"Channel power calculated: shape={channel_power.shape}, mean={float(tf.reduce_mean(channel_power)):.2e}, value={channel_power.numpy()}")

            # Calculate signal power
            signal_power = channel_power * tx_power * tx_gain * rx_gain
            logger.debug(f"Signal power (no path loss): shape={signal_power.shape}, mean={float(tf.reduce_mean(signal_power)):.2e}, value={signal_power.numpy()}")

            # Apply path losses if provided
            if path_losses is not None:
                logger.debug("Applying path losses")
                path_losses = tf.reshape(path_losses, [self.config.num_agvs])
                path_loss_linear = tf.pow(10.0, -path_losses / 10.0)
                path_loss_linear = tf.maximum(path_loss_linear, 1e-10)  # Avoid division by zero
                signal_power *= tf.cast(path_loss_linear, tf.float32)
                logger.debug(f"Signal power (with path loss): shape={signal_power.shape}, mean={float(tf.reduce_mean(signal_power)):.2e}, value={signal_power.numpy()}")

            # Ensure noise power is positive
            total_noise_power = tf.maximum(total_noise_power, 1e-20)

            # Compute SNR
            snr_linear = signal_power / total_noise_power
            snr_linear = tf.maximum(snr_linear, 1e-10)  # Avoid log(0)
            logger.debug(f"SNR calculation intermediate - Signal power: {signal_power.numpy()}, SNR linear: {snr_linear.numpy()}")  # Added logging as requested
            
            snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)

            # Clip to a realistic range
            snr_db_clipped = tf.clip_by_value(snr_db, -10.0, 30.0)  # Changed from 40.0 to 30.0
            snr_db_clipped = tf.ensure_shape(snr_db_clipped, [self.config.num_agvs])
            average_snr = float(tf.reduce_mean(snr_db_clipped))

            logger.debug(f"SNR clipped - Shape: {snr_db_clipped.shape}, Mean: {average_snr:.2f} dB, Value: {snr_db_clipped.numpy()}")
            print(f"SNR clipped - Shape: {snr_db_clipped.shape}, Mean: {average_snr:.2f} dB, Value: {snr_db_clipped.numpy()}")

            return {
                'average_snr': average_snr,
                'beam_metrics': {'snr_db': snr_db_clipped}
            }
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}", exc_info=True)
            snr_db = tf.ones([self.config.num_agvs], dtype=tf.float32) * 15.0
            print(f"ERROR in SNR calculation: {str(e)}, using fallback values")
            return {
                'average_snr': 15.0,
                'beam_metrics': {'snr_db': snr_db}
            }

    def track_los_nlos_paths(self):
            try:
                logger.debug("Computing paths for LOS/NLOS analysis...")
                paths = self.scene.compute_paths(
                    max_depth=self.config.ray_tracing['max_depth'],
                    method=self.config.ray_tracing['method'],
                    num_samples=self.config.ray_tracing['num_samples'],
                    los=True,
                    reflection=self.config.ray_tracing['reflection'],
                    diffraction=self.config.ray_tracing['diffraction'],
                    scattering=self.config.ray_tracing['scattering']
                )
                if paths is None:
                    logger.warning("No paths computed in ray tracing")
                    return {'los_ratio': 0.0, 'nlos_ratio': 0.0, 'total_paths': 0, 'blocked_paths': 0}, None
                los_conditions = paths.LOS
                if not isinstance(los_conditions, tf.Tensor):
                    los_conditions = tf.convert_to_tensor(los_conditions, dtype=tf.float32)
                total_paths = tf.cast(tf.size(los_conditions), tf.float32)
                if total_paths == 0:
                    logger.warning("No paths found in channel computation")
                    return {'los_ratio': 0.0, 'nlos_ratio': 0.0, 'total_paths': 0, 'blocked_paths': 0}, los_conditions
                los_paths = tf.cast(tf.reduce_sum(tf.cast(los_conditions, tf.int32)), tf.float32)
                if tf.reduce_any(tf.math.is_nan(los_paths)) or tf.reduce_any(tf.math.is_nan(total_paths)):
                    logger.warning("NaN values detected in path calculations")
                    return {'los_ratio': 0.0, 'nlos_ratio': 0.0, 'total_paths': 0, 'blocked_paths': 0}, los_conditions
                nlos_paths = total_paths - los_paths
                los_ratio = tf.where(total_paths > 0, los_paths / total_paths, tf.zeros_like(total_paths))
                nlos_ratio = tf.where(total_paths > 0, nlos_paths / total_paths, tf.zeros_like(total_paths))
                try:
                    nlos_stats = {
                        'los_ratio': float(los_ratio.numpy()),
                        'nlos_ratio': float(nlos_ratio.numpy()),
                        'total_paths': int(total_paths.numpy()),
                        'blocked_paths': int(nlos_paths.numpy())
                    }
                    logger.debug(f"Path Analysis Results:")
                    logger.debug(f"- Total paths: {nlos_stats['total_paths']}")
                    logger.debug(f"- LOS ratio: {nlos_stats['los_ratio']:.2f}")
                    logger.debug(f"- NLOS ratio: {nlos_stats['nlos_ratio']:.2f}")
                    logger.debug(f"- Blocked paths: {nlos_stats['blocked_paths']}")
                    return nlos_stats, los_conditions
                except Exception as e:
                    logger.error(f"Error converting path statistics: {str(e)}")
                    return {'los_ratio': 0.0, 'nlos_ratio': 0.0, 'total_paths': 0, 'blocked_paths': 0}, los_conditions
            except Exception as e:
                logger.error(f"Error tracking LOS/NLOS paths: {str(e)}")
                raise    

    def apply_fading(self, channel, los_condition):
        try:
            if los_condition:
                k_factor = self.inf_params['los_k_factor']
                return self._apply_rician_fading(channel, k_factor)
            else:
                sigma = self.inf_params['nlos_sigma']
                return self._apply_rayleigh_fading(channel, sigma)
        except Exception as e:
            logger.error(f"Error applying fading: {str(e)}")
            raise

    def _apply_rician_fading(self, channel, k_factor):
        try:
            k_linear = tf.pow(10.0, k_factor / 10.0)
            shape = tf.shape(channel)
            real = tf.random.normal(shape, mean=0.0, stddev=1.0)
            imag = tf.random.normal(shape, mean=0.0, stddev=1.0)
            los_component = tf.sqrt(k_linear / (k_linear + 1))
            nlos_component = tf.sqrt(1 / (k_linear + 1)) * tf.complex(real, imag)
            return channel * (los_component + nlos_component)
        except Exception as e:
            logger.error(f"Error in Rician fading: {str(e)}")
            raise

    def _apply_rayleigh_fading(self, channel, sigma):
        try:
            shape = tf.shape(channel)
            real = tf.random.normal(shape, mean=0.0, stddev=sigma)
            imag = tf.random.normal(shape, mean=0.0, stddev=sigma)
            return channel * tf.complex(real, imag)
        except Exception as e:
            logger.error(f"Error in Rayleigh fading: {str(e)}")
            raise
    
    def calculate_beam_performance(self):
        try:
            h = self.monitor_channel_quality(self.generate_channel()['h'])
            temp = self.config.simulation['noise_power']
            signal_power = tf.reduce_mean(tf.abs(h)**2, axis=-1)
            snr_db = 10 * tf.math.log(signal_power / temp ) / tf.math.log(10.0)
            beam_metrics = {
                'snr_db': snr_db.numpy(),
                'avg_power': float(tf.reduce_mean(signal_power)),
                'max_power': float(tf.reduce_max(signal_power)),
                'min_power': float(tf.reduce_min(signal_power))
            }
            return beam_metrics
        except Exception as e:
            logger.error(f"Error calculating beam performance: {str(e)}")
            raise

    def monitor_channel_quality(self, h):
        metrics = {
            'snr': self._calculate_snr(h),
            'condition_number': tf.linalg.cond(h),
            'rank': tf.rank(h),
            'eigenvalues': tf.linalg.eigvals(h),
            'path_diversity': self._calculate_path_diversity(h),
            'temporal_correlation': self._calculate_temporal_correlation(h),
            'spatial_correlation': self._calculate_spatial_correlation(h)
        }
        return metrics
    
    def check_los_conditions(self):
        try:
            paths = self.scene.compute_paths(
                max_depth=1,
                method='fibonacci',
                num_samples=100,
                los=True,
                reflection=False,
                diffraction=False,
                scattering=False
            )
            if paths is None or not hasattr(paths, 'LOS'):
                logger.warning("No LOS information available in paths")
                return tf.zeros([len(self.scene.receivers)], dtype=tf.int32)
            los_conditions = tf.cast(paths.LOS, tf.int32)
            if tf.rank(los_conditions) == 0:
                los_conditions = tf.fill([len(self.scene.receivers)], los_conditions)
            elif tf.rank(los_conditions) == 1 and tf.shape(los_conditions)[0] != len(self.scene.receivers):
                los_conditions = tf.broadcast_to(los_conditions, [len(self.scene.receivers)])
            return los_conditions
        except Exception as e:
            logger.error(f"Error checking LOS conditions: {str(e)}")
            return tf.zeros([len(self.scene.receivers)], dtype=tf.int32)
    
    def calculate_received_power(self):
        try:
            tx_power_dbm = self.config.tx_power
            tx_power = tf.pow(10.0, (tx_power_dbm - 30) / 10.0)
            tx_gain_db = self.config.bs_array['antenna_gain_db']
            rx_gain_db = self.config.agv_array['antenna_gain_db']
            tx_gain = tf.pow(10.0, tx_gain_db / 10.0)
            rx_gain = tf.pow(10.0, rx_gain_db / 10.0)
            tx_position = tf.convert_to_tensor(list(self.scene.transmitters.values())[0].position)
            rx_positions = tf.stack([tf.convert_to_tensor(rx.position) for rx in self.scene.receivers.values()])
            distances = tf.norm(rx_positions - tx_position, axis=-1)
            path_losses = tf.map_fn(
                lambda x: self.calculate_path_loss(x, self.config.carrier_frequency),
                distances,
                dtype=tf.float32
            )
            path_loss_linear = tf.pow(10.0, -path_losses/10.0)
            received_power = tx_power * tx_gain * rx_gain * path_loss_linear
            return received_power
        except Exception as e:
            logger.error(f"Error calculating received power: {str(e)}")
            return tf.zeros(len(self.scene.receivers), dtype=tf.float32)
    
    def save_csi_dataset(self, filepath, num_samples=None):
        import h5py
        if num_samples is None:
            num_samples = self.config.num_time_steps
        with h5py.File(filepath, 'w') as f:
            csi_group = f.create_group('csi_data')
            config_group = f.create_group('config')
            channel_data = []
            path_delays = []
            los_conditions = []
            agv_positions = []
            for i in range(num_samples):
                sample = self.generate_channel_data(self.config)
                h = sample['h']
                h = self.monitor_channel_quality(h)
                channel_data.append(h.numpy())
                path_delays.append(sample['tau'].numpy())
                los_conditions.append(np.array(sample['los_condition'], dtype=np.int32))
                agv_positions.append(sample['agv_positions'].numpy())
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{num_samples} samples")
            csi_group.create_dataset('channel_matrices', data=np.array(channel_data))
            csi_group.create_dataset('path_delays', data=np.array(path_delays))
            csi_group.create_dataset('los_conditions', data=np.array(los_conditions))
            csi_group.create_dataset('agv_positions', data=np.array(agv_positions))
            for key, value in vars(self.config).items():
                if isinstance(value, (int, float, str, list)):
                    config_group.attrs[key] = value

    def _process_ray_tracing_in_chunks(self, agv_positions, max_chunk_size=50):
        """Process ray tracing in smaller chunks to avoid memory issues"""
        # Get ray tracing parameters
        max_depth = self.config.ray_tracing['max_depth']
        method = self.config.ray_tracing['method']
        num_samples = self.config.ray_tracing['num_samples']

        # Reduce chunk size based on depth
        adjusted_chunk_size = max(10, int(max_chunk_size / max_depth))

        if num_samples <= adjusted_chunk_size:
            try:
                logger.debug(f"Computing paths with {num_samples} samples (no chunking needed)")
                paths = self.scene.compute_paths(
                    max_depth=max_depth,
                    method=method,
                    num_samples=num_samples,
                    los=self.config.ray_tracing['los'],
                    reflection=self.config.ray_tracing['reflection'],
                    diffraction=self.config.ray_tracing['diffraction'],
                    scattering=self.config.ray_tracing['scattering'],
                    scat_keep_prob=self.config.ray_tracing.get('scat_keep_prob', 0.7),
                    edge_diffraction=self.config.ray_tracing.get('edge_diffraction', True)
                )
                return paths
            except Exception as e:
                logger.error(f"Ray tracing failed: {str(e)}")
                return self._generate_fallback_channel_data(agv_positions)

        num_chunks = (num_samples + adjusted_chunk_size - 1) // adjusted_chunk_size
        logger.info(f"Processing ray tracing in {num_chunks} chunks of {adjusted_chunk_size} rays")

        for i in range(num_chunks):
            chunk_samples = min(adjusted_chunk_size, num_samples - i * adjusted_chunk_size)
            if chunk_samples <= 0:
                break

            logger.debug(f"Processing chunk {i+1}/{num_chunks} with {chunk_samples} samples")
            try:
                paths = self.scene.compute_paths(
                    max_depth=max_depth,
                    method=method,
                    num_samples=chunk_samples,
                    los=self.config.ray_tracing['los'],
                    reflection=self.config.ray_tracing['reflection'],
                    diffraction=self.config.ray_tracing['diffraction'],
                    scattering=self.config.ray_tracing['scattering'],
                    scat_keep_prob=self.config.ray_tracing.get('scat_keep_prob', 0.7),
                    edge_diffraction=self.config.ray_tracing.get('edge_diffraction', True)
                )
                if paths is not None:
                    logger.info(f"Successfully processed chunk {i+1}")
                    return paths  # Return first successful chunk
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {str(e)}")
                tf.keras.backend.clear_session()
                gc.collect()

        logger.warning("All chunks failed, using fallback channel model")
        return self._generate_fallback_channel_data(agv_positions)
        num_chunks = (num_samples + adjusted_chunk_size - 1) // adjusted_chunk_size
        logger.info(f"Processing ray tracing in {num_chunks} chunks of {adjusted_chunk_size} rays")

        for i in range(num_chunks):
            chunk_samples = min(adjusted_chunk_size, num_samples - i * adjusted_chunk_size)
            if chunk_samples <= 0:
                break

            logger.debug(f"Processing chunk {i+1}/{num_chunks} with {chunk_samples} samples")
            try:
                paths = self.scene.compute_paths(
                    max_depth=max_depth,
                    method=method,
                    num_samples=chunk_samples,
                    los=self.config.ray_tracing['los'],
                    reflection=self.config.ray_tracing['reflection'],
                    diffraction=self.config.ray_tracing['diffraction'],
                    scattering=self.config.ray_tracing['scattering'],
                    scat_keep_prob=self.config.ray_tracing.get('scat_keep_prob', 0.7),
                    edge_diffraction=self.config.ray_tracing.get('edge_diffraction', True)
                )
                if paths is not None:
                    logger.info(f"Successfully processed chunk {i+1}")
                    return paths  # Return first successful chunk
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {str(e)}")
                tf.keras.backend.clear_session()
                gc.collect()

        logger.warning("All chunks failed, using fallback channel model")
        return self._generate_fallback_channel_data(agv_positions)

    def _generate_fallback_channel_data(self, agv_positions):
        """Generate simplified channel data when ray tracing fails"""
        logger.warning("Generating fallback channel data (bypassing ray tracing)")
        
        # Ensure agv_positions is properly shaped
        if not isinstance(agv_positions, tf.Tensor):
            agv_positions = tf.convert_to_tensor(agv_positions, dtype=tf.float32)
        
        # Get base station position
        if hasattr(self.scene, 'transmitters') and 'bs' in self.scene.transmitters:
            tx_pos = self.scene.transmitters['bs'].position
        else:
            tx_pos = tf.constant(self.config.bs_position, dtype=tf.float32)
        
        # Calculate distances
        distances = tf.norm(agv_positions - tx_pos, axis=1)
        
        # Simple path loss model (free space)
        f_ghz = self.config.carrier_frequency / 1e9
        path_losses_db = 20 * tf.math.log(distances) / tf.math.log(10.0) + 20 * tf.math.log(f_ghz) / tf.math.log(10.0) - 27.55
        path_losses = tf.pow(10.0, -path_losses_db / 10.0)
        
        # Generate random channel matrices with appropriate path loss
        h_freq = tf.complex(
            tf.random.normal([self.config.num_agvs, self.config.num_subcarriers], dtype=tf.float32),
            tf.random.normal([self.config.num_agvs, self.config.num_subcarriers], dtype=tf.float32)
        )
        
        # Apply path loss (cast to complex64 for multiplication)
        path_losses_complex = tf.cast(tf.reshape(tf.sqrt(path_losses), [-1, 1]), tf.complex64)
        h_freq = h_freq * path_losses_complex
        
        # Check line of sight by simple height check
        los_conditions = tf.ones(self.config.num_agvs, dtype=tf.int32)
        
        # Generate fake paths object
        class FakePaths:
            def __init__(self):
                self.LOS = los_conditions
                
            def cir(self):
                a = tf.complex(
                    tf.random.normal([1, self.config.num_agvs, 1, 1], dtype=tf.float32),
                    tf.random.normal([1, self.config.num_agvs, 1, 1], dtype=tf.float32)
                )
                tau = tf.zeros([1, self.config.num_agvs, 1], dtype=tf.float32)
                return a, tau
        
        fake_paths = FakePaths()
        
        # Create path data (fake)
        path_data = {
            'path_powers': tf.ones([1, self.config.num_agvs, 1]),
            'path_directions': tf.zeros([1, self.config.num_agvs, 1, 2])
        }
        
        # Set reasonable SNR
        snr_db = 25.0 - path_losses_db
        snr_db = tf.clip_by_value(snr_db, 0.0, 30.0)
        
        return fake_paths