#channel_generator.py#
import time
import tensorflow as tf
import numpy as np
import sionna
from scene_setup import setup_scene
from sionna.constants import SPEED_OF_LIGHT
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray, RadioMaterial, Paths
from sionna.rt import DiscretePhaseProfile, CellGrid
import logging
from sionna.channel.utils import subcarrier_frequencies
from beam_manager import BeamManager
from agv_path_manager import AGVPathManager
from scipy.special import erfc
# Initialize logger
logger = logging.getLogger(__name__)

class SmartFactoryChannel:
    """Smart Factory Channel Generator using Sionna"""
    
    def __init__(self, config, scene=None):
        """Initialize the environment with configuration and scene setup."""
        try:
            logger.debug("=== Initializing SmartFactoryChannel ===")
            logger.debug(f"Config parameters:")
            logger.debug(f"- Number of AGVs: {config.num_agvs}")
            logger.debug(f"- Room dimensions: {config.room_dim}")
            logger.debug(f"- Carrier frequency: {config.carrier_frequency} Hz")
            logger.debug(f"- Ray tracing config: {config.ray_tracing}")
            
            self.config = config
            sionna.config.xla_compat = True
            tf.random.set_seed(config.seed if hasattr(config, 'seed') else 42)
            logger.debug(f"Random seed set to: {config.seed if hasattr(config, 'seed') else 42}")

            # Initialize indoor factory parameters first
            self._setup_indoor_factory_params()
            logger.debug("Indoor factory parameters initialized")
            
            # Initialize position tracking
            self.positions_history = [[] for _ in range(config.num_agvs)]
            self.agv_positions = self._generate_initial_agv_positions()
            logger.debug(f"Initial AGV positions:\n{self.agv_positions.numpy()}")

            # Initialize scene
            logger.debug("Setting up scene...")
            self.scene = scene if scene is not None else setup_scene(config)
            logger.debug("Scene setup completed")
            
            # Configure antenna arrays
            logger.debug("Configuring antenna arrays...")
            self._setup_antenna_arrays()
            logger.debug("Antenna arrays configured successfully")
            
            # Verify scene configuration
            self.verify_scene_configuration()
            logger.debug("Scene configuration verified successfully")
            
            # Initialize managers
            self.path_manager = AGVPathManager(config)
            self.beam_manager = BeamManager(config)
            
            self.performance_metrics = {
            'ber_history': [],
            'snr_history': [],
            'packet_stats': {
                'total': 0,
                'successful': 0
                }
                }   
            
        except Exception as e:
            logger.error(f"Channel initialization failed: {str(e)}")
        raise RuntimeError(f"Channel initialization failed: {str(e)}") from e

    def calculate_ber(self, signal_data):
        """
        Calculate Bit Error Rate (BER) from signal data
        """
        try:
            # Calculate SNR from channel data
            if 'beam_metrics' in signal_data and 'snr_db' in signal_data['beam_metrics']:
                snr_db = np.mean(signal_data['beam_metrics']['snr_db'])
                # Convert SNR from dB to linear scale
                snr_linear = 10**(snr_db/10)
                # Theoretical BER calculation for QPSK modulation
                ber = 0.5 * erfc(np.sqrt(snr_linear))
            else:
                ber = 1.0  # Worst case if SNR data is not available
            
            # Store BER history
            self.performance_metrics['ber_history'].append({
                'timestamp': time.time(),
                'value': float(ber)
            })
            
            return ber
            
        except Exception as e:
            logger.error(f"Error calculating BER: {str(e)}")
            return 1.0  # Return worst case BER on error
    
    def track_performance_metrics(self, signal_data):
        """
        Track all performance metrics including BER, PSR, and SNR
        """
        metrics = {
            'packet_success_rate': self.calculate_psr(),
            'ber_during_switch': self.calculate_ber(signal_data),
            'snr_variations': self.performance_metrics['snr_history']
        }
        
        return metrics
    
    def calculate_psr(self):
        """
        Calculate Packet Success Rate
        """
        if self.performance_metrics['packet_stats']['total'] == 0:
            return 0.0
        return (self.performance_metrics['packet_stats']['successful'] / 
                self.performance_metrics['packet_stats']['total'])
    
    def _setup_indoor_factory_params(self):
        """Setup indoor factory specific parameters"""
        try:
            # Indoor factory mmWave parameters
            self.inf_params = {
                'los_k_factor': 17.0,      # Rician K-factor for LOS
                'nlos_sigma': 3.5,         # Rayleigh sigma for NLOS
                'path_loss_exp': 1.6,       # Optimize from 1.8
                'shadow_std': 2.5,          # Further reduce from 3.0 # Shadow fading std dev (dB)
                'penetration_loss': 8.0,  # Material penetration loss (dB)
                'reflection_coeff': 0.85     # Reflection coefficient
            }
            logger.debug(f"Indoor factory parameters initialized: {self.inf_params}")
        except Exception as e:
            logger.error(f"Error setting up indoor factory parameters: {str(e)}")
            raise

    def calculate_path_loss(self, distance, frequency):
        """Calculate path loss with enhanced indoor factory model"""
        try:
            # Basic free space path loss
            wavelength = SPEED_OF_LIGHT / frequency
            basic_loss = 20 * tf.math.log(4 * np.pi * distance / wavelength) / tf.math.log(10.0)
            
            # Add distance-dependent loss
            n = self.inf_params['path_loss_exp']
            dist_loss = 10 * n * tf.math.log(distance) / tf.math.log(10.0)
            
            # Add shadow fading
            shadow_std = self.inf_params['shadow_std']
            shadow_fading = tf.random.normal([], mean=0.0, stddev=shadow_std)
            
            # Add penetration loss for obstacles
            penetration = self.inf_params['penetration_loss']
            
            # Calculate total path loss
            total_loss = basic_loss + dist_loss + shadow_fading + penetration
            
            logger.debug(f"Path loss components:")
            logger.debug(f"- Basic loss: {float(basic_loss):.2f} dB")
            logger.debug(f"- Distance loss: {float(dist_loss):.2f} dB")
            logger.debug(f"- Shadow fading: {float(shadow_fading):.2f} dB")
            logger.debug(f"- Total loss: {float(total_loss):.2f} dB")
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error calculating path loss: {str(e)}")
            raise
    
    def calculate_snr(self, h_freq, config, path_losses=None):
        """
        Calculate SNR for the channel with proper parameter handling and detailed logging
        
        Args:
            h_freq: Channel frequency response
            config: Configuration object containing system parameters
            path_losses: Path losses for each receiver
            
        Returns:
            dict: Dictionary containing SNR metrics including average_snr and detailed beam metrics
        """
        try:
            # Ensure path_losses is a tensor
            if not isinstance(path_losses, tf.Tensor):
                path_losses = tf.convert_to_tensor(path_losses, dtype=tf.float32)
                logger.debug(f"Converted path_losses to tensor with shape: {path_losses.shape}")
            
            # System parameters for indoor factory scenario
            tx_power_dbm = 65  # Transmit power in dBm for mmWave indoor BS
            tx_antenna_gain_db = 40  # BS antenna array gain
            rx_antenna_gain_db = 25   # AGV antenna gain
            min_snr_threshold = 10.0  # Minimum usable SNR in dB
            
            logger.debug(f"Initial parameters:")
            logger.debug(f"- TX power: {tx_power_dbm} dBm")
            logger.debug(f"- TX antenna gain: {tx_antenna_gain_db} dB")
            logger.debug(f"- RX antenna gain: {rx_antenna_gain_db} dB")
            
            # Convert powers from dB to linear scale
            tx_power = tf.pow(10.0, (tx_power_dbm - 30) / 10.0)  # Convert dBm to W
            tx_gain = tf.pow(10.0, tx_antenna_gain_db / 10.0)
            rx_gain = tf.pow(10.0, rx_antenna_gain_db / 10.0)
            
            logger.debug(f"Linear scale conversions:")
            logger.debug(f"- TX power: {float(tx_power):.2e} W")
            logger.debug(f"- TX gain: {float(tx_gain):.2f}")
            logger.debug(f"- RX gain: {float(rx_gain):.2f}")
            
            # Noise calculation parameters
            k_boltzmann = 1.380649e-23
            temperature = 290  # Room temperature in Kelvin
            bandwidth = config.subcarrier_spacing * config.num_subcarriers
            noise_figure_db = 5  # Reduced for better performance
            implementation_loss_db = 1  # Reduced for better performance
            
            logger.debug(f"Noise parameters:")
            logger.debug(f"- Bandwidth: {bandwidth/1e6:.2f} MHz")
            logger.debug(f"- Noise figure: {noise_figure_db} dB")
            logger.debug(f"- Implementation loss: {implementation_loss_db} dB")
            
            # Calculate noise power
            thermal_noise = k_boltzmann * temperature * bandwidth
            noise_figure_linear = tf.pow(10.0, noise_figure_db / 10.0)
            implementation_loss_linear = tf.pow(10.0, implementation_loss_db / 10.0)
            total_noise_power = thermal_noise * noise_figure_linear * implementation_loss_linear
            
            logger.debug(f"Noise power calculations:")
            logger.debug(f"- Thermal noise: {thermal_noise:.2e} W")
            logger.debug(f"- Total noise power: {total_noise_power:.2e} W")
            
            # Calculate channel power
            channel_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=-1)
            logger.debug(f"Channel power: {float(tf.reduce_mean(channel_power)):.2e}")
            
            # Calculate signal power with antenna gains and path losses
            signal_power = channel_power * tx_power * tx_gain * rx_gain
            
            # Apply path losses if provided
            if path_losses is not None:
                path_loss_linear = tf.pow(10.0, -path_losses / 10.0)
                signal_power = signal_power * tf.cast(path_loss_linear, tf.float32)
                logger.debug(f"Applied path losses. New signal power mean: {float(tf.reduce_mean(signal_power)):.2e} W")
            
            logger.debug(f"Signal power after gains: {float(tf.reduce_mean(signal_power)):.2e} W")
            
            # SNR calculation
            snr_linear = signal_power / total_noise_power
            snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)
            
            logger.debug(f"SNR calculations:")
            logger.debug(f"- Linear SNR: {float(tf.reduce_mean(snr_linear)):.2e}")
            logger.debug(f"- SNR (dB): {float(tf.reduce_mean(snr_db)):.2f} dB")
            
            # Clip SNR to realistic range for indoor factory
            max_snr_db =  30.0  # Increased maximum expected SNR
            min_snr_db = -10.0  # Lowered minimum usable SNR
            snr_db_clipped = tf.clip_by_value(snr_db, min_snr_db, max_snr_db)
            logger.debug(f"SNR clipping bounds:")
            logger.debug(f"- Max SNR: {max_snr_db} dB (limited by practical factory conditions)")
            logger.debug(f"- Min SNR: {min_snr_db} dB (minimum for reliable communication)")
            average_snr = float(tf.reduce_mean(snr_db_clipped))
            
            logger.debug(f"Final SNR after clipping:")
            logger.debug(f"- Average SNR: {average_snr:.2f} dB")
            
            return {
            'average_snr': average_snr,
            'beam_metrics': {
                'snr_db': snr_db_clipped.numpy(),
                'raw_snr_db': snr_db.numpy(),  # Before clipping
                'clipping_stats': {
                    'max_snr_db': max_snr_db,
                    'min_snr_db': min_snr_db,
                    'clipped_samples': tf.reduce_sum(
                        tf.cast(tf.logical_or(
                            snr_db > max_snr_db, 
                            snr_db < min_snr_db), 
                        tf.int32)).numpy()
                },
                'avg_power': float(tf.reduce_mean(signal_power)),
                'tx_power_dbm': tx_power_dbm,
                'antenna_gains': {
                    'tx_gain_db': tx_antenna_gain_db,
                    'rx_gain_db': rx_antenna_gain_db
                },
                'noise_power': float(total_noise_power)
            }
        }
            
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            logger.error(f"Channel shape: {h_freq.shape}")
            logger.error(f"Path losses shape: {path_losses.shape if path_losses is not None else None}")
            logger.error(f"Debug values:")
            logger.error(f"- TX power: {tx_power_dbm} dBm")
            logger.error(f"- Channel power: {float(tf.reduce_mean(channel_power)) if 'channel_power' in locals() else 'N/A'}")
            logger.error(f"- Signal power: {float(tf.reduce_mean(signal_power)) if 'signal_power' in locals() else 'N/A'}")
            logger.error(f"- Noise power: {total_noise_power if 'total_noise_power' in locals() else 'N/A'}")
            raise

    def verify_scene_configuration(self):
        """Verify scene configuration before channel generation"""
        if self.scene is None:
            raise RuntimeError("Scene not initialized")
        
        if not hasattr(self, 'bs_array') or not hasattr(self, 'agv_array'):
            raise RuntimeError("Antenna arrays not properly configured")

    def _setup_antenna_arrays(self):
        """Configure antenna arrays for BS and AGVs"""
        try:
            logger.debug("=== Setting up antenna arrays ===")
            
            # Updated BS array configuration for 16x4
            self.bs_array = PlanarArray(
                num_rows=16,  # Changed to 16 rows
                num_cols=4,   # Changed to 4 columns
                vertical_spacing=0.5 * self.config.wavelength,
                horizontal_spacing=0.5 * self.config.wavelength,
                pattern="tr38901",  # Using 3GPP TR 38.901 pattern
                polarization="VH",  # Dual polarization
                dtype=tf.complex64
            )
            logger.debug("BS array configured successfully")
            
            # Simplified AGV array configuration
            self.agv_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5 * self.config.wavelength,
                horizontal_spacing=0.5 * self.config.wavelength,
                pattern="dipole",
                polarization="cross",
                dtype=tf.complex64
            )
            logger.debug("AGV array configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup antenna arrays: {str(e)}")
            raise RuntimeError(f"Failed to setup antenna arrays: {str(e)}")

    def _generate_initial_agv_positions(self):
        """Generate initial AGV positions from configuration
        
        Returns:
            tf.Tensor: Initial positions for all AGVs with shape [num_agvs, 3]
        """
        if not hasattr(self.config, 'agv_positions'):
            raise AttributeError("AGV positions not defined in config")
            
        return tf.constant(self.config.agv_positions, dtype=tf.float32)

    def _update_agv_positions(self, time_step):
        """Update AGV positions following predefined trajectories"""
        current_positions = self.agv_positions.numpy()
        new_positions = []
        
        for i in range(self.config.num_agvs):
            current_pos = current_positions[i]
            
            # Get next position from path manager
            # Change this line to use proper AGV ID format
            new_pos = self.path_manager.get_next_position(f'agv_{i+1}', current_pos)
            
            # Store new position
            new_positions.append(new_pos)
            self.positions_history[i].append(new_pos.copy())
        
        self.agv_positions = tf.convert_to_tensor(new_positions, dtype=self.config.real_dtype)

    def get_agv_status(self):
        """Get current status of all AGVs"""
        status = {}
        for i in range(self.config.num_agvs):
            agv_id = f'agv_{i+1}'
            status[agv_id] = self.path_manager.get_current_status(agv_id)
        return status
        

    def simulate_movement(self, num_steps):
        """Simulate AGV movements for specified number of steps"""
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
        """Enhanced channel matrix quality monitoring"""
        nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(h), tf.int32))
        inf_count = tf.reduce_sum(tf.cast(tf.math.is_inf(h), tf.int32))
        
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Channel matrix contains {nan_count} NaN and {inf_count} Inf values")
        
        # Calculate channel quality metrics
        avg_power = tf.reduce_mean(tf.abs(h)**2)
        peak_power = tf.reduce_max(tf.abs(h)**2)
        min_power = tf.reduce_min(tf.abs(h)**2)
        
        # Calculate condition number for channel matrix
        s = tf.linalg.svd(h, compute_uv=False)
        condition_number = s[0] / s[-1]
        
        logger.info(f"Channel Quality Metrics:")
        logger.info(f"- Average power: {avg_power}")
        logger.info(f"- Peak power: {peak_power}")
        logger.info(f"- Minimum power: {min_power}")
        logger.info(f"- Condition number: {condition_number}")
        
        return h
    # Add new method for beam switching analysis
    def analyze_beam_switching(self, channel_data):
        """Analyze channel conditions for beam switching"""
        try:
            h_freq = channel_data['channel_matrices']
            los_conditions = channel_data['los_conditions']
            
            # Convert los_conditions to tensor if it's not already
            if not isinstance(los_conditions, tf.Tensor):
                los_conditions = tf.convert_to_tensor(los_conditions, dtype=tf.int32)
            
            # Calculate SNR for each beam
            noise_power = 1e-13  # Typical thermal noise power
            signal_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=-1)
            snr_db = 10 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)
            
            # Detect significant SNR drops
            snr_threshold = self.config.beamforming['min_snr_threshold']
            beam_switches_needed = tf.where(snr_db < snr_threshold, True, False)
            
            return {
                'snr_db': snr_db.numpy(),
                'beam_switches_needed': beam_switches_needed.numpy(),
                'los_conditions': los_conditions.numpy()  # Now los_conditions is guaranteed to be a tensor
            }
        except Exception as e:
            logger.error(f"Error in beam switching analysis: {str(e)}")
            raise
    
    def generate_channel_data(self, config):
        """Generate channel data using ray tracing"""
        try:
            logger.debug("=== Generating channel data ===")
            
            # Compute paths using parameters from config
            paths = self.scene.compute_paths(
                max_depth=config.ray_tracing['max_depth'],
                method=config.ray_tracing['method'],
                num_samples=config.ray_tracing['num_samples'],
                los=config.ray_tracing['los'],
                reflection=config.ray_tracing['reflection'],
                diffraction=config.ray_tracing['diffraction'],
                scattering=config.ray_tracing['scattering'],
                scat_keep_prob=config.ray_tracing['scat_keep_prob'],
                edge_diffraction=config.ray_tracing['edge_diffraction']
            )
            
            if paths is None:
                logger.error("Path computation failed - no paths found")
                raise RuntimeError("Path computation failed")

            # Get channel impulse responses and convert to tensors
            a, tau = paths.cir()
            a = tf.convert_to_tensor(a, dtype=tf.complex64)
            tau = tf.convert_to_tensor(tau, dtype=tf.float32)

            # Rest of your existing code remains the same...
            los_conditions = tf.convert_to_tensor(paths.LOS, dtype=tf.int32)
            
            frequencies = subcarrier_frequencies(
                num_subcarriers=config.num_subcarriers,
                subcarrier_spacing=config.subcarrier_spacing
            )
            
            # Process channel data in batches using TensorArray
            batch_size = tf.constant(32)
            num_samples = tf.shape(a)[0]
            num_batches = tf.cast(tf.math.ceil(tf.cast(num_samples, tf.float32) / tf.cast(batch_size, tf.float32)), tf.int32)
            
            h_freq_list = tf.TensorArray(tf.complex64, size=num_batches)
            
            for i in tf.range(num_batches):
                start_idx = i * batch_size
                end_idx = tf.minimum((i + 1) * batch_size, num_samples)
                
                # Changed tf.slice to tf.gather with tf.range
                batch_a = tf.gather(a, tf.range(start_idx, end_idx))
                batch_tau = tf.gather(tau, tf.range(start_idx, end_idx))
                
                logger.debug(f"Processing batch {i//batch_size + 1}, shape: {batch_a.shape}")
                
                batch_h_freq = cir_to_ofdm_channel(
                    frequencies=frequencies,
                    a=batch_a,
                    tau=batch_tau,
                    normalize=True
                )
                h_freq_list = h_freq_list.write(i, batch_h_freq)
            
            h_freq = h_freq_list.concat()
            logger.debug(f"Combined h_freq shape: {h_freq.shape}")
            
            # Calculate path losses
            tx_position = tf.convert_to_tensor(list(self.scene.transmitters.values())[0].position)
            rx_positions = tf.stack([tf.convert_to_tensor(rx.position) for rx in self.scene.receivers.values()])
            distances = tf.norm(rx_positions - tx_position, axis=-1)
            
            path_losses = tf.map_fn(
                lambda x: self.calculate_path_loss(x, config.carrier_frequency),
                distances,
                dtype=tf.float32
            )
            logger.debug(f"Calculated path losses: {path_losses}")
            
            path_losses_linear = tf.pow(10.0, -path_losses/20.0)
            path_losses_linear = tf.cast(path_losses_linear, tf.complex64)
            
            path_losses_shape = tf.ones_like(tf.shape(h_freq))
            path_losses_shape = tf.tensor_scatter_nd_update(
                path_losses_shape,
                [[1]],
                [tf.shape(path_losses)[0]]
            )
            path_losses_linear = tf.reshape(path_losses_linear, path_losses_shape)
            
            h_freq = h_freq * path_losses_linear
            path_losses_tensor = tf.identity(path_losses)
            
            snr_metrics = self.calculate_snr(h_freq, config, path_losses_tensor)
            
            channel_data = {
                'channel_matrices': h_freq,
                'path_delays': tau,
                'los_conditions': los_conditions,
                'agv_positions': rx_positions,
                'num_paths': tf.size(paths.LOS),
                'path_losses': path_losses_tensor,
                'reflection_paths': getattr(paths, 'reflection_paths', None),
                'diffraction_paths': getattr(paths, 'diffraction_paths', None),
                'average_snr': snr_metrics['average_snr'],
                'beam_metrics': snr_metrics['beam_metrics']
            }
            
            beam_analysis = self.analyze_beam_switching(channel_data)
            channel_data.update(beam_analysis)
            
            return channel_data

        except Exception as e:
            tf.print("Error in channel data generation:", e)
            tf.print("Shapes - a:", tf.shape(a), "tau:", tf.shape(tau))
            if 'h_freq' in locals():
                tf.print("h_freq shape:", tf.shape(h_freq))
            if 'path_losses' in locals():
                tf.print("path_losses shape:", tf.shape(path_losses))
            raise

    def calculate_doppler_shift(self):
        """Calculate Doppler shift based on AGV velocity and carrier frequency"""
        try:
            # Get current and previous positions
            current_positions = self.agv_positions
            previous_positions = tf.stack([self.positions_history[i][-2] if len(self.positions_history[i]) > 1 
                                        else current_positions[i] for i in range(self.config.num_agvs)])
            
            # Calculate velocity vectors (m/s)
            time_step = 1.0 / self.config.sampling_frequency
            velocities = (current_positions - previous_positions) / time_step
            
            # Calculate relative velocities along BS-AGV paths
            bs_position = tf.constant(self.config.bs_position, dtype=tf.float32)
            bs_to_agv = tf.nn.l2_normalize(current_positions - bs_position, axis=1)
            relative_velocities = tf.reduce_sum(velocities * bs_to_agv, axis=1)
            
            # Calculate Doppler shift: f_d = (v/c) * f_c
            doppler_shifts = (relative_velocities / SPEED_OF_LIGHT) * self.config.carrier_frequency
            
            return doppler_shifts, velocities
            
        except Exception as e:
            logger.error(f"Error calculating Doppler shift: {str(e)}")
            raise

    def track_los_nlos_paths(self):
        """Track and analyze LOS/NLOS paths with improved error handling"""
        try:
            logger.debug("Computing paths for LOS/NLOS analysis...")
            
            # Get paths from ray tracing with full configuration
            paths = self.scene.compute_paths(
                max_depth=self.config.ray_tracing['max_depth'],
                method=self.config.ray_tracing['method'],
                num_samples=self.config.ray_tracing['num_samples'],
                los=True,  # Ensure LOS paths are computed
                reflection=self.config.ray_tracing['reflection'],
                diffraction=self.config.ray_tracing['diffraction'],
                scattering=self.config.ray_tracing['scattering']
            )
            
            if paths is None:
                logger.warning("No paths computed in ray tracing")
                return {
                    'los_ratio': 0.0,
                    'nlos_ratio': 0.0,
                    'total_paths': 0,
                    'blocked_paths': 0
                }, None
                    
            # Extract LOS conditions and ensure it's a tensor
            los_conditions = paths.LOS
            if not isinstance(los_conditions, tf.Tensor):
                los_conditions = tf.convert_to_tensor(los_conditions, dtype=tf.float32)
                
            # Calculate statistics with type checking
            total_paths = tf.cast(tf.size(los_conditions), tf.float32)
            
            # Early warning for no paths
            if total_paths == 0:
                logger.warning("No paths found in channel computation")
                return {
                    'los_ratio': 0.0,
                    'nlos_ratio': 0.0,
                    'total_paths': 0,
                    'blocked_paths': 0
                }, los_conditions
                
            # Calculate LOS and NLOS paths with safe casting
            los_paths = tf.cast(tf.reduce_sum(tf.cast(los_conditions, tf.int32)), tf.float32)
            
            # Check for NaN values
            if tf.reduce_any(tf.math.is_nan(los_paths)) or tf.reduce_any(tf.math.is_nan(total_paths)):
                logger.warning("NaN values detected in path calculations")
                return {
                    'los_ratio': 0.0,
                    'nlos_ratio': 0.0,
                    'total_paths': 0,
                    'blocked_paths': 0
                }, los_conditions
                
            nlos_paths = total_paths - los_paths
            
            # Safe division for ratios
            los_ratio = tf.where(total_paths > 0, 
                                los_paths / total_paths,
                                tf.zeros_like(total_paths))
            nlos_ratio = tf.where(total_paths > 0,
                                nlos_paths / total_paths,
                                tf.zeros_like(total_paths))
            
            # Convert to numpy and handle potential conversion errors
            try:
                nlos_stats = {
                    'los_ratio': float(los_ratio.numpy()),
                    'nlos_ratio': float(nlos_ratio.numpy()),
                    'total_paths': int(total_paths.numpy()),
                    'blocked_paths': int(nlos_paths.numpy())
                }
                
                # Log detailed statistics
                logger.debug(f"Path Analysis Results:")
                logger.debug(f"- Total paths: {nlos_stats['total_paths']}")
                logger.debug(f"- LOS ratio: {nlos_stats['los_ratio']:.2f}")
                logger.debug(f"- NLOS ratio: {nlos_stats['nlos_ratio']:.2f}")
                logger.debug(f"- Blocked paths: {nlos_stats['blocked_paths']}")
                
                return nlos_stats, los_conditions
                
            except Exception as e:
                logger.error(f"Error converting path statistics: {str(e)}")
                return {
                    'los_ratio': 0.0,
                    'nlos_ratio': 0.0,
                    'total_paths': 0,
                    'blocked_paths': 0
                }, los_conditions
                
        except Exception as e:
            logger.error(f"Error tracking LOS/NLOS paths: {str(e)}")
            raise

    def apply_fading(self, channel, los_condition):
        """Apply appropriate fading model based on LOS condition"""
        try:
            if los_condition:
                # Rician fading for LOS
                k_factor = self.inf_params['los_k_factor']
                return self._apply_rician_fading(channel, k_factor)
            else:
                # Rayleigh fading for NLOS
                sigma = self.inf_params['nlos_sigma']
                return self._apply_rayleigh_fading(channel, sigma)
        except Exception as e:
            logger.error(f"Error applying fading: {str(e)}")
            raise

    def _apply_rician_fading(self, channel, k_factor):
        """Apply Rician fading to channel"""
        try:
            # Convert K-factor to linear scale
            k_linear = tf.pow(10.0, k_factor/10.0)
            
            # Generate complex Gaussian components
            shape = tf.shape(channel)
            real = tf.random.normal(shape, mean=0.0, stddev=1.0)
            imag = tf.random.normal(shape, mean=0.0, stddev=1.0)
            
            # Combine for Rician fading
            los_component = tf.sqrt(k_linear/(k_linear + 1))
            nlos_component = tf.sqrt(1/(k_linear + 1)) * tf.complex(real, imag)
            
            return channel * (los_component + nlos_component)
        except Exception as e:
            logger.error(f"Error in Rician fading: {str(e)}")
            raise

    def _apply_rayleigh_fading(self, channel, sigma):
        """Apply Rayleigh fading to channel"""
        try:
            shape = tf.shape(channel)
            real = tf.random.normal(shape, mean=0.0, stddev=sigma)
            imag = tf.random.normal(shape, mean=0.0, stddev=sigma)
            return channel * tf.complex(real, imag)
        except Exception as e:
            logger.error(f"Error in Rayleigh fading: {str(e)}")
            raise
    
    def calculate_beam_performance(self):
        """Track beam performance metrics"""
        try:
            # Get channel matrix
            h = self.monitor_channel_quality(self.generate_channel()['h'])
            
            # Calculate SNR for each beam
            noise_power = 1e-13  # Typical thermal noise power
            signal_power = tf.reduce_mean(tf.abs(h)**2, axis=-1)
            snr_db = 10 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)
            
            # Calculate beam metrics
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
    # In channel_generator.py, add enhanced monitoring
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
    
    def save_csi_dataset(self, filepath, num_samples=None):
        """Save CSI dataset with quality monitoring"""
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
            quality_metrics = []
            
            for i in range(num_samples):
                sample = self.generate_channel()
                h = sample['h']
                
                # Monitor channel quality
                h = self.monitor_channel_quality(h)
                
                channel_data.append(h.numpy())
                path_delays.append(sample['tau'].numpy())
                los_conditions.append(np.array(sample['los_condition'], dtype=np.int32))
                agv_positions.append(sample['agv_positions'].numpy())
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{num_samples} samples")
            
            # Save datasets
            csi_group.create_dataset('channel_matrices', data=np.array(channel_data))
            csi_group.create_dataset('path_delays', data=np.array(path_delays))
            csi_group.create_dataset('los_conditions', data=np.array(los_conditions))
            csi_group.create_dataset('agv_positions', data=np.array(agv_positions))
            
            # Save configuration
            for key, value in vars(self.config).items():
                if isinstance(value, (int, float, str, list)):
                    config_group.attrs[key] = value