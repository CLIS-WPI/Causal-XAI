import tensorflow as tf
import numpy as np
import sionna
from scene_setup import setup_scene
from sionna.constants import SPEED_OF_LIGHT
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.rt import Scene, Transmitter, Receiver, RIS, PlanarArray, RadioMaterial, Paths
from sionna.rt import DiscretePhaseProfile, CellGrid
import logging
from sionna.channel.utils import subcarrier_frequencies

# Initialize logger
logger = logging.getLogger(__name__)

class SmartFactoryChannel:
    """Smart Factory Channel Generator using Sionna"""
    
    def __init__(self, config, scene=None):
        """Initialize the environment with configuration and scene setup."""
        try:
            self.config = config
            sionna.config.xla_compat = True
            tf.random.set_seed(config.seed if hasattr(config, 'seed') else 42)

            # Initialize position tracking
            self.positions_history = [[] for _ in range(config.num_agvs)]
            self.agv_positions = self._generate_initial_agv_positions()

            # Initialize scene
            self.scene = scene if scene is not None else setup_scene(config)
            
            # Configure antenna arrays
            self._setup_antenna_arrays()
            
            # Verify scene configuration
            self.verify_scene_configuration()

        except Exception as e:
            raise RuntimeError(f"Channel initialization failed: {str(e)}") from e

    def verify_scene_configuration(self):
        """Verify scene configuration before channel generation"""
        if self.scene is None:
            raise RuntimeError("Scene not initialized")
        
        if not hasattr(self, 'bs_array') or not hasattr(self, 'agv_array'):
            raise RuntimeError("Antenna arrays not properly configured")

    def _setup_antenna_arrays(self):
        """Configure antenna arrays for BS and AGVs"""
        try:
            self.bs_array = PlanarArray(
                num_rows=self.config.bs_array[0],
                num_cols=self.config.bs_array[1],
                vertical_spacing=self.config.bs_array_spacing,
                horizontal_spacing=self.config.bs_array_spacing,
                pattern=self.config.bs_array_pattern,
                polarization=self.config.bs_polarization,
                dtype=tf.complex64
            )
            
            self.agv_array = PlanarArray(
                num_rows=self.config.agv_array[0],
                num_cols=self.config.agv_array[1],
                vertical_spacing=self.config.agv_array_spacing,
                horizontal_spacing=self.config.agv_array_spacing,
                pattern=self.config.agv_array_pattern,
                polarization=self.config.agv_polarization,
                dtype=tf.complex64
            )
            
        except Exception as e:
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
        """Update AGV positions"""
        current_positions = self.agv_positions.numpy()
        new_positions = []
        
        for i in range(self.config.num_agvs):
            current_pos = current_positions[i]
            new_pos = current_pos + np.array([0.1, 0.1, 0]) * time_step
            
            new_pos[0] = np.clip(new_pos[0], 0, self.config.room_dim[0])
            new_pos[1] = np.clip(new_pos[1], 0, self.config.room_dim[1])
            new_pos[2] = self.config.agv_height
            
            new_positions.append(new_pos)
            self.positions_history[i].append(new_pos.copy())
        
        return tf.constant(new_positions, dtype=tf.float32)

    def monitor_channel_quality(self, h):
        """Monitor channel matrix quality"""
        nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(h), tf.int32))
        inf_count = tf.reduce_sum(tf.cast(tf.math.is_inf(h), tf.int32))
        
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Channel matrix contains {nan_count} NaN and {inf_count} Inf values")
        
        avg_power = tf.reduce_mean(tf.abs(h)**2)
        logger.info(f"Average channel power: {avg_power}")
        
        return h

    def generate_channel_data(scene, config):
        """Generate channel data using ray tracing"""
        try:
            print("Generating channel data...")
            
            # Compute paths using ray tracing with correct method
            paths = scene.compute_paths(
                max_depth=config.ray_tracing['max_depth'],
                method="fibonacci",  # Changed from "image" to "fibonacci"
                num_samples=config.ray_tracing['num_samples'],
                los=config.ray_tracing['los'],
                reflection=config.ray_tracing['reflection'],
                diffraction=config.ray_tracing['diffraction'],
                scattering=config.ray_tracing['scattering'],
                ris=config.ray_tracing['ris'],
                scat_keep_prob=config.ray_tracing['scat_keep_prob'],
                edge_diffraction=config.ray_tracing['edge_diffraction']
            )
            
            if paths is None:
                raise RuntimeError("Path computation failed")

            # Get channel impulse responses with explicit normalization
            a, tau = paths.cir(normalize=True)
            
            # Calculate frequencies for the subcarriers
            frequencies = sionna.channel.utils.subcarrier_frequencies(
                num_subcarriers=config.num_subcarriers,
                subcarrier_spacing=config.subcarrier_spacing
            )
            
            # Convert to OFDM channel with improved stability
            h_freq = cir_to_ofdm_channel(
                frequencies=frequencies,
                a=tf.cast(a, tf.complex64),
                tau=tf.cast(tau, tf.float32),
                normalize=True
            )
            
            # Create channel data dictionary
            channel_data = {
                'channel_matrices': h_freq,
                'path_delays': tau,
                'los_conditions': paths.LOS,
                'agv_positions': tf.stack([rx.position for rx in scene.receivers.values()])
            }
            
            print("Channel data generation completed")
            return channel_data
            
        except Exception as e:
            logger.error(f"Error generating channel data: {str(e)}")
            raise
    # In channel_generator.py, add these new methods:

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
        """Enhanced tracking of LOS/NLOS conditions"""
        try:
            # Get paths from ray tracing
            paths = self.scene.compute_paths(
                max_depth=self.config.ray_tracing['max_depth'],
                method=self.config.ray_tracing['method']
            )
            
            if paths is None:
                raise ValueError("No paths computed")
                
            # Extract LOS conditions
            los_conditions = paths.LOS
            
            # Calculate statistics
            total_paths = tf.size(los_conditions)
            los_paths = tf.reduce_sum(tf.cast(los_conditions, tf.int32))
            nlos_paths = total_paths - los_paths
            
            nlos_stats = {
                'los_ratio': float(los_paths) / total_paths,
                'nlos_ratio': float(nlos_paths) / total_paths,
                'total_paths': total_paths,
                'blocked_paths': nlos_paths
            }
            
            return nlos_stats, los_conditions
            
        except Exception as e:
            logger.error(f"Error tracking LOS/NLOS paths: {str(e)}")
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