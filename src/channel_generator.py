import tensorflow as tf
import numpy as np
import sionna
from scene_setup import setup_scene
from sionna.constants import SPEED_OF_LIGHT
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.rt import Scene, Transmitter, Receiver, RIS, PlanarArray, RadioMaterial, Paths
from sionna.rt import DiscretePhaseProfile, CellGrid
import logger

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
        """Generate initial AGV positions"""
        return tf.constant([
            [12.0, 5.0, self.config.agv_height],
            [8.0, 15.0, self.config.agv_height]
        ], dtype=tf.float32)

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

    def generate_channel(self):
        """Generate channel matrices with RIS modeling"""
        try:
            # Update and verify positions
            current_positions = self._update_agv_positions(self.config.simulation['time_step'])
            current_positions = tf.expand_dims(current_positions, axis=0)

            for i in range(self.config.num_agvs):
                agv = self.scene.get(f"agv_{i}")
                if agv is not None:
                    agv.position = current_positions[0, i].numpy()

            # Generate paths with improved stability
            paths = self.scene.compute_paths(
                max_depth=2,  # Fixed depth for stability
                method="image",  # Fixed method for stability
                num_samples=self.config.ray_tracing['num_samples'],
                los=True,
                reflection=True,
                diffraction=True,
                scattering=False  # Disabled for stability
            )

            if paths is None:
                raise RuntimeError("Path computation failed")

            # Get CIR with explicit normalization
            a, tau = paths.cir(normalize=True)
            
            # Handle numerical stability
            epsilon = 1e-10
            a = tf.cast(a, tf.complex64)
            tau = tf.cast(tau, tf.float32) + epsilon
            
            a = tf.where(tf.math.is_nan(a), tf.zeros_like(a, dtype=tf.complex64), a)
            tau = tf.where(tf.math.is_nan(tau), tf.zeros_like(tau), tau)

            # Calculate frequencies
            frequencies = tf.cast(
                tf.range(self.config.num_subcarriers, dtype=tf.float32) * 
                self.config.subcarrier_spacing,
                tf.float32
            )

            # Generate OFDM channel
            h = cir_to_ofdm_channel(
                frequencies=frequencies,
                a=a,
                tau=tau,
                normalize=True
            )

            # Post-process channel matrix
            h = tf.where(tf.math.is_nan(h), tf.zeros_like(h, dtype=tf.complex64), h)
            h = tf.where(tf.math.is_inf(h), tf.zeros_like(h, dtype=tf.complex64), h)
            h = h + tf.cast(epsilon, tf.complex64)

            # Monitor channel quality
            h = self.monitor_channel_quality(h)

            return {
                'h': h,
                'tau': tau,
                'paths': paths,
                'los_condition': paths.LOS,
                'agv_positions': current_positions
            }

        except Exception as e:
            logger.error(f"Error in generate_channel: {str(e)}")
            raise RuntimeError(f"Error generating channel response: {str(e)}")

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