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
    """Smart Factory Channel Generator using Sionna
    
    This class handles the generation and analysis of wireless channels
    in a smart factory environment with two mobile AGVs and RIS elements.
    """
    
    def __init__(self, config, scene=None):
        """Initialize the environment with configuration and scene setup."""
        try:
            # Initialize basic attributes
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

        except Exception as e:
            raise RuntimeError(f"Channel initialization failed: {str(e)}") from e

    def _setup_antenna_arrays(self):
        """Configure antenna arrays for BS and AGVs"""
        try:
            # Base station array
            self.bs_array = PlanarArray(
                num_rows=self.config.bs_array[0],
                num_cols=self.config.bs_array[1],
                vertical_spacing=self.config.bs_array_spacing,
                horizontal_spacing=self.config.bs_array_spacing,
                pattern=self.config.bs_array_pattern,
                polarization=self.config.bs_polarization,
                dtype=tf.complex64
            )
            
            # AGV array
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
            [12.0, 5.0, self.config.agv_height],   # AGV1
            [8.0, 15.0, self.config.agv_height]    # AGV2
        ], dtype=tf.float32)

    def _update_agv_positions(self, time_step):
        """Update AGV positions"""
        current_positions = self.agv_positions.numpy()
        new_positions = []
        
        # Simple linear movement for demonstration
        for i in range(self.config.num_agvs):
            current_pos = current_positions[i]
            new_pos = current_pos + np.array([0.1, 0.1, 0]) * time_step
            
            # Keep within room bounds
            new_pos[0] = np.clip(new_pos[0], 0, self.config.room_dim[0])
            new_pos[1] = np.clip(new_pos[1], 0, self.config.room_dim[1])
            new_pos[2] = self.config.agv_height
            
            new_positions.append(new_pos)
            self.positions_history[i].append(new_pos.copy())
        
        return tf.constant(new_positions, dtype=tf.float32)

    def generate_channel(self):
        """Generate channel matrices with RIS modeling"""
        try:
            # Update AGV positions
            current_positions = self._update_agv_positions(self.config.simulation['time_step'])
            current_positions = tf.expand_dims(current_positions, axis=0)

            # Update AGV positions in scene
            for i in range(self.config.num_agvs):
                agv = self.scene.get(f"agv_{i}")
                if agv is not None:
                    agv.position = current_positions[0, i].numpy()

            # Generate paths using ray tracing
            paths = self.scene.compute_paths(
                max_depth=self.config.ray_tracing['max_depth'],
                method=self.config.ray_tracing['method'],
                num_samples=self.config.ray_tracing['num_samples'],
                los=self.config.ray_tracing['los'],
                reflection=self.config.ray_tracing['reflection'],
                diffraction=self.config.ray_tracing['diffraction'],
                scattering=self.config.ray_tracing['scattering']
            )

            # Get channel impulse response with explicit type casting
            a, tau = paths.cir()
            
            # Handle potential NaN values in CIR components
            a = tf.cast(a, tf.complex64)
            tau = tf.cast(tau, tf.float32)
            
            # Replace NaN values with zeros
            a = tf.where(tf.math.is_nan(a), tf.zeros_like(a, dtype=tf.complex64), a)
            tau = tf.where(tf.math.is_nan(tau), tf.zeros_like(tau), tau)

            # Calculate frequencies with proper type casting
            frequencies = tf.cast(
                tf.range(self.config.num_subcarriers, dtype=tf.float32) * 
                self.config.subcarrier_spacing,
                tf.float32
            )

            # Add small epsilon to avoid numerical instability
            epsilon = 1e-10
            tau = tau + epsilon

            # Generate OFDM channel with explicit normalization and type handling
            h = sionna.channel.utils.cir_to_ofdm_channel(
                frequencies=frequencies,
                a=a,
                tau=tau,
                normalize=True  # Enable normalization for numerical stability
            )

            # Post-process channel matrix
            h = tf.where(tf.math.is_nan(h), tf.zeros_like(h, dtype=tf.complex64), h)
            h = tf.where(tf.math.is_inf(h), tf.zeros_like(h, dtype=tf.complex64), h)

            # Add small epsilon to avoid zero values
            h = h + tf.cast(epsilon, tf.complex64)

            # Verify channel matrix
            if tf.reduce_any(tf.math.is_nan(h)):
                logger.warning("NaN values detected in final channel matrix")
                h = tf.where(tf.math.is_nan(h), tf.zeros_like(h, dtype=tf.complex64), h)

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
        """Save CSI dataset to HDF5 file"""
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
            
            for _ in range(num_samples):
                sample = self.generate_channel()
                channel_data.append(sample['h'].numpy())
                path_delays.append(sample['tau'].numpy())
                los_conditions.append(np.array(sample['los_condition'], dtype=np.int32))
                agv_positions.append(sample['agv_positions'].numpy())
            
            # Save datasets
            csi_group.create_dataset('channel_matrices', data=np.array(channel_data))
            csi_group.create_dataset('path_delays', data=np.array(path_delays))
            csi_group.create_dataset('los_conditions', data=np.array(los_conditions))
            csi_group.create_dataset('agv_positions', data=np.array(agv_positions))
            
            # Save configuration
            for key, value in vars(self.config).items():
                if isinstance(value, (int, float, str, list)):
                    config_group.attrs[key] = value