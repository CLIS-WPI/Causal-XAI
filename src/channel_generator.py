#channel_generator.py
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
            self.path_manager = AGVPathManager(config)
            self.beam_manager = BeamManager(config)
            
        except Exception as e:
            logger.error(f"Channel initialization failed: {str(e)}")
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
                'los_conditions': los_conditions.numpy()
            }
        except Exception as e:
            logger.error(f"Error in beam switching analysis: {str(e)}")
            raise

    def generate_channel_data(self, config):
        """Generate channel data using ray tracing"""
        try:
            logger.debug("=== Generating channel data ===")
            
            # Optimize ray tracing parameters for beam switching scenario
            paths = self.scene.compute_paths(
                max_depth=4,  # Reduced for faster computation
                method="fibonacci",
                num_samples=2000,  # Increased for better accuracy
                los=True,
                reflection=True,
                diffraction=True,
                scattering=True,
                scat_keep_prob=1.0,
                edge_diffraction=True
            )
            
            if paths is None:
                logger.error("Path computation failed - no paths found")
                raise RuntimeError("Path computation failed")

            # Get channel impulse responses
            a, tau = paths.cir()
            
            # Calculate frequencies
            frequencies = subcarrier_frequencies(
                num_subcarriers=config.num_subcarriers,
                subcarrier_spacing=config.subcarrier_spacing
            )
            
            # Convert to OFDM channel
            h_freq = cir_to_ofdm_channel(
                frequencies=frequencies,
                a=tf.cast(a, tf.complex64),
                tau=tf.cast(tau, tf.float32),
                normalize=True
            )
            
            # Enhanced channel data dictionary
            channel_data = {
                'channel_matrices': h_freq,
                'path_delays': tau,
                'los_conditions': paths.LOS,
                'agv_positions': tf.stack([rx.position for rx in self.scene.receivers.values()]),
                'num_paths': tf.size(paths.LOS),
                'reflection_paths': paths.reflection_paths if hasattr(paths, 'reflection_paths') else None,
                'diffraction_paths': paths.diffraction_paths if hasattr(paths, 'diffraction_paths') else None
            }
            
            # Add beam switching analysis
            beam_analysis = self.analyze_beam_switching(channel_data)
            channel_data.update(beam_analysis)
            
            logger.debug("Channel data generation completed successfully")
            return channel_data
            
        except Exception as e:
            logger.error(f"Error generating channel data: {str(e)}")
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
        if los_condition:
            # Rician fading for LOS
            k_factor = self.config.channel_params['rician_k_factor'] 
            return self._apply_rician_fading(channel, k_factor)
        else:
            # Rayleigh fading for NLOS
            sigma = self.config.channel_params['rayleigh_sigma']
            return self._apply_rayleigh_fading(channel, sigma)
    
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