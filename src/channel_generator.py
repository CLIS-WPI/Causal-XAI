#channel_generator.py#
from utils import ensure_mitsuba_variant
import mitsuba
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
     
    def generate_channel_data(self, config):
        print("\n=== CHANNEL GENERATOR STATE ===")
        print(f"Scene transmitters exist: {hasattr(self.scene, 'transmitters')}")
        print(f"Number of transmitters: {len(self.scene.transmitters)}")
        print(f"Transmitter keys: {list(self.scene.transmitters.keys())}")
        print("==============================\n")
        try:
            import mitsuba
            variant = ensure_mitsuba_variant('cuda_ad_rgb')
            logger.debug(f"Mitsuba variant in use: {variant}")
            if not hasattr(mitsuba, '_variant_name'):
                logger.error("Mitsuba variant not set correctly!")
                raise RuntimeError("Mitsuba variant not set")

            logger.debug("=== Generating channel data ===")
            logger.debug(f"Scene transmitters: {list(self.scene.transmitters.keys())}")
            logger.debug(f"Scene receivers: {list(self.scene.receivers.keys())}")

            agv_positions = self.path_manager.update_positions()
            agv_positions = tf.convert_to_tensor(agv_positions, dtype=config.real_dtype)
            logger.debug(f"AGV positions updated: shape={agv_positions.shape}")
            for i in range(config.num_agvs):
                self.scene.receivers[f'rx_agv_{i}'].position = agv_positions[i]
                logger.debug(f"Receiver rx_agv_{i} position set to: {agv_positions[i]}")

            tx_pos = list(self.scene.transmitters.values())[0].position
            logger.debug(f"Transmitter position: {tx_pos}")

            logger.debug("Starting compute_paths...")
            paths = self.scene.compute_paths(
                max_depth=config.ray_tracing['max_depth'],
                method=config.ray_tracing['method'],
                num_samples=config.ray_tracing['num_samples'],
                los=config.ray_tracing['los'],
                reflection=config.ray_tracing['reflection'],
                diffraction=config.ray_tracing['diffraction'],
                scattering=config.ray_tracing['scattering'],
                scat_keep_prob=config.ray_tracing.get('scat_keep_prob', 0.7),
                edge_diffraction=config.ray_tracing.get('edge_diffraction', True)
            )
            logger.debug("compute_paths completed successfully")

            a, tau = paths.cir()
            a = tf.convert_to_tensor(a, dtype=tf.complex64)
            tau = tf.convert_to_tensor(tau, dtype=tf.float32)
            logger.debug(f"CIR shape: a={a.shape}, tau={tau.shape}")

            reflection_mask = tf.logical_not(tf.cast(paths.LOS, tf.bool))
            a = tf.where(reflection_mask, a * self.inf_params['reflection_coeff'], a)
            los_conditions = tf.cast(paths.LOS, tf.bool)
            if tf.size(los_conditions) == 1:
                los_conditions = tf.tile([los_conditions], [config.num_agvs])
            logger.debug(f"LOS conditions: shape={los_conditions.shape}")

            for i in range(config.num_agvs):
                if los_conditions[i]:
                    a = self._apply_rician_fading(a, self.inf_params['los_k_factor'], i)
                else:
                    a = self._apply_rayleigh_fading(a, self.inf_params['nlos_sigma'], i)

            path_powers = tf.reduce_mean(tf.abs(a)**2, axis=-1)
            direction_vectors = agv_positions - tx_pos
            magnitude = tf.norm(direction_vectors, axis=-1)
            theta = tf.math.acos(direction_vectors[..., 2] / (magnitude + tf.keras.backend.epsilon()))
            phi = tf.math.atan2(direction_vectors[..., 1], direction_vectors[..., 0])
            path_directions = tf.stack([theta, phi], axis=-1)
            logger.debug(f"Path powers: shape={path_powers.shape}")
            logger.debug(f"Path directions: shape={path_directions.shape}")

            frequencies = subcarrier_frequencies(
                num_subcarriers=config.num_subcarriers,
                subcarrier_spacing=config.subcarrier_spacing
            )
            logger.debug(f"Frequencies: shape={frequencies.shape}")

            h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
            logger.debug(f"h_freq initial shape: {h_freq.shape}")
            h_freq = tf.squeeze(h_freq, axis=[0, 3, 5])
            logger.debug(f"h_freq after squeeze: shape={h_freq.shape}")
            h_freq = tf.reduce_mean(h_freq, axis=[1, 2])
            logger.debug(f"h_freq after reducing antennas and paths: shape={h_freq.shape}")
            h_freq = tf.ensure_shape(h_freq, [config.num_agvs, config.num_subcarriers])
            logger.debug(f"h_freq final shape: {h_freq.shape}")

            distances = tf.norm(agv_positions - tx_pos, axis=-1)
            logger.debug(f"Distances computed: shape={distances.shape}")
            # محاسبه path_losses به صورت ساده‌تر با NumPy
            distances_np = distances.numpy()
            carrier_freq_np = float(config.carrier_frequency)
            path_losses_np = 0.7 * (20 * np.log10(distances_np + 1e-6) + 20 * np.log10(carrier_freq_np) - 147.55)
            path_losses = tf.convert_to_tensor(path_losses_np, dtype=tf.float32)
            logger.debug(f"Path losses computed directly: shape={path_losses.shape}, value={path_losses.numpy()}")

            path_losses_linear = tf.pow(10.0, -path_losses / 10.0)
            path_losses_linear = tf.ensure_shape(path_losses_linear, [config.num_agvs])
            h_freq = h_freq * tf.cast(path_losses_linear[:, tf.newaxis], tf.complex64)
            logger.debug(f"Path losses linear: shape={path_losses_linear.shape}")

            snr_metrics = self.calculate_snr(h_freq, config, path_losses)
            los_conditions = tf.cast(paths.LOS, tf.int32)
            if tf.size(los_conditions) == 1:
                los_conditions = tf.tile([los_conditions], [config.num_agvs])
            logger.debug(f"SNR metrics: average_snr={snr_metrics['average_snr']}, snr_db shape={snr_metrics['beam_metrics']['snr_db'].shape}")

            channel_data = {
                'channel_matrices': h_freq,
                'path_delays': tau,
                'los_conditions': los_conditions,
                'agv_positions': agv_positions,
                'path_losses': path_losses,
                'beam_metrics': {
                    'snr_db': snr_metrics['beam_metrics']['snr_db'],
                    'beam_directions': self.beam_manager.get_current_beams()
                },
                'path_data': {
                    'path_powers': path_powers,
                    'path_directions': path_directions
                }
            }
            logger.debug(f"Channel data keys: {channel_data.keys()}")
            return channel_data

        except Exception as e:
            logger.error(f"Error in channel data generation: {str(e)}", exc_info=True)
            return None

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
            position = rx.position
            if len(position.shape) > 1 and position.shape[0] > 1:
                logger.warning(f"Receiver {rx_name} has invalid position shape {position.shape}")
                # Take only the first row if it's a 2D array
                corrected_position = position[0]
                logger.warning(f"Correcting to: {corrected_position}")
                rx.position = corrected_position

    def calculate_snr(self, h_freq, config, path_losses=None):
        try:
            logger.debug("=== Starting SNR calculation ===")
            logger.debug(f"Input h_freq: shape={h_freq.shape}, dtype={h_freq.dtype}, sample={h_freq.numpy()[:2, :5]}")
            print(f"Input h_freq: shape={h_freq.shape}, dtype={h_freq.dtype}, sample={h_freq.numpy()[:2, :5]}")

            h_freq = h_freq * 7000.0
            if not isinstance(path_losses, tf.Tensor):
                logger.debug("Converting path_losses to tensor")
                path_losses = tf.convert_to_tensor(path_losses, dtype=tf.float32)
                print(f"Path losses converted to tensor: shape={path_losses.shape}, value={path_losses.numpy()}")
            logger.debug(f"Input path_losses: shape={path_losses.shape}, value={path_losses.numpy()}")

            tx_power_dbm = config.tx_power
            tx_antenna_gain_db = config.bs_array['antenna_gain_db']
            rx_antenna_gain_db = config.agv_array['antenna_gain_db']
            total_noise_power = config.simulation['noise_power']
            logger.debug(f"Config parameters - TX power: {tx_power_dbm} dBm, TX gain: {tx_antenna_gain_db} dB, RX gain: {rx_antenna_gain_db} dB, Noise power: {total_noise_power:.2e} W")
            print(f"Transmit power dBm: {tx_power_dbm}, TX gain dB: {tx_antenna_gain_db}, RX gain dB: {rx_antenna_gain_db}, Noise power: {total_noise_power:.2e} W")

            tx_power = tf.pow(10.0, (tx_power_dbm - 30) / 10.0)
            tx_gain = tf.pow(10.0, tx_antenna_gain_db / 10.0)
            rx_gain = tf.pow(10.0, rx_antenna_gain_db / 10.0)
            logger.debug(f"Linear conversions - TX power: {float(tx_power):.2e} W, TX gain: {float(tx_gain):.2f}, RX gain: {float(rx_gain):.2f}, Noise power: {float(total_noise_power):.2e} W")
            print(f"Transmit power: {float(tx_power):.2e} W, TX gain: {float(tx_gain):.2f}, RX gain: {float(rx_gain):.2f}, Noise power: {float(total_noise_power):.2e} W")

            logger.debug("Calculating channel power from h_freq")
            channel_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=1)
            logger.debug(f"Channel power calculated: shape={channel_power.shape}, mean={float(tf.reduce_mean(channel_power)):.2e}, value={channel_power.numpy()}")
            print(f"Channel power: shape={channel_power.shape}, mean={float(tf.reduce_mean(channel_power)):.2e}, value={channel_power.numpy()}")

            signal_power = channel_power * tx_power * tx_gain * rx_gain
            logger.debug(f"Signal power (no path loss): shape={signal_power.shape}, mean={float(tf.reduce_mean(signal_power)):.2e}, value={signal_power.numpy()}")
            print(f"Signal power before path loss: shape={signal_power.shape}, mean={float(tf.reduce_mean(signal_power)):.2e}, value={signal_power.numpy()}")

            if path_losses is not None:
                logger.debug("Applying path losses")
                path_loss_linear = tf.pow(10.0, -path_losses / 10.0)
                path_loss_linear = tf.ensure_shape(path_loss_linear, [self.config.num_agvs])
                signal_power = signal_power * tf.cast(path_loss_linear, tf.float32)
                logger.debug(f"Path loss linear: shape={path_loss_linear.shape}, mean={float(tf.reduce_mean(path_loss_linear)):.2e}, value={path_loss_linear.numpy()}")
                logger.debug(f"Signal power (with path loss): shape={signal_power.shape}, mean={float(tf.reduce_mean(signal_power)):.2e}, value={signal_power.numpy()}")
                print(f"Path loss applied: shape={path_loss_linear.shape}, mean={float(tf.reduce_mean(path_loss_linear)):.2e}, value={path_loss_linear.numpy()}")
                print(f"Updated signal power: shape={signal_power.shape}, mean={float(tf.reduce_mean(signal_power)):.2e}, value={signal_power.numpy()}")

            logger.debug("Computing SNR")
            snr_linear = signal_power / total_noise_power
            logger.debug(f"SNR linear: shape={snr_linear.shape}, mean={float(tf.reduce_mean(snr_linear)):.2e}, value={snr_linear.numpy()}")
            print(f"SNR linear: shape={snr_linear.shape}, mean={float(tf.reduce_mean(snr_linear)):.2e}, value={snr_linear.numpy()}")

            snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)  # اصلاح شده
            logger.debug(f"SNR dB calculated: shape={snr_db.shape}, mean={float(tf.reduce_mean(snr_db)):.2f}, value={snr_db.numpy()}")
            print(f"SNR dB: shape={snr_db.shape}, mean={float(tf.reduce_mean(snr_db)):.2f}, value={snr_db.numpy()}")

            snr_db_clipped = tf.clip_by_value(snr_db, -10.0, 40.0)
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
            raise

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

    def _apply_rayleigh_fading(self, channel, sigma, agv_idx):
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
    
    