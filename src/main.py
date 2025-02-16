#main.py
import tensorflow as tf
from config import SmartFactoryConfig
from scene_setup import setup_scene, verify_los_paths
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
import sionna
from sionna.rt import Scene, Camera
import logging
logger = logging.getLogger(__name__)
import h5py
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.channel.utils import subcarrier_frequencies
from sionna_ply_generator import SionnaPLYGenerator
from sionna.constants import SPEED_OF_LIGHT
from scene_setup import verify_geometry
from beam_manager import BeamManager
from channel_generator import SmartFactoryChannel

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_factory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable XLA compatibility
sionna.config.xla_compat = True

def ensure_result_dir():
    """Create result directory if it doesn't exist"""
    result_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def validate_config(config):
    """Validate configuration parameters"""
    required_attrs = [
        'carrier_frequency', 'num_time_steps',
        'num_agvs', 'room_dim', 'bs_array'
    ]
    
    missing_attrs = [attr for attr in required_attrs 
                    if not hasattr(config, attr)]
    
    if missing_attrs:
        raise ValueError(f"Missing required configuration attributes: {missing_attrs}")
    
    if config.carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be positive")

def calculate_snr(h_freq, config):
    """
    Calculate realistic SNR for mmWave systems accounting for thermal noise and hardware impairments
    
    Parameters:
        h_freq: Channel frequency response
        config: Configuration object containing system parameters
    """
    # Physical constants
    k_boltzmann = 1.380649e-23  # Boltzmann constant [J/K]
    temperature = 290           # Room temperature [K]
    
    # System parameters for 28GHz
    bandwidth = config.subcarrier_spacing * config.num_subcarriers  # Total bandwidth
    noise_figure_db = 10        # Typical mmWave receiver noise figure [dB]
    implementation_loss_db = 3  # Implementation loss [dB]
    
    # Calculate thermal noise power
    thermal_noise = k_boltzmann * temperature * bandwidth
    
    # Convert noise figure from dB to linear
    noise_figure_linear = 10 ** (noise_figure_db / 10)
    
    # Convert implementation loss from dB to linear
    implementation_loss_linear = 10 ** (implementation_loss_db / 10)
    
    # Total noise power including noise figure and implementation loss
    total_noise_power = thermal_noise * noise_figure_linear * implementation_loss_linear
    
    # Calculate average signal power across all antennas and subcarriers
    # Assuming h_freq shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_streams, num_subcarriers]
    signal_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=[-1])  # Average over subcarriers
    
    # Apply path loss and shadowing if provided in channel model
    if hasattr(config, 'path_loss_db'):
        path_loss_linear = 10 ** (-config.path_loss_db / 10)
        signal_power = signal_power * path_loss_linear
    
    # Calculate SNR
    snr_linear = signal_power / total_noise_power
    
    # Convert to dB with realistic bounds for mmWave systems
    snr_db = 10 * tf.math.log(snr_linear) / tf.math.log(10.0)
    
    # Apply realistic bounds for indoor mmWave systems
    max_snr_db = 40.0  # Maximum realistic SNR for indoor mmWave
    min_snr_db = -20.0 # Minimum detectable SNR
    
    # Clip SNR to realistic range
    snr_db_clipped = tf.clip_by_value(snr_db, min_snr_db, max_snr_db)
    
    # Calculate statistics for logging
    avg_snr = tf.reduce_mean(snr_db_clipped)
    max_snr = tf.reduce_max(snr_db_clipped)
    min_snr = tf.reduce_min(snr_db_clipped)
    
    logger.info(f"SNR Statistics:")
    logger.info(f"- Average SNR: {avg_snr:.2f} dB")
    logger.info(f"- Maximum SNR: {max_snr:.2f} dB")
    logger.info(f"- Minimum SNR: {min_snr:.2f} dB")
    
    return snr_db_clipped

# Add necessary configuration parameters to SmartFactoryConfig
def add_snr_config(config):
    """Add SNR-related parameters to configuration"""
    config.snr_parameters = {
        'noise_figure_db': 10,        # Typical value for mmWave receivers
        'implementation_loss_db': 3,  # Typical system implementation loss
        'thermal_noise_temp': 290,    # Room temperature in Kelvin
        'min_snr_db': -20,            # Minimum detectable SNR
        'max_snr_db': 40,             # Maximum expected SNR
    }
    
    # Add typical path loss for indoor factory at 28GHz
    config.path_loss_db = 80  # Typical value for indoor factory environment


def generate_channel_data(scene, config, beam_manager=None):
    """Generate enhanced channel data using ray tracing"""
    try:
        logger.info("Starting channel data generation...")
        
        # this is new section for beam management
        if beam_manager is not None:
            logger.info("Using beam manager for adaptive beamforming...")
            current_beams = beam_manager.get_current_beams()
            if current_beams is not None:
                for tx in scene.transmitters.values():
                    tx.antenna.apply_beam_weights(current_beams)

        # Initialize expected_los with a default value
        expected_los = False
        
        # Get visibility results
        if hasattr(scene, 'visibility_results'):
            logger.info("Using pre-computed visibility results")
            expected_los = any(result['los_available'] for result in scene.visibility_results.values())
            if not expected_los:
                logger.warning("No LOS paths expected based on geometry!")
                
        # Compute paths with visibility awareness
        paths = scene.compute_paths(
            max_depth=config.ray_tracing['max_depth'],
            method=config.ray_tracing['method'],
            num_samples=config.ray_tracing['num_samples'],
            los=True,  # Always try for LOS
            reflection=True,
            diffraction=True,
            scattering=True,
            scat_keep_prob=config.ray_tracing['scat_keep_prob'],
            edge_diffraction=True
        )
        
        # Validate paths against visibility expectations
        if paths is not None:
            los_paths = tf.reduce_sum(tf.cast(paths.LOS, tf.int32))
            if los_paths == 0 and expected_los:
                logger.warning("No LOS paths found despite geometric visibility!")
                logger.warning("Increasing number of samples might help detect paths")

        epsilon = tf.constant(1e-8, dtype=tf.float32)
        logger.debug("=== Generating channel data ===")
        logger.debug("Ray tracing configuration:")
        logger.debug(f"- Method: {config.ray_tracing['method']}")
        logger.debug(f"- Max depth: {config.ray_tracing['max_depth']}")
        logger.debug(f"- Num samples: {config.ray_tracing['num_samples']}")
        logger.debug(f"- LOS enabled: {config.ray_tracing['los']}")

        # Compute paths using ray tracing with correct parameters
        paths = scene.compute_paths(
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
            raise ValueError("Path computation failed - no paths found")
            
        # Path analysis
        num_paths = tf.size(paths.LOS)
        los_count = tf.reduce_sum(tf.cast(paths.LOS, tf.int32))
        logger.debug("=== Path Computation Results ===")
        logger.debug(f"- Paths object type: {type(paths)}")
        logger.debug(f"- Number of paths: {num_paths}")
        logger.debug(f"- LOS paths: {los_count}")
        logger.debug(f"- NLOS paths: {num_paths - los_count}")
        
        # Get channel impulse responses
        logger.debug("Computing channel impulse responses...")
        a, tau = paths.cir()
        logger.debug(f"CIR shapes - a: {a.shape}, tau: {tau.shape}")

        # Add this section here to replace NaN values with zeros in CIR coefficients
        a = tf.where(
            tf.logical_or(
                tf.math.is_nan(tf.math.real(a)),
                tf.math.is_nan(tf.math.imag(a))
            ),
            tf.zeros_like(a),
            a
        )

        # After computing paths but before CIR calculation
        if beam_manager is not None:
            # Update beam manager with current channel state
            beam_manager.update_channel_state({
                'paths': paths,
                'los_available': expected_los,
                'scene_state': {
                    'agv_positions': [rx.position for rx in scene.receivers.values()],
                    'obstacle_positions': [obj.center for obj in scene.objects.values() 
                                        if 'shelf' in obj.name]
                }
            })

        # Then continue with the existing code
        # Check CIR values - handle complex numbers correctly
        a_abs = tf.abs(a)
        logger.debug("Checking CIR values...")
        
        # Check CIR values - handle complex numbers correctly
        a_abs = tf.abs(a)
        logger.debug("Checking CIR values...")
        if tf.reduce_any(tf.math.is_nan(tf.math.real(a))) or tf.reduce_any(tf.math.is_nan(tf.math.imag(a))):
            logger.warning("NaN values detected in CIR coefficients")
        if tf.reduce_any(a_abs < epsilon):
            logger.warning(f"Values below epsilon ({epsilon}) detected in CIR coefficients")
        logger.debug(f"CIR coefficient range: min={tf.reduce_min(a_abs)}, max={tf.reduce_max(a_abs)}")
        
        # Calculate frequencies for subcarriers
        frequencies = subcarrier_frequencies(
            num_subcarriers=config.num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing
        )
        
        # Convert to OFDM channel
        logger.debug("Converting to OFDM channel...")
        h_freq = cir_to_ofdm_channel(
            frequencies=frequencies,
            a=tf.cast(a, tf.complex64),
            tau=tf.cast(tau, tf.float32),
            normalize=True
        )
        
        # Check OFDM channel values - handle complex numbers correctly
        h_abs = tf.abs(h_freq)
        h_real = tf.math.real(h_freq)
        h_imag = tf.math.imag(h_freq)
        
        logger.debug(f"OFDM channel shape: {h_freq.shape}")
        logger.debug("Checking OFDM channel values...")
        
        # Check for invalid values in real and imaginary parts separately
        invalid_real = tf.reduce_any(tf.logical_or(
            tf.math.is_nan(h_real),
            tf.math.is_inf(h_real)
        ))
        invalid_imag = tf.reduce_any(tf.logical_or(
            tf.math.is_nan(h_imag),
            tf.math.is_inf(h_imag)
        ))
        
        if invalid_real or invalid_imag:
            logger.warning("Invalid values detected in OFDM channel")
            
        # Replace invalid values with zeros
        h_freq = tf.where(
            tf.logical_or(
                tf.logical_or(tf.math.is_nan(h_real), tf.math.is_inf(h_real)),
                tf.logical_or(tf.math.is_nan(h_imag), tf.math.is_inf(h_imag))
            ),
            tf.zeros_like(h_freq),
            h_freq
        )
        
        # Calculate power
        power = tf.reduce_mean(tf.abs(h_freq)**2, axis=-1, keepdims=True)
        power = tf.maximum(power + epsilon, epsilon)
        
        # Normalize channel
        h_freq_norm = tf.cast(tf.sqrt(power), tf.complex64)
        h_freq = h_freq / h_freq_norm

        # After normalization
        logger.debug("Channel statistics after normalization:")
        logger.debug(f"- Mean magnitude: {tf.reduce_mean(tf.abs(h_freq))}")
        logger.debug(f"- Max magnitude: {tf.reduce_max(tf.abs(h_freq))}")
        logger.debug(f"- Min magnitude: {tf.reduce_min(tf.abs(h_freq))}")


        # Add SNR configuration and calculation
        add_snr_config(config)
        snr = calculate_snr(h_freq, config)
        logger.debug(f"Calculated realistic SNR values for mmWave system")

        # Calculate Doppler shifts
        agv_positions = tf.stack([rx.position for rx in scene.receivers.values()])
        bs_position = tf.constant(config.bs_position, dtype=tf.float32)
        
        # Calculate relative velocities and directions
        relative_positions = agv_positions - bs_position
        directions = tf.nn.l2_normalize(relative_positions, axis=-1)
        
        # Assuming constant AGV speed from config
        velocities = tf.ones_like(agv_positions) * config.agv_speed
        relative_velocities = tf.reduce_sum(velocities * directions, axis=-1)
        
        # Calculate Doppler shifts: f_d = (v/c) * f_c
        doppler_shifts = (relative_velocities / SPEED_OF_LIGHT) * config.carrier_frequency
        
        # Calculate LOS/NLOS statistics with safe division and type conversion
        los_conditions = paths.LOS
        if not isinstance(los_conditions, tf.Tensor):
            los_conditions = tf.cast(los_conditions, tf.float32)  # Ensure float type

        total_paths = tf.cast(tf.size(los_conditions), tf.float32)
        los_paths = tf.reduce_sum(tf.cast(los_conditions, tf.int32))

        # Add debug prints HERE, right before the subtraction
        print(f"Debug - total_paths type: {total_paths.dtype}")
        print(f"Debug - total_paths shape: {total_paths.shape}")
        print(f"Debug - total_paths value: {total_paths}")

        print(f"Debug - los_paths type: {los_paths.dtype}")
        print(f"Debug - los_paths shape: {los_paths.shape}")
        print(f"Debug - los_paths value: {los_paths}")

        # Fix the type mismatch by casting los_paths to float32
        los_paths = tf.cast(los_paths, tf.float32)

        nlos_paths = total_paths - los_paths  # Now this line should work

        print("=== Path Calculation Results ===")
        print(f"Debug - nlos_paths: {nlos_paths}")

        # Handle case where no paths are found or NaN values exist
        if total_paths == 0 or tf.math.is_nan(total_paths) or tf.math.is_nan(los_paths):
            if tf.math.is_nan(total_paths) or tf.math.is_nan(los_paths):
                logger.warning("NaN values detected in path calculations")
            else:
                logger.warning("No paths found in channel computation")
            
            los_statistics = {
                'los_ratio': 0.0,
                'nlos_ratio': 0.0,
                'total_paths': 0,
                'los_paths': 0,
                'nlos_paths': 0
            }
        else:
            # Safe division with NaN checking
            try:
                los_ratio = float(los_paths.numpy()) / float(total_paths.numpy())
                nlos_ratio = float(nlos_paths.numpy()) / float(total_paths.numpy())
                
                # Check for NaN in computed ratios
                if tf.math.is_nan(los_ratio) or tf.math.is_nan(nlos_ratio):
                    logger.warning("NaN values detected in LOS/NLOS ratio calculations")
                    los_ratio = 0.0
                    nlos_ratio = 0.0
                    
                los_statistics = {
                    'los_ratio': los_ratio,
                    'nlos_ratio': nlos_ratio,
                    'total_paths': int(total_paths.numpy()),
                    'los_paths': int(los_paths.numpy()),
                    'nlos_paths': int(nlos_paths.numpy())
                }
            except Exception as e:
                logger.error(f"Error in LOS/NLOS statistics calculation: {str(e)}")
                los_statistics = {
                    'los_ratio': 0.0,
                    'nlos_ratio': 0.0,
                    'total_paths': 0,
                    'los_paths': 0,
                    'nlos_paths': 0
                }

        # Log statistics for debugging
        logger.debug(f"LOS Statistics: {los_statistics}")
        
        # After calculating signal power (around line 250-260 in your code)
        signal_power = tf.reduce_mean(tf.abs(h_freq)**2, axis=-1)
        # Add debug messages here
        logger.debug(f"Signal power statistics:")
        logger.debug(f"- Mean: {tf.reduce_mean(signal_power)}")
        logger.debug(f"- Max: {tf.reduce_max(signal_power)}")
        logger.debug(f"- Min: {tf.reduce_min(signal_power)}")
        logger.debug(f"- Contains inf: {tf.reduce_any(tf.math.is_inf(signal_power))}")
        logger.debug(f"- Contains nan: {tf.reduce_any(tf.math.is_nan(signal_power))}")

        # Calculate average SNR from the previously computed snr value
        average_snr = tf.reduce_mean(snr)

        # Use the 'snr' value that was already calculated
        logger.debug(f"SNR statistics (from calculated values):")
        logger.debug(f"- Mean SNR: {tf.reduce_mean(snr):.2f} dB")
        logger.debug(f"- Max SNR: {tf.reduce_max(snr):.2f} dB")
        logger.debug(f"- Min SNR: {tf.reduce_min(snr):.2f} dB")

        # After calculating path loss (around line 270-280)
        a_abs = tf.abs(a)  # Get magnitude of complex values
        a_abs = tf.cast(a_abs, tf.float32)  # Ensure float32 type
        path_loss = tf.where(
            a_abs > 0,
            20.0 * tf.math.log(a_abs) / tf.math.log(10.0),
            tf.zeros_like(a_abs)
        )

        # Add path loss debug messages here
        logger.debug(f"Path loss statistics:")
        logger.debug(f"- Mean path loss: {tf.reduce_mean(path_loss)}")
        logger.debug(f"- Max path loss: {tf.reduce_max(path_loss)}")
        logger.debug(f"- Min path loss: {tf.reduce_min(path_loss)}")
        
        beam_metrics = {
            'snr_db': snr.numpy(),  # Use snr instead of snr_db
            'avg_power': float(tf.reduce_mean(signal_power)),
            'max_power': float(tf.reduce_max(signal_power)),
            'min_power': float(tf.reduce_min(signal_power)),
            'path_loss': path_loss
        }

        # Before creating the channel_data dictionary, convert los_conditions to a tensor if it isn't already
        los_conditions = tf.convert_to_tensor(los_conditions)

        # Create enhanced channel data dictionary
        channel_data = {
            'channel_matrices': h_freq,
            'path_delays': tau,
            'los_conditions': los_conditions,
            'agv_positions': agv_positions,
            'doppler_shifts': doppler_shifts,
            'los_statistics': los_statistics,
            'beam_metrics': beam_metrics,
            'average_snr': average_snr,
            'temporal_data': {
                'timestamp': tf.timestamp(),
                'agv_velocities': velocities.numpy(),
                'path_conditions': los_conditions.numpy(),
            'beam_adaptation': {
                'current_beam_weights': beam_manager.get_current_beams() if beam_manager is not None else None,
                'beam_optimization_history': beam_manager.get_optimization_history() if beam_manager is not None else None,
                'adaptation_performance': {
                    'snr_improvement': beam_manager.get_snr_improvement() if beam_manager is not None else 0.0,
                    'convergence_time': beam_manager.get_convergence_time() if beam_manager is not None else 0.0
                }
            }
        }

                
        }
        
        # Log key metrics only if paths exist
        if total_paths > 0:
            logger.info(f"Channel Generation Metrics:")
            logger.info(f"LOS Ratio: {los_statistics['los_ratio']:.2f}")
            logger.info(f"Average SNR: {tf.reduce_mean(snr):.2f} dB")  # Changed from snr_db to snr
            logger.info(f"Maximum Doppler Shift: {tf.reduce_max(tf.abs(doppler_shifts)):.2f} Hz")

        if beam_manager is not None:
            channel_data['beam_data'] = {
                'current_beams': beam_manager.get_current_beams(),
                'beam_history': beam_manager.get_beam_history(),
                'adaptation_metrics': beam_manager.get_adaptation_metrics()
            }
        print("Enhanced channel data generation completed")
        
        return channel_data
        
    except Exception as e:
        logger.error(f"Error generating channel data: {str(e)}")
        raise

def save_channel_data(channel_data, filepath):
    """Save channel data to H5 file with enhanced organization and error handling"""
    try:
        with h5py.File(filepath, 'w') as f:
            # Create main groups for better organization
            csi_group = f.create_group('csi_data')
            mobility_group = f.create_group('mobility_data')
            beam_group = f.create_group('beam_data')
            temporal_group = f.create_group('temporal_data')
            
            # Save CSI data
            if 'channel_matrices' in channel_data:
                csi_group.create_dataset('channel_matrices', 
                    data=tf.cast(channel_data['channel_matrices'], tf.complex64).numpy())
            if 'path_delays' in channel_data:
                csi_group.create_dataset('path_delays', 
                    data=tf.cast(channel_data['path_delays'], tf.float32).numpy())
                
            # Save mobility data
            if 'agv_positions' in channel_data:
                mobility_group.create_dataset('agv_positions', 
                    data=tf.cast(channel_data['agv_positions'], tf.float32).numpy())
            if 'doppler_shifts' in channel_data:
                mobility_group.create_dataset('doppler_shifts', 
                    data=tf.cast(channel_data['doppler_shifts'], tf.float32).numpy())
                
            # Save beam data
            if 'los_conditions' in channel_data:
                beam_group.create_dataset('los_conditions', 
                    data=tf.cast(channel_data['los_conditions'], tf.int32).numpy())
            if 'beam_metrics' in channel_data:
                for metric_name, metric_value in channel_data['beam_metrics'].items():
                    beam_group.create_dataset(f'beam_metrics/{metric_name}', 
                        data=np.array(metric_value))
                        
            # Save temporal data if available
            if 'temporal_data' in channel_data:
                for key, value in channel_data['temporal_data'].items():
                    if isinstance(value, (np.ndarray, tf.Tensor)):
                        temporal_group.create_dataset(key, data=np.array(value))
                    else:
                        temporal_group.attrs[key] = value
                        
            # Add metadata
            f.attrs['creation_time'] = str(datetime.now())
            f.attrs['num_receivers'] = len(channel_data.get('agv_positions', []))
            if 'channel_matrices' in channel_data:
                f.attrs['matrix_shape'] = str(channel_data['channel_matrices'].shape)
                
            print(f"Channel data successfully saved to: {filepath}")
            
    except Exception as e:
        logger.error(f"Error saving channel data to {filepath}: {str(e)}")
        raise

def verify_camera_renders(scene, config, result_dir):
    """Verify that all camera renders were successful"""
    for cam_name, cam_params in config.cameras.items():
        render_path = os.path.join(result_dir, cam_params['filename'])
        if os.path.exists(render_path):
            file_size = os.path.getsize(render_path)
            logger.info(f"Camera {cam_name} render saved: {file_size} bytes")
        else:
            logger.warning(f"Failed to generate render for {cam_name} camera")
def main():
    """Main execution function"""
    try:
        # Set up logging
        logger.info("Starting smart factory channel simulation...")
        
        # Create results directory
        result_dir = ensure_result_dir()
        logger.debug(f"Results directory: {result_dir}")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Initialize configuration
        config = SmartFactoryConfig()
        validate_config(config)
        logger.debug("Configuration initialized and validated")
        
        # Ensure the meshes directory exists
        os.makedirs(config.ply_config['output_dir'], exist_ok=True)
        
        # Update PLY output directory in config
        config.ply_config['output_dir'] = os.path.join(os.path.dirname(__file__), 'meshes')
        
        # Generate PLY files using config
        logger.info("Generating PLY files...")
        SionnaPLYGenerator.generate_factory_geometries(
            config=config,
            output_dir=config.ply_config['output_dir']
        )
        logger.info("PLY files generated successfully")
        
        # Setup scene
        logger.info("Setting up scene...")
        scene = setup_scene(config)
        if not scene:
            raise ValueError("Scene setup failed")
        logger.debug("Scene setup completed")

        try:
            logger.info("Validating scene geometry...")
            if not SionnaPLYGenerator.validate_scene_geometry(scene):
                logger.warning("Scene geometry validation failed but continuing execution")
        except Exception as e:
            logger.error(f"Scene validation error: {str(e)}")

        # Add verification
        verify_geometry(scene)

        # Configure cameras and render settings
        logger.info("Configuring scene cameras and render settings...")
        scene.render_config = config.render_config

        # Camera setup and rendering (existing camera code remains the same)
        for cam_name, cam_params in config.cameras.items():
            try:
                logger.info(f"Setting up {cam_name} camera...")
                camera = Camera(
                    name=cam_name,
                    position=cam_params['position']
                )
                camera.look_at(cam_params['look_at'])
                camera.up = cam_params['up']
                scene.add(camera)
                
                logger.info(f"Rendering from {cam_name} camera...")
                render_filename = os.path.join(result_dir, cam_params['filename'])
                scene.render_to_file(camera=cam_name, filename=render_filename)
                logger.info(f"Successfully rendered {render_filename}")
                
            except Exception as e:
                logger.error(f"Error with {cam_name} camera: {str(e)}")
                continue

        # Verify scene setup
        logger.info(f"Registered cameras: {list(scene.cameras.keys())}")
        verify_camera_renders(scene, config, result_dir)
        verify_los_paths(scene)

        # Set scene frequency and initialize beam manager
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        logger.debug(f"Scene frequency set to {config.carrier_frequency/1e9:.2f} GHz")
        
        # Initialize beam manager
        logger.info("Initializing beam manager...")
        beam_manager = BeamManager(config)

        # Get AGV and obstacle positions
        agv_positions = [receiver.position for receiver in scene.receivers.values()]
        obstacle_positions = [obj.position for obj in scene.objects.values() if 'shelf' in obj.name]

        # Simulation loop for adaptive beamforming
        logger.info("Starting adaptive beamforming simulation...")
        channel_data_history = []
        num_iterations = config.simulation.get('num_iterations', 10)

        for iteration in range(num_iterations):
            logger.debug(f"Iteration {iteration+1}/{num_iterations}")

            # Generate channel data
            channel_generator = SmartFactoryChannel(config, scene)
            channel_data = channel_generator.generate_channel_data(scene, config)

            # Detect blockages
            los_blocked = beam_manager.detect_blockage(
                channel_data=channel_data,
                agv_positions=agv_positions,
                obstacle_positions=obstacle_positions
            )

            # Optimize beam direction
            beam_manager.optimize_beam_direction(
                channel_data=channel_data,
                agv_positions=agv_positions,
                obstacle_positions=obstacle_positions
            )

            # Get optimal beams
            optimal_beams = beam_manager.get_current_beams()

            # Apply optimal beams
            for beam_idx, beam in enumerate(optimal_beams):
                scene.transmitters['bs'].array.steering_angle = beam

            # Generate channel data
            current_channel = generate_channel_data(scene, config, beam_manager)
            channel_data_history.append(current_channel)

            # Update causal analysis data
            for beam_idx, beam in enumerate(optimal_beams):
                beam_manager.update_causal_data(
                    beam,
                    obstacle_positions,
                    channel_metrics={
                        'snr': current_channel['average_snr'],
                        'throughput': current_channel.get('throughput', 0),
                        'los_blocked': los_blocked[beam_idx]
                    },
                    agv_state={
                        'speed': config.agv_speed,
                        'distance': tf.norm(agv_positions[beam_idx] - 
                                        scene.transmitters['bs'].position)
                    }
                )

            # Apply optimal beams
            optimal_beams = beam_manager.get_current_beams()
            for beam_idx, beam in enumerate(optimal_beams):
                scene.transmitters['bs'].array.steering_angle = beam

            # Generate channel data
            current_channel = generate_channel_data(scene, config, beam_manager)
            channel_data_history.append(current_channel)

            # Update causal analysis data
            for agv_idx, beam in enumerate(optimal_beams):
                beam_manager.update_causal_data(
                    beam,  # Pass beam as first positional argument
                    obstacle_positions,  # Pass obstacle_positions as second positional argument
                    channel_metrics={
                        'snr': current_channel['average_snr'],
                        'throughput': current_channel.get('throughput', 0),
                        'los_blocked': los_blocked[agv_idx]
                    },
                    agv_state={
                        'speed': config.agv_speed,
                        'distance': tf.norm(agv_positions[agv_idx] - 
                                        scene.transmitters['bs'].position)
                    }
                )

        # Perform causal analysis
        logger.info("Performing causal analysis...")
        causal_effect = beam_manager.perform_causal_analysis()
        
        # Aggregate final channel data
        final_channel_data = channel_data_history[-1]
        final_channel_data['causal_analysis'] = {
            'effect_size': causal_effect.value if causal_effect else None,
            'confidence_intervals': causal_effect.confidence_intervals if causal_effect else None,
            'beam_history': beam_manager.beam_history
        }

        # Save final channel data
        h5_filepath = os.path.join(result_dir, 'channel_data.h5')
        logger.info(f"Saving channel data to: {h5_filepath}")
        save_channel_data(final_channel_data, h5_filepath)
        
        # Print statistics
        logger.info("\nChannel Data Statistics:")
        stats = {
            'Channel matrices shape': final_channel_data['channel_matrices'].shape,
            'Path delays shape': final_channel_data['path_delays'].shape,
            'Number of receivers': len(final_channel_data['agv_positions']),
            'LOS ratio': final_channel_data['los_statistics']['los_ratio'],
            'Average SNR (dB)': final_channel_data['average_snr'],
            'Max Doppler shift (Hz)': tf.reduce_max(tf.abs(final_channel_data['doppler_shifts'])),
            'Causal effect size': final_channel_data['causal_analysis']['effect_size']
        }
        
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Error in execution: {str(e)}", exc_info=True)
        raise
        
    finally:
        logger.info("Execution completed")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('smart_factory.log'),
            logging.StreamHandler()
        ]
    )
    main()