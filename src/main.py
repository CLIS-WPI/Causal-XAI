#
from config import SmartFactoryConfig
from scene_setup import setup_scene
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
import sionna
from sionna.rt import Scene
import logging
logger = logging.getLogger(__name__)
import h5py
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.channel.utils import subcarrier_frequencies
from sionna_ply_generator import SionnaPLYGenerator
from sionna.constants import SPEED_OF_LIGHT
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
        'num_agvs', 'room_dim', 'bs_array', 'ris_elements'
    ]
    
    missing_attrs = [attr for attr in required_attrs 
                    if not hasattr(config, attr)]
    
    if missing_attrs:
        raise ValueError(f"Missing required configuration attributes: {missing_attrs}")
    
    if config.carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be positive")

def calculate_snr(h_freq, noise_power=1.0):
    # Calculate power across all dimensions
    channel_power = tf.reduce_mean(tf.abs(h_freq)**2)
    
    # Ensure we don't divide by zero and handle very small values
    noise_power = tf.maximum(noise_power, 1e-10)
    channel_power = tf.maximum(channel_power, 1e-10)
    
    # Calculate SNR in dB with safety checks
    snr = tf.where(
        channel_power > 0,
        10.0 * tf.math.log(channel_power / noise_power) / tf.math.log(10.0),
        -100.0  # Return a very low but finite value instead of -inf
    )
    
    return tf.where(tf.math.is_finite(snr), snr, -100.0)


def generate_channel_data(scene, config):
    """Generate enhanced channel data using ray tracing"""
    try:
        print("Generating channel data...")
        
        # Compute paths using ray tracing with correct parameters
        paths = scene.compute_paths(
            max_depth=config.ray_tracing['max_depth'],
            method=config.ray_tracing['method'],
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
            logger.error("Path computation failed - no paths found")
            raise ValueError("Path computation failed")
        
        logger.debug("=== Path Computation Results ===")
        logger.debug(f"- Paths object type: {type(paths)}")
        # Get number of paths from paths.LOS tensor size
        num_paths = tf.size(paths.LOS)
        logger.debug(f"- Number of paths: {num_paths}")
        logger.debug(f"- LOS paths: {tf.reduce_sum(tf.cast(paths.LOS, tf.int32))}")
        logger.debug(f"- NLOS paths: {num_paths - tf.reduce_sum(tf.cast(paths.LOS, tf.int32))}")
                
        # Get channel impulse responses 
        logger.debug("Computing channel impulse responses...")
        a, tau = paths.cir()
        logger.debug(f"CIR shapes - a: {a.shape}, tau: {tau.shape}")
        
        # Calculate frequencies for the subcarriers
        frequencies = subcarrier_frequencies(
            num_subcarriers=config.num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing
        )
        
        # Convert to OFDM channel with proper type casting
        logger.debug("Converting to OFDM channel...")
        h_freq = cir_to_ofdm_channel(
            frequencies=frequencies,
            a=tf.cast(a, tf.complex64),
            tau=tf.cast(tau, tf.float32),
            normalize=True
        )
        logger.debug(f"OFDM channel shape: {h_freq.shape}")
        
        # Add normalization and SNR calculation here
        # Check real and imaginary parts separately since is_finite doesn't support complex
        h_freq = tf.where(
            tf.logical_and(
                tf.math.is_finite(tf.math.real(h_freq)),
                tf.math.is_finite(tf.math.imag(h_freq))
            ),
            h_freq,
            tf.zeros_like(h_freq)
        )
        
        # Right before normalization
        logger.debug("Channel statistics before normalization:")
        logger.debug(f"- Mean magnitude: {tf.reduce_mean(tf.abs(h_freq))}")
        logger.debug(f"- Max magnitude: {tf.reduce_max(tf.abs(h_freq))}")
        logger.debug(f"- Min magnitude: {tf.reduce_min(tf.abs(h_freq))}")

        # Calculate power (this will be real-valued)
        power = tf.reduce_mean(tf.abs(h_freq)**2, axis=-1, keepdims=True)
        power = tf.cast(power, tf.float32)  # Ensure float type

        # Add epsilon to power (both are now float)
        epsilon = 1e-10
        power = power + epsilon

        # Take square root (still float)
        h_freq_norm = tf.sqrt(power)

        # Cast to complex for division
        h_freq_norm = tf.cast(h_freq_norm, tf.complex64)

        # Normalize
        h_freq = h_freq / h_freq_norm

        # After normalization
        logger.debug("Channel statistics after normalization:")
        logger.debug(f"- Mean magnitude: {tf.reduce_mean(tf.abs(h_freq))}")
        logger.debug(f"- Max magnitude: {tf.reduce_max(tf.abs(h_freq))}")
        logger.debug(f"- Min magnitude: {tf.reduce_min(tf.abs(h_freq))}")

        # When calculating SNR, use a minimum signal power threshold
        signal_power = tf.maximum(tf.reduce_mean(tf.abs(h_freq)**2, axis=-1), epsilon)
        noise_power = tf.constant(1e-13, dtype=tf.float32)  # Adjust this value based on your system
        snr_db = 10.0 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)

        # Add clipping to prevent -inf SNR values
        min_snr_db = -50.0  # Adjust this value based on your requirements
        snr_db = tf.maximum(snr_db, min_snr_db)
        

        # Calculate SNR
        average_snr = calculate_snr(h_freq)

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

        noise_power = tf.constant(1e-13, dtype=tf.float32)
        snr_db = 10.0 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)

        # Add SNR debug messages right after SNR calculation
        logger.debug(f"SNR statistics:")
        logger.debug(f"- Raw SNR values: {snr_db}")
        logger.debug(f"- Contains inf: {tf.reduce_any(tf.math.is_inf(snr_db))}")
        logger.debug(f"- Contains nan: {tf.reduce_any(tf.math.is_nan(snr_db))}")

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
            'snr_db': snr_db.numpy(),
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
                'path_conditions': los_conditions.numpy()
                
            }
        }
        
        # Log key metrics only if paths exist
        if total_paths > 0:
            logger.info(f"Channel Generation Metrics:")
            logger.info(f"LOS Ratio: {los_statistics['los_ratio']:.2f}")
            logger.info(f"Average SNR: {tf.reduce_mean(snr_db):.2f} dB")
            logger.info(f"Maximum Doppler Shift: {tf.reduce_max(tf.abs(doppler_shifts)):.2f} Hz")
        
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

def main():
    """Main execution function"""
    result_dir = ensure_result_dir()
    scene = None
    
    try:
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # First generate PLY files in src/meshes
        print("Generating PLY files...")
        SionnaPLYGenerator.generate_factory_geometries(
            room_dims=[20, 20, 5],
            shelf_dims=[2, 1, 4],
            output_dir=os.path.join(os.path.dirname(__file__), 'meshes')  # This ensures meshes is created in src
        )
        print("PLY files generated successfully")
        
        # Initialize configuration
        config = SmartFactoryConfig()
        validate_config(config)
        
        # Setup scene
        scene = setup_scene(config)
        if not scene:
            raise ValueError("Scene setup failed")
        
        # Set scene frequency
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        
        # Generate channel data
        print("Generating channel data...")
        channel_data = generate_channel_data(scene, config)
        
        # Save channel data
        h5_filepath = os.path.join(result_dir, 'channel_data.h5')
        save_channel_data(channel_data, h5_filepath)
        print(f"Channel data saved to: {h5_filepath}")
        
        # Print basic statistics
        print("\nChannel Data Statistics:")
        print(f"Channel matrices shape: {channel_data['channel_matrices'].shape}")
        print(f"Path delays shape: {channel_data['path_delays'].shape}")
        print(f"Number of receivers: {len(channel_data['agv_positions'])}")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        logger.error(f"Error in execution: {str(e)}")
        raise
        
    finally:
        print("Execution completed")

if __name__ == "__main__":
    main()