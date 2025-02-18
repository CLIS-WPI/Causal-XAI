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
from agv_path_manager import AGVPathManager

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
    """Simplified channel data generation focusing on beam switching"""
    try:
        logger.debug("Generating channel data...")
        
        # Apply beam configuration
        if beam_manager is not None:
            current_beams = beam_manager.get_current_beams()
            if current_beams is not None:
                for tx in scene.transmitters.values():
                    tx.array.steering_angle = current_beams
        
        # Compute paths
        paths = scene.compute_paths(
            max_depth=4,
            method="fibonacci",
            num_samples=2000,
            los=True,
            reflection=True,
            diffraction=True,
            scattering=True
        )
        
        if paths is None:
            raise ValueError("Path computation failed")
        
        # Get channel impulse responses
        a, tau = paths.cir()
        
        # Convert to OFDM channel
        frequencies = subcarrier_frequencies(
            num_subcarriers=config.num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing
        )
        
        h_freq = cir_to_ofdm_channel(
            frequencies=frequencies,
            a=tf.cast(a, tf.complex64),
            tau=tf.cast(tau, tf.float32),
            normalize=True
        )
        
        # Calculate SNR
        snr = calculate_snr(h_freq, config)
        
        return {
            'channel_matrices': h_freq,
            'path_delays': tau,
            'los_conditions': paths.LOS,
            'average_snr': tf.reduce_mean(snr),
            'beam_metrics': {
                'snr_db': snr.numpy(),
                'avg_power': float(tf.reduce_mean(tf.abs(h_freq)**2))
            }
        }
        
    except Exception as e:
        logger.error(f"Channel data generation failed: {str(e)}")
        raise

def save_channel_data(channel_data, filepath):
    """
    Save channel data to H5 file with enhanced organization and error handling
    
    Args:
        channel_data: Dictionary containing channel simulation data
        filepath: Path to save the H5 file
    """
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
            
            # Save beam adaptation data if available
            if 'beam_adaptation' in channel_data:
                beam_adapt_group = beam_group.create_group('adaptation')
                for key, value in channel_data['beam_adaptation'].items():
                    if value is not None:
                        if isinstance(value, dict):
                            sub_group = beam_adapt_group.create_group(key)
                            for sub_key, sub_value in value.items():
                                if sub_value is not None:
                                    sub_group.create_dataset(sub_key, data=np.array(sub_value))
                        else:
                            beam_adapt_group.create_dataset(key, data=np.array(value))
                            
            # Save temporal data
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
            
            # Add beam switching specific metadata
            if 'beam_metrics' in channel_data:
                f.attrs['avg_snr'] = float(np.mean(channel_data['beam_metrics'].get('snr_db', [0])))
                f.attrs['num_beam_switches'] = len(channel_data.get('beam_adaptation', {}).get('beam_history', []))
            
            logger.info(f"Channel data successfully saved to: {filepath}")
            logger.debug(f"Saved data groups: {list(f.keys())}")
            
    except Exception as e:
        logger.error(f"Error saving channel data to {filepath}: {str(e)}")
        raise


def main():
    """Streamlined main execution focusing on beam switching"""
    try:
        # Basic setup
        print("Starting simulation...")
        logger.info("Starting smart factory beam switching simulation...")
        result_dir = ensure_result_dir()
        print(f"Results will be saved to: {result_dir}")
        
        tf.random.set_seed(42)
        
        # Initialize configuration and validate
        print("Initializing configuration...")
        config = SmartFactoryConfig()
        add_snr_config(config)
        validate_config(config)
        
        # Generate scene geometry
        print("Setting up simulation environment...")
        logger.info("Setting up simulation environment...")
        ply_output_dir = os.path.join(os.path.dirname(__file__), 'meshes')
        print(f"Generating geometries in: {ply_output_dir}")
        
        SionnaPLYGenerator.generate_factory_geometries(
            config=config,
            output_dir=ply_output_dir
        )
        
        # Setup scene and initialize managers
        print("Setting up scene...")
        scene = setup_scene(config)
        if not scene:
            raise ValueError("Scene setup failed")
            
        print("Initializing managers...")
        path_manager = AGVPathManager(config)
        beam_manager = BeamManager(config)
        
        # Set scene frequency
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        
        # Main simulation loop
        print(f"Starting simulation for {config.num_time_steps} time steps...")
        logger.info("Starting beam switching simulation...")
        channel_data_history = []
        
        for iteration in range(config.num_time_steps):
            print(f"\rSimulating step {iteration+1}/{config.num_time_steps}", end="")
            
            # Update AGV positions
            agv_positions = []
            for i in range(config.num_agvs):
                agv_id = f'agv_{i+1}'
                current_pos = scene.receivers[agv_id].position
                new_pos = path_manager.get_next_position(agv_id, current_pos)
                scene.receivers[agv_id].position = new_pos
                agv_positions.append(new_pos)
            
            # Generate channel data
            channel_generator = SmartFactoryChannel(config, scene)
            channel_data = channel_generator.generate_channel_data(config)
            
            # Log metrics every 10 steps
            if iteration % 10 == 0:
                print(f"\nStep {iteration+1} metrics:")
                print(f"Average SNR: {channel_data['average_snr']:.2f} dB")
            
            channel_data_history.append(channel_data)
        
        print("\nSimulation completed. Saving results...")
        
        # Save final results
        final_data = {
            'channel_data': channel_data_history[-1],
            'beam_history': beam_manager.get_beam_history(),
            'path_data': {
                'final_positions': {
                    agv_id: path_manager.get_current_status(agv_id)
                    for agv_id in [f'agv_{i+1}' for i in range(config.num_agvs)]
                }
            }
        }
        
        results_file = os.path.join(result_dir, 'beam_switching_results.h5')
        save_channel_data(final_data, results_file)
        print(f"Results saved to: {results_file}")
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()