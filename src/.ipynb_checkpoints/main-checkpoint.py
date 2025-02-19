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
import h5py
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.channel.utils import subcarrier_frequencies
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

def generate_channel_data(scene, config):
    """Generate channel data using ray tracing"""
    try:
        print("Generating channel data...")
        
        # Compute paths using ray tracing with correct parameters
        paths = scene.compute_paths(
            max_depth=config.ray_tracing['max_depth'],
            method="fibonacci",
            num_samples=config.ray_tracing['num_samples'],
            los=config.ray_tracing['los'],
            reflection=config.ray_tracing['reflection'],
            diffraction=config.ray_tracing['diffraction'],
            scattering=config.ray_tracing['scattering'],
            ris=True,  # Enable RIS paths
            scat_keep_prob=0.001,  # Default scattering keep probability
            edge_diffraction=False  # Default edge diffraction setting
        )
        
        # Get channel impulse responses
        a, tau = paths.cir()
        
        # Calculate frequencies for the subcarriers
        frequencies = subcarrier_frequencies(
            num_subcarriers=config.num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing
        )
        
        # Convert to OFDM channel
        h_freq = cir_to_ofdm_channel(
            frequencies=frequencies,
            a=tf.convert_to_tensor(a),
            tau=tf.convert_to_tensor(tau),
            normalize=False
        )
        
        # Create channel data dictionary with corrected LOS attribute name
        channel_data = {
            'channel_matrices': h_freq,
            'path_delays': tau,
            'los_conditions': paths.LOS,  # Changed from los to LOS
            'agv_positions': tf.stack([rx.position for rx in scene.receivers.values()])
        }
        
        print("Channel data generation completed")
        return channel_data
        
    except Exception as e:
        logger.error(f"Error generating channel data: {str(e)}")
        raise

def save_channel_data(channel_data, filepath):
    """Save channel data to H5 file"""
    with h5py.File(filepath, 'w') as f:
        for key, value in channel_data.items():
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            f.create_dataset(key, data=value)

def main():
    """Main execution function"""
    result_dir = ensure_result_dir()
    scene = None
    
    try:
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)
        
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