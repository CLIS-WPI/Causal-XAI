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

    
    # Add typical path loss for indoor factory at 28GHz
    config.path_loss_db = 80  # Typical value for indoor factory environment

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
        
        # main simulation loop:
        for iteration in range(config.num_time_steps):
            print(f"\rSimulating step {iteration+1}/{config.num_time_steps}", end="")
            
            # Update AGV positions
            agv_positions = []
            for i in range(config.num_agvs):
                agv_id = str(i)  # Just use the index as ID
                current_pos = scene.receivers[f'agv_{i}'].position  # Keep 'agv_' prefix only for scene receivers
                new_pos = path_manager.get_next_position(agv_id, current_pos)
                scene.receivers[f'agv_{i}'].position = new_pos
                agv_positions.append(new_pos)
            
            # Generate channel data
            channel_generator = SmartFactoryChannel(config, scene)
            channel_data = channel_generator.generate_channel_data(config)
            
            # Log metrics every 10 steps
            if iteration % 10 == 0:
                print(f"\nStep {iteration+1} metrics:")
                if 'average_snr' in channel_data:
                    print(f"Average SNR: {channel_data['average_snr']:.2f} dB")
                elif 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                    avg_snr = np.mean(channel_data['beam_metrics']['snr_db'])
                    print(f"Average SNR: {avg_snr:.2f} dB")
                else:
                    print("SNR data not available")
            
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