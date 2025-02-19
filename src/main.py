##main.py
import tensorflow as tf
from config import SmartFactoryConfig
from scene_setup import setup_scene, verify_los_paths
import os
import time
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
from tensorflow import autograph
from data_store import save_performance_metrics

tf.config.run_functions_eagerly(True)
# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure TensorFlow and XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation

# Silence AutoGraph warnings
tf.autograph.set_verbosity(0)

def generate_channel(channel_generator, config):
    return channel_generator.generate_channel_data(config)

def setup_logging():
    """Configure logging with colors and file output"""
    # ANSI escape codes for colors
    class ColorFormatter(logging.Formatter):
        COLORS = {
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[31m',   # Red
            'DEBUG': '\033[37m',      # Light Gray
            'INFO': '\033[0m',        # Default
            'RESET': '\033[0m'        # Reset
        }

        def format(self, record):
            # Add colors if the level has a color defined
            if record.levelname in self.COLORS:
                record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
                record.msg = f"{self.COLORS[record.levelname]}{record.msg}{self.COLORS['RESET']}"
            return super().format(record)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatters
    console_formatter = ColorFormatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler (with colors)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler('smart_factory.log')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger

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
            # Create main groups
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
                    
            # Handle beam metrics with type checking and conversion
            if 'beam_metrics' in channel_data:
                metrics_group = beam_group.create_group('beam_metrics')
                for metric_name, metric_value in channel_data['beam_metrics'].items():
                    try:
                        # Convert dict to JSON string if necessary
                        if isinstance(metric_value, dict):
                            metrics_group.attrs[metric_name] = str(metric_value)
                        # Convert numpy array or tensor
                        elif isinstance(metric_value, (np.ndarray, tf.Tensor)):
                            metrics_group.create_dataset(metric_name, data=np.array(metric_value))
                        # Convert basic numeric types
                        elif isinstance(metric_value, (int, float, bool)):
                            metrics_group.attrs[metric_name] = metric_value
                        # Convert lists to numpy arrays
                        elif isinstance(metric_value, list):
                            metrics_group.create_dataset(metric_name, data=np.array(metric_value))
                        else:
                            logger.warning(f"Skipping unsupported metric type: {metric_name} ({type(metric_value)})")
                    except Exception as e:
                        logger.warning(f"Could not save metric {metric_name}: {str(e)}")
            
            # Save beam adaptation data
            if 'beam_adaptation' in channel_data:
                adapt_group = beam_group.create_group('adaptation')
                for key, value in channel_data['beam_adaptation'].items():
                    if value is not None:
                        if isinstance(value, dict):
                            # Convert dict to string representation
                            adapt_group.attrs[key] = str(value)
                        elif isinstance(value, (np.ndarray, list)):
                            adapt_group.create_dataset(key, data=np.array(value))
                        else:
                            adapt_group.attrs[key] = value
                            
            # Save temporal data
            if 'temporal_data' in channel_data:
                for key, value in channel_data['temporal_data'].items():
                    if isinstance(value, (np.ndarray, tf.Tensor)):
                        temporal_group.create_dataset(key, data=np.array(value))
                    else:
                        temporal_group.attrs[key] = str(value) if isinstance(value, dict) else value
                        
            # Add metadata
            f.attrs['creation_time'] = str(datetime.now())
            f.attrs['num_receivers'] = len(channel_data.get('agv_positions', []))
            if 'channel_matrices' in channel_data:
                f.attrs['matrix_shape'] = str(channel_data['channel_matrices'].shape)
            
            # Add beam switching metadata
            if 'beam_metrics' in channel_data:
                if 'snr_db' in channel_data['beam_metrics']:
                    snr_values = channel_data['beam_metrics']['snr_db']
                    if isinstance(snr_values, (np.ndarray, list)):
                        f.attrs['avg_snr'] = float(np.mean(snr_values))
                if 'beam_history' in channel_data.get('beam_adaptation', {}):
                    f.attrs['num_beam_switches'] = len(channel_data['beam_adaptation']['beam_history'])
            
            logger.info(f"Channel data successfully saved to: {filepath}")
            logger.debug(f"Saved data groups: {list(f.keys())}")
            
    except Exception as e:
        logger.error(f"Error saving channel data to {filepath}: {str(e)}")
        raise

def main():
    """Streamlined main execution focusing on beam switching"""
    # Setup logging first with colors and file output
    logger = setup_logging()
    logger.info("Starting smart factory beam switching simulation...")

    try:
        # Rest of your main function code...
        print("Starting simulation...")
        logger.info("Starting smart factory beam switching simulation...")
        
        # Create results directory
        result_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(result_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {result_dir}")
        
        tf.random.set_seed(42)
        
        # Initialize configuration and validate
        logger.info("Initializing configuration...")
        config = SmartFactoryConfig()
        validate_config(config)
        
        # Generate scene geometry
        logger.info("Setting up simulation environment...")
        ply_output_dir = os.path.join(os.path.dirname(__file__), 'meshes')
        logger.info(f"Generating geometries in: {ply_output_dir}")
        
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
        
        # Create dataset file
        dataset_file = os.path.join(result_dir, 'channel_dataset.h5')
        
        # Initialize performance tracking
        performance_metrics = {
            'beam_switches': [],
            'packet_stats': [],
            'ber_history': [],
            'snr_history': [],
            'packet_stats': {
                'total': 0,
                'successful': 0,
                'failed_during_switch': 0,
            }
        }

        # Add these lines:
        switch_timing_metrics = {
            'switch_start_time': None,
            'switch_durations': [],
            'packet_success_count': 0,
            'total_packets': 0,
            'ber_during_switch': [],
            'snr_during_switch': []
        }

        # Main simulation loop:
        for iteration in range(config.num_time_steps):
            print(f"\rSimulating step {iteration+1}/{config.num_time_steps}", end="")
            
            # Update AGV positions
            agv_positions = []
            for i in range(config.num_agvs):
                agv_id = f'agv_{i}'  # Use consistent format
                current_pos = scene.receivers[f'agv_{i}'].position
                new_pos = path_manager.get_next_position(agv_id, current_pos)
                scene.receivers[f'agv_{i}'].position = new_pos
                agv_positions.append(new_pos)
            
            # Generate channel data
            channel_generator = SmartFactoryChannel(config, scene)
            channel_data = generate_channel(channel_generator, config)

            # Get optimal beam direction based on channel conditions
            optimal_beam = beam_manager.optimize_beam_direction(
                channel_data, 
                path_manager,
                config.scene_objects.get('obstacles', [])
            )

            # Update the beam configuration
            beam_manager.update_beam(optimal_beam)

            # Now check if switch occurred
            if beam_manager.has_switch_occurred():
                switch_timing_metrics['switch_start_time'] = time.time()
                
                # Calculate BER during switch (if available)
                if hasattr(channel_generator, 'calculate_ber'):
                    current_ber = channel_generator.calculate_ber(channel_data)
                    switch_timing_metrics['ber_during_switch'].append(current_ber)
                
                # Track SNR during switch
                if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                    switch_timing_metrics['snr_during_switch'].append(
                        np.mean(channel_data['beam_metrics']['snr_db'])
                    )
                
                # Calculate switch duration if switch completed
                if switch_timing_metrics['switch_start_time'] is not None:
                    switch_duration = time.time() - switch_timing_metrics['switch_start_time']
                    switch_timing_metrics['switch_durations'].append(switch_duration)
                    switch_timing_metrics['switch_start_time'] = None

            # Track packet success rate
            switch_timing_metrics['total_packets'] += 1
            if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                if np.mean(channel_data['beam_metrics']['snr_db']) > config.beamforming['min_snr_threshold']:
                    switch_timing_metrics['packet_success_count'] += 1
            
            # Add temporal information
            channel_data['temporal_data'] = {
                'timestamp': iteration,
                'agv_positions': agv_positions
            }
            
            # Save intermediate results every N steps
            if iteration % 10 == 0:
                intermediate_file = os.path.join(result_dir, f'channel_data_step_{iteration}.h5')
                save_channel_data(channel_data, intermediate_file)
            
            # Log metrics every 10 steps
            if iteration % 10 == 0:
                print(f"\nStep {iteration+1} metrics:")
                if 'average_snr' in channel_data:
                    print(f"Average SNR: {channel_data['average_snr']:.2f} dB")
                elif 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                    avg_snr = np.mean(channel_data['beam_metrics']['snr_db'])
                    print(f"Average SNR: {avg_snr:.2f} dB")
            
            channel_data_history.append(channel_data)
        
        print("\nSimulation completed. Saving results...")
        
        # Calculate final performance metrics
        performance_summary = {
            'beam_switching': {
                'average_switch_time': np.mean(switch_timing_metrics['switch_durations']) if switch_timing_metrics['switch_durations'] else 0,
                'total_switches': len(switch_timing_metrics['switch_durations']),
                'packet_success_rate': (switch_timing_metrics['packet_success_count'] / 
                                    switch_timing_metrics['total_packets'] if switch_timing_metrics['total_packets'] > 0 else 0),
                'average_ber_during_switch': np.mean(switch_timing_metrics['ber_during_switch']) if switch_timing_metrics['ber_during_switch'] else 0,
                'snr_variation_during_switch': np.std(switch_timing_metrics['snr_during_switch']) if switch_timing_metrics['snr_during_switch'] else 0
            }
        }

        # Prepare final dataset
        # The final dataset preparation is correct as is:
        final_dataset = {
            'performance_metrics': performance_summary,
            'channel_data': channel_data_history[-1],
            'beam_history': beam_manager.get_beam_history(),
            'path_data': {
                'final_positions': {
                    f'agv_{i}': path_manager.get_current_status(f'agv_{i}')
                    for i in range(config.num_agvs)
                }
            },
            'temporal_data': {
                'total_steps': config.num_time_steps,
                'simulation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # Add this line at the end to save performance metrics separately
        save_performance_metrics(performance_metrics, os.path.join(result_dir, 'performance_metrics.h5'))
                
        # Save final dataset
        save_channel_data(final_dataset, dataset_file)
        print(f"Dataset saved to: {dataset_file}")
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()