#main.py
#tttt the very top of main.py (before all imports)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_mitsuba_variant
ensure_mitsuba_variant('cuda_ad_rgb')
import mitsuba
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
from config import SmartFactoryConfig
from scene_setup import setup_scene, verify_los_paths
import time
import numpy as np
from datetime import datetime
import sionna
from sionna.rt import Scene
import logging
logger = logging.getLogger(__name__)
import h5py
from sionna_ply_generator import SionnaPLYGenerator
from beam_manager import BeamManager
from channel_generator import SmartFactoryChannel
from agv_path_manager import AGVPathManager
from data_store import save_performance_metrics

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Number of GPUs available:', len(gpus))
        print('GPUs:', gpus)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Clear GPU memory using TensorFlow instead of PyTorch
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Optional: Force garbage collection on GPU
        try:
            # Only if running on Linux
            import subprocess
            subprocess.run('nvidia-smi -r', shell=True, check=False)
        except:
            pass
            
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

logger.debug(f"Initializing MirroredStrategy - Number of devices: {strategy.num_replicas_in_sync}")
logger.debug(f"Strategy scope active: {hasattr(strategy, '_scope')}")  # بررسی وضعیت scope

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True, 'constant_folding': True, 'shape_optimization': True,
    'remapping': True, 'arithmetic_optimization': True, 'dependency_optimization': True,
    'loop_optimization': True, 'function_optimization': True, 'debug_stripper': True,
    'scoped_allocator_optimization': True, 'implementation_selector': True,
    'auto_mixed_precision': True, 'min_graph_nodes': 1,
})
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
tf.config.run_functions_eagerly(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def clear_memory():
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

tf.keras.backend.clear_session()

def print_gpu_utilization():
    import subprocess
    try:
        gpu_stats = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
        gpu_stats = gpu_stats.decode("utf-8").strip().split('\n')
        for i, stats in enumerate(gpu_stats):
            util, mem = stats.split(', ')
            print(f"GPU {i}: Utilization {util}%, Memory Used {mem}MB")
    except:
        print("Could not get GPU statistics")


def setup_logging():
    class ColorFormatter(logging.Formatter):
        COLORS = {
            'WARNING': '\033[33m', 'ERROR': '\033[31m', 'CRITICAL': '\033[31m',
            'DEBUG': '\033[37m', 'INFO': '\033[0m', 'RESET': '\033[0m'
        }
        def format(self, record):
            original_levelname, original_msg = record.levelname, record.msg
            if record.levelname in self.COLORS:
                record.msg = f"{self.COLORS[record.levelname]}{record.msg}{self.COLORS['RESET']}"
            formatted_record = super().format(record)
            record.levelname, record.msg = original_levelname, original_msg
            return formatted_record

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('smart_factory.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def convert_to_numpy(data):
    if isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif hasattr(data, '_values'):
        return convert_to_numpy(strategy.experimental_local_results(data)[0])
    elif isinstance(data, tf.Tensor):
        return data.numpy()
    return data

def ensure_result_dir():
    result_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def validate_config(config):
    required_attrs = ['carrier_frequency', 'num_time_steps', 'num_agvs', 'room_dim', 'bs_array']
    missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing_attrs:
        raise ValueError(f"Missing required configuration attributes: {missing_attrs}")
    if config.carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be positive")

def cleanup():
    tf.keras.backend.clear_session()
    try:
        # Get the current TensorFlow version
        tf_version = tf.__version__
        
        # Try to access the current strategy
        try:
            current_strategy = tf.distribute.get_strategy()
            logger.debug(f"Current strategy type: {type(current_strategy)}")
            
            # Check for the _distribution_strategy_stack attribute
            if hasattr(current_strategy, '_distribution_strategy_stack'):
                stack = current_strategy._distribution_strategy_stack
                logger.debug(f"Strategy stack content: {stack}")
                
                # Use safer approach to clear stack
                while len(stack) > 0:
                    try:
                        stack.pop()
                    except (IndexError, AttributeError):
                        break
            
            # Only do this for newer TF versions that have this API
            if hasattr(tf.distribute, 'experimental_reset_distribution_strategy'):
                tf.distribute.experimental_reset_distribution_strategy()
                logger.debug("Distribution strategy reset successfully")
            else:
                logger.info("TensorFlow version doesn't support explicit strategy reset")
                
        except (AttributeError, ValueError) as e:
            logger.warning(f"Could not access current strategy: {str(e)}")
            
    except Exception as e:
        logger.error(f"Strategy cleanup error: {str(e)}")
        # Don't re-raise, just log - this is cleanup code
        
    # Force garbage collection
    import gc
    gc.collect()

def safe_to_numpy(data):
    return data.numpy() if isinstance(data, tf.Tensor) else data

def get_batch_shape(batch):
    if hasattr(batch, 'values'):
        return batch.values[0].shape  
    return batch.shape  

def main():
    logger = setup_logging()
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
        handler.flush = lambda: sys.stdout.flush()
    logger.info("Starting smart factory beam switching simulation...")
    
    try:
        print("Starting simulation...")
        logger.info("Starting smart factory beam switching simulation...")
        
        result_dir = ensure_result_dir()
        logger.info(f"Results will be saved to: {result_dir}")
        
        tf.random.set_seed(42)
        config = SmartFactoryConfig()
        validate_config(config)
        logger.info("Configuration initialized successfully")
        logger.info(f"Total time steps from config: {config.num_time_steps}")
        
        scene = setup_scene(config)
        print("Current Mitsuba variant:", mitsuba.variant())
        if not scene:
            raise ValueError("Scene setup failed")
        logger.info("Scene setup completed")

        with strategy.scope():
            logger.debug("Entering MirroredStrategy scope")
            agv_manager = AGVPathManager(config, scene)
            beam_manager = BeamManager(config)
            channel_generator = SmartFactoryChannel(config, scene)
            scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
            logger.info("Scene frequency set")

            obstacle_positions = tf.constant([
                [5.0, 5.0, 1.0],
                [15.0, 15.0, 1.0],
                [10.0, 8.0, 2.0]
            ], dtype=tf.float32)
            logger.info(f"Obstacle positions defined: {obstacle_positions.numpy()}")

            print("Generating AGV paths...")
            logger.info("Generating AGV paths...")
            agv1_path = np.linspace([2, 2, 0.5], [18, 2, 0.5], config.num_time_steps, dtype=np.float32)
            agv2_trajectory = np.array(config.agv_trajectories['agv_2'], dtype=np.float32)
            t = np.linspace(0, 1, config.num_time_steps, dtype=np.float32)
            agv2_x = np.interp(t, np.linspace(0, 1, len(agv2_trajectory), dtype=np.float32), agv2_trajectory[:, 0])
            agv2_y = np.interp(t, np.linspace(0, 1, len(agv2_trajectory), dtype=np.float32), agv2_trajectory[:, 1])
            agv2_z = np.full(config.num_time_steps, 0.5, dtype=np.float32)
            agv2_path = np.stack([agv2_x, agv2_y, agv2_z], axis=1)
            positions = np.stack([agv1_path, agv2_path], axis=1, dtype=np.float32)
            logger.info("AGV paths generated")

            print("Creating distributed dataset...")
            logger.info("Creating distributed dataset...")
            batch_size = 16
            base_dataset = tf.data.Dataset.from_tensor_slices(positions).padded_batch(
                batch_size, padded_shapes=[batch_size, 3], padding_values=0.0
            ).prefetch(tf.data.AUTOTUNE)
            logger.info("Dataset distributed successfully")

            dataset_file = os.path.join(result_dir, 'simulation_data.h5')
            performance_metrics = {'beam_switches': [], 'ber_history': [], 'snr_history': []}
            switch_timing_metrics = {
                'switch_start_time': None, 'switch_durations': [],
                'packet_success_count': 0, 'total_packets': 0,
                'ber_during_switch': [], 'snr_during_switch': []
            }

            print(f"Opening HDF5 file: {dataset_file}")
            logger.info(f"Opening HDF5 file for writing: {dataset_file}")
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    with h5py.File(dataset_file, 'w') as h5f:
                        h5f.attrs['config'] = str(vars(config))
                        h5f.attrs['simulation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        h5f.attrs['total_steps'] = config.num_time_steps
                        logger.info("HDF5 file opened and attributes set")

                        print(f"TensorFlow version: {tf.__version__}")
                        print("Simulation loop started.")
                        logger.info("Entering simulation loop...")
                        total_steps = config.num_time_steps
                        start_time = time.time()

                        def process_batch(batch):
                            logger.debug("Processing batch in MirroredStrategy scope")
                            agv_positions = batch[0]
                            for i in range(config.num_agvs):
                                scene.receivers[f'rx_agv_{i}'].position = agv_positions[i]
                            raw_channel_data = channel_generator.generate_channel_data(config)
                            logger.debug("Batch processed - Channel data keys: {raw_channel_data.keys() if isinstance(raw_channel_data, dict) else 'None'}")
                            return raw_channel_data

                        tf.profiler.experimental.start('logdir')
                        for step in range(total_steps):
                            progress = (step + 1) / total_steps * 100
                            elapsed_time = time.time() - start_time
                            eta = (elapsed_time / (step + 1)) * (total_steps - (step + 1)) if step > 0 else 0
                            eta_str = f"{int(eta // 60)}m {int(eta % 60)}s" if eta > 0 else "Calculating..."
                            print(f"\rSimulating step {step+1}/{total_steps} ({progress:.1f}%) - ETA: {eta_str}", end="")

                            if step % 10 == 0:
                                clear_memory()
                                tf.keras.backend.clear_session()
                                print_gpu_utilization()
                                logger.info(f"Progress: {progress:.1f}%, Step {step+1}/{total_steps}, ETA: {eta_str}")

                            try:
                                # هر قدم یه iterator جدید بساز
                                dataset_iterator = iter(strategy.experimental_distribute_dataset(base_dataset))
                                logger.debug(f"Dataset iterator created - Strategy replicas: {strategy.num_replicas_in_sync}")
                                batch = next(dataset_iterator)
                                logger.debug(f"Batch fetched - Shape: {get_batch_shape(batch)}")
                                channel_data = strategy.run(process_batch, args=(batch,))
                                logger.debug(f"Channel data processed - Type: {type(channel_data)}, Keys: {channel_data.keys() if isinstance(channel_data, dict) else 'Not a dict'}")
                                channel_data = convert_to_numpy(channel_data)
                                
                                if channel_data is None or not isinstance(channel_data, dict):
                                    logger.error(f"Step {step}: Channel data is invalid, using fallback")
                                    real_part = tf.random.normal([config.num_agvs, config.num_subcarriers], dtype=tf.float32)
                                    imag_part = tf.random.normal([config.num_agvs, config.num_subcarriers], dtype=tf.float32)
                                    fallback_channel = tf.complex(real_part, imag_part)
                                    channel_data = {
                                        'channel_matrices': fallback_channel,
                                        'path_delays': tf.zeros([1, config.num_agvs, 1], dtype=tf.float32),
                                        'los_conditions': tf.zeros([config.num_agvs], dtype=tf.int32),
                                        'agv_positions': agv_manager.update_positions(),
                                        'path_losses': tf.zeros([config.num_agvs], dtype=tf.float32),
                                        'beam_metrics': {'snr_db': tf.zeros([config.num_agvs], dtype=tf.float32)},
                                        'path_data': {
                                            'path_powers': tf.zeros([1, config.num_agvs, 1]),
                                            'path_directions': tf.zeros([1, config.num_agvs, 1, 2])
                                        }
                                    }

                                optimal_beam = beam_manager.optimize_beam_direction(channel_data, agv_manager, obstacle_positions)
                                logger.debug(f"Step {step}: Optimal beam: {optimal_beam}")

                                for i in range(config.num_agvs):
                                    agv_id = f'agv_{i}'
                                    los_status = channel_data['los_conditions'][i] if 'los_conditions' in channel_data else 0
                                    position = channel_data['agv_positions'][i]
                                    agv_manager.record_movement(agv_id, position, agv_manager.current_velocities[agv_id], los_status)

                                success = False
                                if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                                    snr_per_agv = channel_data['beam_metrics']['snr_db']
                                    mean_snr = np.mean(snr_per_agv)
                                    success = mean_snr > config.beamforming['min_snr_threshold']
                                    beam_manager.log_snr(mean_snr)
                                    logger.debug(f"Step {step}: SNR per AGV: {snr_per_agv}, Mean SNR: {mean_snr}, Success: {success}")
                                else:
                                    logger.warning(f"No SNR data at step {step}")

                                beam_manager.update_beam(optimal_beam, success=success)

                                if beam_manager.has_switch_occurred() and beam_manager.switch_times:
                                    switch_timing_metrics['switch_start_time'] = time.time()
                                    if hasattr(channel_generator, 'calculate_ber'):
                                        current_ber = channel_generator.calculate_ber(channel_data)
                                        switch_timing_metrics['ber_during_switch'].append(current_ber)
                                    if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                                        switch_timing_metrics['snr_during_switch'].append(np.mean(channel_data['beam_metrics']['snr_db']))
                                    if switch_timing_metrics['switch_start_time'] is not None:
                                        switch_duration = time.time() - switch_timing_metrics['switch_start_time']
                                        switch_timing_metrics['switch_durations'].append(switch_duration)
                                        switch_timing_metrics['switch_start_time'] = None

                                switch_timing_metrics['total_packets'] += 1
                                if success:
                                    switch_timing_metrics['packet_success_count'] += 1

                                with h5py.File(dataset_file, 'a') as h5f:
                                    step_group = h5f.create_group(f'step_{step}')
                                    csi_group = step_group.create_group('csi_data')
                                    mobility_group = step_group.create_group('mobility_data')
                                    beam_group = step_group.create_group('beam_data')

                                    csi_group.create_dataset('channel_matrices', data=safe_to_numpy(channel_data['channel_matrices']), compression='gzip')
                                    csi_group.create_dataset('path_delays', data=safe_to_numpy(channel_data['path_delays']), compression='gzip')

                                    mobility_group.create_dataset('agv_positions', data=safe_to_numpy(channel_data['agv_positions']), compression='gzip')
                                    velocities = np.array([agv_manager.current_velocities[f'agv_{i}'] for i in range(config.num_agvs)])
                                    mobility_group.create_dataset('velocities', data=velocities, compression='gzip')
                                    los_conditions_data = safe_to_numpy(channel_data['los_conditions']) if 'los_conditions' in channel_data else np.zeros(config.num_agvs, dtype=np.int32)
                                    mobility_group.create_dataset('los_conditions', data=los_conditions_data, compression='gzip')
                                    distance_to_bs = np.linalg.norm(safe_to_numpy(channel_data['agv_positions']) - np.array(config.bs_position), axis=1)
                                    mobility_group.create_dataset('distance_to_bs', data=distance_to_bs, compression='gzip')
                                    blocked = safe_to_numpy(beam_manager.detect_blockage(channel_data, channel_data['agv_positions'], obstacle_positions))
                                    mobility_group.create_dataset('blockage_status', data=blocked, compression='gzip')

                                    bs_pos = tf.cast(config.bs_position, tf.float32)
                                    direction_vectors = channel_data['agv_positions'] - bs_pos
                                    azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
                                    horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1)
                                    elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
                                    azimuths = tf.where(azimuths < 0, azimuths + 360, azimuths)
                                    elevations = tf.clip_by_value(elevations, -30, 30)
                                    azimuths = tf.clip_by_value(azimuths, -config.beamforming['max_steering_angle'], config.beamforming['max_steering_angle'])
                                    relative_angles = tf.stack([azimuths, elevations], axis=1).numpy()

                                    beam_group.create_dataset('beam_directions', data=optimal_beam, compression='gzip')
                                    snr_db_data = safe_to_numpy(channel_data['beam_metrics']['snr_db']) if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics'] else np.zeros(config.num_agvs, dtype=np.float32)
                                    beam_group.create_dataset('snr_db', data=snr_db_data, compression='gzip')
                                    beam_group.create_dataset('path_powers', data=safe_to_numpy(channel_data['path_data']['path_powers']), compression='gzip')
                                    beam_group.create_dataset('path_directions', data=safe_to_numpy(channel_data['path_data']['path_directions']), compression='gzip')
                                    beam_group.create_dataset('relative_angle', data=relative_angles, compression='gzip')
                                    last_switch = beam_manager.switch_times[-1] if beam_manager.has_switch_occurred() and beam_manager.switch_times else {}
                                    switch_reason = last_switch.get('reason', 'None')
                                    beam_group.attrs['switch_reason'] = switch_reason

                                    step_group.attrs['timestamp'] = step
                                    step_group.attrs['success'] = success

                                if step % 2 == 0:
                                    avg_snr = np.mean(snr_db_data)
                                    print(f"\nStep {step+1} metrics: Average SNR: {avg_snr:.2f} dB")

                                clear_memory()
                                tf.keras.backend.clear_session()

                            except Exception as e:
                                logger.error(f"Step {step}: Error processing batch: {str(e)}")
                                continue

                        tf.profiler.experimental.stop()
                        print("\nSimulation completed. Saving performance metrics...")
                        logger.info("Simulation completed. Saving performance metrics...")
                        performance_summary = {
                            'beam_switching': {
                                'average_switch_time': np.mean(switch_timing_metrics['switch_durations']) if switch_timing_metrics['switch_durations'] else 0,
                                'total_switches': len(switch_timing_metrics['switch_durations']),
                                'packet_success_rate': (switch_timing_metrics['packet_success_count'] / 
                                                        switch_timing_metrics['total_packets']) if switch_timing_metrics['total_packets'] > 0 else 0,
                                'average_ber_during_switch': np.mean(switch_timing_metrics['ber_during_switch']) if switch_timing_metrics['ber_during_switch'] else 0,
                                'snr_variation_during_switch': np.std(switch_timing_metrics['snr_during_switch']) if switch_timing_metrics['snr_during_switch'] else 0
                            }
                        }

                        beam_mgr_metrics = beam_manager.get_performance_metrics()
                        performance_metrics['snr_history'] = beam_mgr_metrics['snr_history']
                        performance_metrics['beam_switches'] = beam_mgr_metrics['switch_times']
                        performance_metrics['packet_stats'] = {
                            'total': switch_timing_metrics['total_packets'],
                            'successful': switch_timing_metrics['packet_success_count'],
                            'failed_during_switch': 0
                        }

                        save_performance_metrics(performance_metrics, os.path.join(result_dir, 'performance_metrics.h5'))
                        print(f"Performance metrics saved to: {result_dir}/performance_metrics.h5")
                        logger.info("Simulation completed successfully")
                        print(f"Total simulation time: {int((time.time() - start_time) // 60)}m {int((time.time() - start_time) % 60)}s")
                        print("Dataset generation completed. Run causal analysis separately if needed.")
                    break
                except BlockingIOError as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1}/{max_attempts}: Could not open HDF5 file due to lock (errno 11). Retrying in 1 second...")
                        time.sleep(1)
                    else:
                        logger.error(f"Failed to open HDF5 file after {max_attempts} attempts: {str(e)}")
                        raise
        logger.debug("Exiting MirroredStrategy scope - Checking strategy stack: {getattr(strategy, '_distribution_strategy_stack', [])}")
        logger.info("Strategy scope exited naturally")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        
        raise
    finally:
        logger.debug("Final cleanup - Checking strategy status: {tf.distribute.has_strategy()}")
        cleanup()
        logger.info("Simulation finished, cleaning up strategy scope")
if __name__ == "__main__":
    main()