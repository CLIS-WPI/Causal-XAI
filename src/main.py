# main.py
# System imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mitsuba setup
# Keep unchanged - external dependencies
from utils import ensure_mitsuba_variant
import mitsuba
import time
import tensorflow as tf
import numpy as np
import logging
from scene_setup import setup_scene
from beam_manager import BeamManager
from agv_path_manager import AGVPathManager
from scipy.special import erfc
import gc

# Sionna core imports
from sionna.constants import SPEED_OF_LIGHT
from sionna.phy.channel.utils import cir_to_ofdm_channel, subcarrier_frequencies

# Sionna RT imports
from sionna.rt import Scene 
from sionna.rt.components import Transmitter, Receiver, PathSolver
from sionna.rt.antenna import PlanarArray, DiscretePhaseProfile
from sionna.rt.materials import RadioMaterial
from sionna.rt.paths import Paths
from sionna.rt.grid import CellGrid

logger = logging.getLogger(__name__)

# Local imports
from config import SmartFactoryConfig
from scene_setup import setup_scene, verify_los_paths
from scene_manager import SceneManager
from sionna_ply_generator import SionnaPLYGenerator
from beam_manager import BeamManager
from channel_generator import SmartFactoryChannel
from agv_path_manager import AGVPathManager
from data_store import save_performance_metrics
# Environment settings (unchanged)
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus) if gpus else 0
    
    if num_gpus == 0:
        print("No GPUs found. Falling back to CPU.")
        logger.warning("No GPUs detected, using CPU")
    else:
        # Restrict to GPU:0 only
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')  # Use only GPU:0
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"Configured TensorFlow to use only GPU:0 ({gpus[0].name})")
            print(f"Configured to use GPU:0 ({gpus[0].name})")
            
            # Optional optimizations
            tf.config.experimental.enable_tensor_float_32_execution(True)
            tf.config.optimizer.set_jit(True)
            tf.config.optimizer.set_experimental_options({
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True,
                'arithmetic_optimization': True,
                'dependency_optimization': True,
                'loop_optimization': True,
            })
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            tf.keras.backend.clear_session()
            gc.collect()
        except RuntimeError as e:
            print(f"Error configuring GPU:0: {e}")
            logger.error(f"Failed to configure GPU:0: {str(e)}")
    
    return 1 if num_gpus > 0 else 0  # Return 1 if GPU:0 is available, 0 if CPU only

def create_optimized_dataset(positions, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(positions)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    return dataset

class MemoryManager:
    @staticmethod
    def clear_memory():
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.reset_memory_stats(device)
        except:
            pass
    
    @staticmethod
    def monitor_memory(step):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        gpu_info = []
        try:
            import subprocess
            gpu_stats = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
            gpu_stats = gpu_stats.decode("utf-8").strip().split('\n')
            for i, stats in enumerate(gpu_stats):
                util, mem_used, mem_total = stats.split(', ')
                gpu_info.append(f"GPU {i}: Utilization {util}%, Memory Used {mem_used}MB / {mem_total}MB")
        except:
            gpu_info = ["GPU stats unavailable"]
        logger.info(f"Step {step} - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        for info in gpu_info:
            logger.info(f"Step {step} - {info}")

class PerformanceMonitor:
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.step_start = None
        
    def start_step(self):
        self.step_start = time.time()
        
    def end_step(self):
        if self.step_start is not None:
            step_time = time.time() - self.step_start
            self.step_times.append(step_time)
            try:
                memory_info = {}
                for i, device in enumerate(tf.config.list_physical_devices('GPU')):
                    memory_info[f'GPU:{i}'] = tf.config.experimental.get_memory_info(f'GPU:{i}')
                self.memory_usage.append(memory_info)
            except:
                pass
        
    def get_statistics(self):
        if not self.step_times:
            return {'avg_step_time': 0, 'max_memory_usage': 0, 'total_steps': 0}
        stats = {
            'avg_step_time': np.mean(self.step_times),
            'total_steps': len(self.step_times)
        }
        if self.memory_usage and any(self.memory_usage):
            peak_values = []
            for mem_entry in self.memory_usage:
                for device, info in mem_entry.items():
                    if 'peak' in info:
                        peak_values.append(info['peak'])
            if peak_values:
                stats['max_memory_usage'] = max(peak_values) / 1024**2  # MB
        return stats

def process_batch_optimized(batch, scene_manager, channel_generator, config, agv_manager, obstacle_positions):
    agv_positions = batch
    # Update AGV positions using SceneManager
    scene_manager.update_scene_with_agv_positions(agv_positions)
    channel_data = channel_generator.generate_channel_data(agv_positions)
    return channel_data

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

def convert_to_numpy(data, strategy):
    if isinstance(data, dict):
        return {k: convert_to_numpy(v, strategy) for k, v in data.items()}
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
    logger.debug("Starting cleanup process")
    tf.keras.backend.clear_session()
    gc.collect()
    logger.debug("Cleanup process completed")

def safe_to_numpy(data):
    return data.numpy() if isinstance(data, tf.Tensor) else data

def get_batch_shape(batch):
    return batch.shape

def main():
    num_gpus = configure_gpus()
    logger = setup_logging()
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
        handler.flush = lambda: sys.stdout.flush()
    logger.info("Starting smart factory beam switching simulation...")
    logger.info(f"Sionna version: {sionna.__version__}")  # Added version logging
    
    perf_monitor = PerformanceMonitor()
    
    try:
        print("Starting simulation...")
        logger.info("Starting smart factory beam switching simulation...")
        
        result_dir = ensure_result_dir()
        logger.info(f"Results will be saved to: {result_dir}")
        
        tf.random.set_seed(42)
        config = SmartFactoryConfig()
        validate_config(config)
        logger.info("Configuration initialized successfully")
        logger.info(f"Total time steps: {config.num_time_steps}, num_samples: {config.ray_tracing['num_samples']}, max_depth: {config.ray_tracing['max_depth']}")
        
        # Generate PLY files
        print("Generating PLY files...")
        logger.info("Generating PLY files...")
        meshes_dir = os.path.join(os.path.dirname(__file__), "meshes")
        SionnaPLYGenerator.generate_factory_geometries(config, meshes_dir)
        logger.info("PLY files generated successfully")

        # Setup scene
        scene = setup_scene(config)
        print("Current Mitsuba variant:", mitsuba.variant())
        if not scene:
            raise ValueError("Scene setup failed")
        logger.info("Scene setup completed")

        # Initialize SceneManager
        scene_manager = SceneManager(scene, config)
        logger.info("SceneManager initialized")

        # Define obstacle_positions from config
        obstacle_positions = tf.constant(config.scene_objects['shelf_positions'], dtype=tf.float32)
        logger.info(f"Obstacle positions defined from config: {obstacle_positions.numpy()}")

        # Use default strategy for single GPU (GPU:0)
        strategy = tf.distribute.get_strategy()
        logger.info("Using default strategy (single GPU:0)")
        print(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")
        
        print("Generating AGV paths...")
        logger.info("Generating AGV paths...")
        agv_manager = AGVPathManager(config, scene)
        logger.info("AGV paths generated")

        print("Creating optimized dataset...")
        logger.info("Creating optimized dataset...")
        batch_size = 32
        positions = np.stack([agv_manager.paths[i] for i in range(config.num_agvs)], axis=1, dtype=np.float32)
        dataset = create_optimized_dataset(positions, batch_size=batch_size)
        logger.info(f"Dataset created successfully: batch_size={batch_size}")

        dataset_file = os.path.join(result_dir, 'simulation_data.h5')
        performance_metrics = {'beam_switches': [], 'ber_history': [], 'snr_history': []}
        switch_timing_metrics = {
            'switch_start_time': None, 'switch_durations': [],
            'packet_success_count': 0, 'total_packets': 0,
            'ber_during_switch': [], 'snr_during_switch': []
        }
        
        # Initialize objects
        beam_manager = BeamManager(config)
        channel_generator = SmartFactoryChannel(config, scene)
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        logger.info("Scene frequency set")

        print(f"Opening HDF5 file: {dataset_file}")
        logger.info(f"Opening HDF5 file for writing: {dataset_file}")
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

            profiler_running = False
            try:
                tf.profiler.experimental.start('logdir')
                profiler_running = True
                global_step = 0
                for batch_idx, batch in enumerate(dataset):
                    for t in range(min(batch.shape[0], total_steps - global_step)):
                        if global_step >= total_steps:
                            break

                        perf_monitor.start_step()
                        progress = (global_step + 1) / total_steps * 100
                        elapsed_time = time.time() - start_time
                        eta = (elapsed_time / (global_step + 1)) * (total_steps - (global_step + 1)) if global_step > 0 else 0
                        eta_str = f"{int(eta // 60)}m {int(eta % 60)}s" if eta > 0 else "Calculating..."
                        print(f"\rSimulating step {global_step+1}/{total_steps} ({progress:.1f}%) - ETA: {eta_str}", end="")

                        if global_step % 5 == 0:
                            MemoryManager.clear_memory()
                            MemoryManager.monitor_memory(global_step)
                            logger.info(f"Progress: {progress:.1f}%, Step {global_step+1}/{total_steps}, ETA: {eta_str}")

                        logger.debug(f"Step {global_step}: Processing batch {batch_idx}, time step {t} - Shape: {get_batch_shape(batch)}")
                        
                        agv_positions = batch[t]  # Shape: (num_agvs, 3)
                        logger.debug(f"Adjusted agv_positions shape: {agv_positions.shape}")
                        channel_data = process_batch_optimized(batch[t], scene_manager, channel_generator, config, agv_manager, obstacle_positions)
                        logger.debug(f"Channel data keys: {channel_data.keys()}, h_freq shape: {channel_data['channel_matrices'].shape}")
                        if channel_data is None or 'paths' not in channel_data:
                            logger.error(f"Step {global_step}: Channel data is invalid, using fallback")
                            channel_data = {
                                'paths': None,
                                'channel_matrices': tf.complex(
                                    tf.random.normal([config.num_agvs, config.num_subcarriers], dtype=tf.float32),
                                    tf.random.normal([config.num_agvs, config.num_subcarriers], dtype=tf.float32)
                                ),
                                'path_delays': tf.zeros([1, config.num_agvs, 1], dtype=tf.float32),
                                'los_conditions': tf.zeros([config.num_agvs], dtype=tf.int32),
                                'agv_positions': agv_positions,
                                'path_losses': tf.zeros([config.num_agvs], dtype=tf.float32),
                                'beam_metrics': {'snr_db': tf.zeros([config.num_agvs], dtype=tf.float32)},
                                'path_data': {
                                    'path_powers': tf.zeros([1, config.num_agvs, 1]),
                                    'path_directions': tf.zeros([1, config.num_agvs, 1, 2])
                                }
                            }

                        with strategy.scope():
                            mitsuba.set_variant('cuda_ad_rgb')
                            logger.debug(f"Forced Mitsuba variant in strategy scope: {mitsuba.variant()}")

                            @tf.function
                            def process_replica(replica_batch, replica_channel_data):
                                if replica_channel_data['paths'] is None:
                                    return replica_channel_data
                                paths = replica_channel_data['paths']
                                snr_db = tf.random.uniform([config.num_agvs], minval=10, maxval=30, dtype=tf.float32)  # Placeholder
                                replica_channel_data['beam_metrics']['snr_db'] = snr_db
                                return replica_channel_data

                            channel_data = process_replica(agv_positions, channel_data)
                            logger.debug(f"Step {global_step}: Channel data processed")

                            optimal_beam = beam_manager.optimize_beam_direction(channel_data, agv_manager, obstacle_positions)
                            logger.debug(f"Optimal beam calculated: shape={optimal_beam.shape}, values={safe_to_numpy(optimal_beam)}")
                            logger.debug(f"Step {global_step}: Optimal beam determined")

                            for i in range(config.num_agvs):
                                agv_id = f'agv_{i}'
                                los_status = channel_data['los_conditions'][i]
                                position = channel_data['agv_positions'][i]
                                agv_manager.record_movement(agv_id, position, agv_manager.current_velocities[agv_id], los_status)

                            success = False
                            if 'beam_metrics' in channel_data and 'snr_db' in channel_data['beam_metrics']:
                                snr_per_agv = channel_data['beam_metrics']['snr_db']
                                mean_snr = np.mean(snr_per_agv)
                                success = mean_snr > config.beamforming['min_snr_threshold']
                                beam_manager.log_snr(mean_snr)
                                logger.debug(f"Step {global_step}: Mean SNR: {mean_snr}, Success: {success}")
                            else:
                                logger.warning(f"Step {global_step}: No SNR data")

                            beam_manager.update_beam(optimal_beam, success=success, channel_data=channel_data)

                            if beam_manager.has_switch_occurred() and beam_manager.switch_times:
                                switch_timing_metrics['switch_start_time'] = time.time()
                                if hasattr(channel_generator, 'calculate_ber'):
                                    current_ber = channel_generator.calculate_ber(channel_data)
                                    switch_timing_metrics['ber_during_switch'].append(current_ber)
                                switch_timing_metrics['snr_during_switch'].append(mean_snr)
                                if switch_timing_metrics['switch_start_time'] is not None:
                                    switch_duration = time.time() - switch_timing_metrics['switch_start_time']
                                    switch_timing_metrics['switch_durations'].append(switch_duration)
                                    switch_timing_metrics['switch_start_time'] = None

                            switch_timing_metrics['total_packets'] += 1
                            if success:
                                switch_timing_metrics['packet_success_count'] += 1

                            # Save to HDF5
                            step_group = h5f.create_group(f'step_{global_step}')
                            csi_group = step_group.create_group('csi_data')
                            mobility_group = step_group.create_group('mobility_data')
                            beam_group = step_group.create_group('beam_data')

                            csi_group.create_dataset('channel_matrices', data=safe_to_numpy(channel_data['channel_matrices']), compression='gzip')
                            csi_group.create_dataset('path_delays', data=safe_to_numpy(channel_data['path_delays']), compression='gzip')

                            mobility_group.create_dataset('agv_positions', data=safe_to_numpy(channel_data['agv_positions']), compression='gzip')
                            velocities = np.array([agv_manager.current_velocities[f'agv_{i}'] for i in range(config.num_agvs)])
                            mobility_group.create_dataset('velocities', data=velocities, compression='gzip')
                            mobility_group.create_dataset('los_conditions', data=safe_to_numpy(channel_data['los_conditions']), compression='gzip')
                            distance_to_bs = tf.norm(channel_data['agv_positions'] - tf.cast(config.bs_position, tf.float32), axis=1)
                            mobility_group.create_dataset('distance_to_bs', data=safe_to_numpy(distance_to_bs), compression='gzip')
                            blocked = safe_to_numpy(beam_manager.detect_blockage(channel_data, channel_data['agv_positions'], obstacle_positions))
                            mobility_group.create_dataset('blockage_status', data=blocked, compression='gzip')

                            bs_pos = tf.cast(config.bs_position, tf.float32)
                            agv_pos = tf.ensure_shape(channel_data['agv_positions'], [config.num_agvs, 3])
                            logger.debug(f"agv_positions shape after ensure: {agv_pos.shape}")
                            direction_vectors = agv_pos - bs_pos
                            logger.debug(f"direction_vectors shape: {direction_vectors.shape}")
                            horizontal_distances = tf.norm(direction_vectors[:, :2], axis=1)
                            elevations = tf.math.atan2(direction_vectors[:, 2], horizontal_distances) * 180.0 / np.pi
                            azimuths = tf.math.atan2(direction_vectors[:, 1], direction_vectors[:, 0]) * 180.0 / np.pi
                            azimuths = tf.where(azimuths < 0, azimuths + 360, azimuths)
                            elevations = tf.clip_by_value(elevations, -30, 30)
                            azimuths = tf.clip_by_value(azimuths, -config.beamforming['max_steering_angle'], config.beamforming['max_steering_angle'])
                            relative_angles = tf.stack([azimuths, elevations], axis=1).numpy()

                            beam_group.create_dataset('beam_directions', data=optimal_beam, compression='gzip')
                            beam_group.create_dataset('snr_db', data=safe_to_numpy(channel_data['beam_metrics']['snr_db']), compression='gzip')
                            beam_group.create_dataset('path_powers', data=safe_to_numpy(channel_data['path_data']['path_powers']), compression='gzip')
                            beam_group.create_dataset('path_directions', data=safe_to_numpy(channel_data['path_data']['path_directions']), compression='gzip')
                            beam_group.create_dataset('relative_angle', data=relative_angles, compression='gzip')
                            last_switch = beam_manager.switch_times[-1] if beam_manager.has_switch_occurred() and beam_manager.switch_times else {}
                            beam_group.attrs['switch_reason'] = last_switch.get('reason', 'None')

                            step_group.attrs['timestamp'] = global_step
                            step_group.attrs['success'] = success

                            if global_step % 2 == 0:
                                avg_snr = np.mean(safe_to_numpy(channel_data['beam_metrics']['snr_db']))
                                print(f"\nStep {global_step+1} metrics: Average SNR: {avg_snr:.2f} dB")

                        perf_monitor.end_step()
                        global_step += 1

                    if global_step >= total_steps:
                        break

                print("\nSimulation completed. Saving performance metrics...")
                logger.info("Simulation completed. Saving performance metrics...")
                
                perf_stats = perf_monitor.get_statistics()
                logger.info(f"Performance stats: {perf_stats}")
                
                performance_summary = {
                    'beam_switching': {
                        'average_switch_time': np.mean(switch_timing_metrics['switch_durations']) if switch_timing_metrics['switch_durations'] else 0,
                        'total_switches': len(switch_timing_metrics['switch_durations']),
                        'packet_success_rate': (switch_timing_metrics['packet_success_count'] / 
                                                switch_timing_metrics['total_packets']) if switch_timing_metrics['total_packets'] > 0 else 0,
                        'average_ber_during_switch': np.mean(switch_timing_metrics['ber_during_switch']) if switch_timing_metrics['ber_during_switch'] else 0,
                        'snr_variation_during_switch': np.std(switch_timing_metrics['snr_during_switch']) if switch_timing_metrics['snr_during_switch'] else 0
                    },
                    'performance_metrics': perf_stats
                }

                beam_mgr_metrics = beam_manager.get_performance_metrics()
                performance_metrics['snr_history'] = beam_mgr_metrics['snr_history']
                performance_metrics['beam_switches'] = beam_mgr_metrics['switch_times']
                performance_metrics['packet_stats'] = {
                    'total': switch_timing_metrics['total_packets'],
                    'successful': switch_timing_metrics['packet_success_count'],
                    'failed_during_switch': 0
                }
                performance_metrics['performance_summary'] = performance_summary

                save_performance_metrics(performance_metrics, os.path.join(result_dir, 'performance_metrics.h5'))
                print(f"Performance metrics saved to: {result_dir}/performance_metrics.h5")
                logger.info("Simulation completed successfully")
                print(f"Total simulation time: {int((time.time() - start_time) // 60)}m {int((time.time() - start_time) % 60)}s")
            
            except Exception as e:
                logger.error(f"Simulation loop failed: {str(e)}", exc_info=True)
                raise
            finally:
                if profiler_running:
                    try:
                        tf.profiler.experimental.stop()
                        profiler_running = False
                    except Exception as e:
                        logger.warning(f"Failed to stop profiler: {str(e)}")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        raise
    finally:
        cleanup_gpu_memory()
        logger.info("Simulation finished, cleaned up resources")

def cleanup_gpu_memory():
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            cleanup()
        except:
            pass
        sys.exit(1)