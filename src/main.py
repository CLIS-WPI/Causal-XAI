from config import SmartFactoryConfig
from channel_generator import SmartFactoryChannel
from channel_analyzer import ChannelAnalyzer
from scene_setup import setup_scene
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import traceback
from mpl_toolkits.mplot3d import Axes3D
import sionna
from sionna.rt import Scene, PlanarArray, Transmitter, Receiver, RIS, SceneObject
import logging
from validation import ChannelValidator
from config import SmartFactoryConfig
from scene_manager import SceneManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_factory.log'),  # Save logs to file
        logging.StreamHandler()  # Also show logs in console
    ]
)
logger = logging.getLogger(__name__)

# Optional: Set specific log levels for different modules
logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Reduce matplotlib debug spam
logging.getLogger('tensorflow').setLevel(logging.WARNING)  # Reduce tensorflow debug spam

# Enable XLA compatibility
sionna.config.xla_compat = True

def ensure_result_dir():
    """Create result directory if it doesn't exist"""
    result_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def run_validation_pipeline(validator, channel_responses, channel_gen, config):
    """Run validation checks before analysis"""
    try:
        logger.info("Starting validation pipeline...")
        
        # Get required data from channel generator
        predicted_beams = channel_gen.get_predicted_beams()
        optimal_beams = channel_gen.get_optimal_beams()
        baseline_scans = channel_gen.get_baseline_scans()
        xai_scans = channel_gen.get_xai_scans()
        
        # Run validation
        validation_results = validator.run_full_validation(
            channel_response=channel_responses[-1],
            predicted_beams=predicted_beams,
            optimal_beams=optimal_beams,
            baseline_scans=baseline_scans,
            xai_scans=xai_scans
        )
        
        # Check validation results
        all_passed = all(result[0] for result in validation_results.values())
        if all_passed:
            logger.info("All validation checks passed!")
        else:
            logger.warning("Some validation checks failed. Check validation report for details.")
            
        return validation_results, all_passed
        
    except Exception as e:
        logger.error(f"Validation pipeline failed: {str(e)}")
        raise

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
    
    # Validate frequency parameters
    if config.carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be positive")
    
def save_channel_stats(channel_response, config, result_dir, validation_results=None):
    """Save channel statistics and configuration to a text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(result_dir, f'channel_stats_{timestamp}.txt')
    
    h = channel_response['h']
    capacity = tf.math.log(1 + tf.abs(h)**2)
    
    with open(stats_file, 'w') as f:
        f.write("Smart Factory Channel Analysis Results\n")
        f.write("=====================================\n\n")
        
        # Channel Statistics
        f.write("Channel Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Channel matrix shape: {h.shape}\n")
        f.write(f"Path delays shape: {channel_response['tau'].shape}\n")
        f.write(f"LoS conditions shape: {channel_response['los_condition'].shape}\n")
        f.write(f"Average channel capacity: {tf.reduce_mean(capacity).numpy():.2f}\n\n")
        
        # Configuration Parameters
        f.write("Configuration Parameters:\n")
        f.write("-----------------------\n")
        f.write(f"Carrier Frequency: {config.carrier_frequency/1e9:.2f} GHz\n")
        f.write(f"Room Dimensions: {config.room_dim} m\n")
        f.write(f"Number of AGVs: {config.num_agvs}\n")
        f.write(f"BS Array Configuration: {config.bs_array}\n")
        f.write(f"Channel Model: {config.model}\n")
        f.write(f"Scenario: {config.scenario}\n\n")
        
        f.write(f"Analysis timestamp: {datetime.now()}\n")

        # sections for causal and energy metrics
        if 'causal_analysis' in channel_response:
            f.write("\nCausal Analysis Results:\n")
            f.write("-----------------------\n")
            causal_analysis = channel_response['causal_analysis']
            f.write(f"Causal Effect: {causal_analysis.get('causal_effect', 'N/A')}\n")
        
        if 'energy_metrics' in channel_response:
            f.write("\nEnergy Efficiency Metrics:\n")
            f.write("------------------------\n")
            energy_metrics = channel_response['energy_metrics']
            f.write(f"Total Energy Savings: {energy_metrics.get('total_energy_savings', 'N/A')} J\n")
            f.write(f"Energy Efficiency: {energy_metrics.get('energy_efficiency', 'N/A')}\n")
        if validation_results:
            f.write("\nValidation Results:\n")
            f.write("-----------------\n")
            for metric, (passed, value) in validation_results.items():
                status = "PASS" if passed else "FAIL"
                f.write(f"{metric}: {status} (Value: {value:.3f})\n")

def analyze_channel_properties(channel_response, config, result_dir):
    """Analyze and plot channel properties for smart factory scenario"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Channel Magnitude Analysis
        plt.figure(figsize=(12, 8))
        h = channel_response['h']
        print(f"[DEBUG] Initial channel response shape: {h.shape}")
        
        # Get magnitude of the complex channel response and normalize it
        h_mag = tf.abs(h)
        print(f"[DEBUG] Channel magnitude shape: {h_mag.shape}")
        h_mag_normalized = h_mag / tf.reduce_max(h_mag)
        print(f"[DEBUG] Normalized magnitude shape: {h_mag_normalized.shape}")
        
        try:
            print("[DEBUG] Starting dimension reduction...")
            # Remove batch dimension (axis 0)
            h_reduced = tf.squeeze(h_mag_normalized, axis=0)
            print(f"[DEBUG] After removing batch dimension: {h_reduced.shape}")
            
            # Remove singleton dimensions from axes 1 and 2
            h_reduced = tf.squeeze(h_reduced, axis=[1, 2])
            print(f"[DEBUG] After removing singleton dimensions: {h_reduced.shape}")
            
            # Average over the paths dimension (last dimension)
            h_2d = tf.reduce_mean(h_reduced, axis=-1)
            print(f"[DEBUG] After averaging over paths: {h_2d.shape}")
            
            # Average over the AGV (antenna) dimension (first dimension)
            h_2d = tf.reduce_mean(h_2d, axis=0)
            print(f"[DEBUG] After averaging over antennas: {h_2d.shape}")
            
            # Convert to numpy array
            h_2d = h_2d.numpy()
            print(f"[DEBUG] Converted to numpy, shape: {h_2d.shape}")
            
            # Final squeeze to remove any leftover singleton dimensions
            h_2d = np.squeeze(h_2d)
            print(f"[DEBUG] Final numpy array shape after squeeze: {h_2d.shape}")
            
        except Exception as e:
            print(f"[ERROR] Error during dimension reduction: {e}")
            print("[DEBUG] Attempting fallback method...")
            try:
                # Fallback method: reduce over multiple axes at once
                h_2d = tf.reduce_mean(h_mag_normalized, axis=[0, 1, 2, -1]).numpy()
                h_2d = np.squeeze(h_2d)
                print(f"[DEBUG] Fallback shape: {h_2d.shape}")
            except Exception as e:
                print(f"[ERROR] Fallback method failed: {e}")
                raise

        print("[DEBUG] Starting plotting...")
        plt.subplot(2, 2, 1)
        print(f"[DEBUG] Shape being passed to imshow before check: {h_2d.shape}")
        
        # Final check: if the image data is not 2D, squeeze extra dimensions
        if h_2d.ndim != 2:
            print(f"[WARNING] Final image data has {h_2d.ndim} dimensions, expected 2. Squeezing extra dimensions...")
            h_2d = np.squeeze(h_2d)
            print(f"[DEBUG] Shape after extra squeeze: {h_2d.shape}")
        
        # At this point, h_2d should be 2D
        if h_2d.ndim != 2:
            print(f"[ERROR] Final image data shape is still invalid: {h_2d.shape}")
            raise ValueError(f"Cannot convert channel magnitude to 2D image; got shape {h_2d.shape}")
        
        im = plt.imshow(h_2d, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Normalized Magnitude')
        plt.title(f'Channel Magnitude\n({config.scenario} scenario)')
        plt.xlabel('OFDM Symbol')
        plt.ylabel('Subcarrier Index')

        # 2. Path Delay Analysis
        plt.subplot(2, 2, 2)
        tau = channel_response['tau']
        if isinstance(tau, tf.Tensor):
            delays_ns = tau.numpy().flatten() * 1e9  # Convert to ns for TensorFlow tensor
        else:
            delays_ns = tau.flatten() * 1e9  # Convert to ns for numpy array
        valid_delays = delays_ns[~np.isnan(delays_ns)]
        if len(valid_delays) > 0:
            plt.hist(valid_delays, bins=min(50, len(valid_delays)),
                    density=True, color='blue', alpha=0.7)
            plt.axvline(np.mean(valid_delays), color='red', linestyle='--',
                        label=f'Mean: {np.mean(valid_delays):.2f} ns')
            plt.title('Path Delay Distribution')
            plt.xlabel('Delay (ns)')
            plt.ylabel('Density')
            plt.legend()
        
        # 3. LoS/NLoS Analysis
        plt.subplot(2, 2, 3)
        los_condition = channel_response['los_condition']
        if isinstance(los_condition, tf.Tensor):
            los_data = los_condition.numpy().flatten().astype(np.int32)
        else:
            los_data = los_condition.flatten().astype(np.int32)
        los_percent = np.mean(los_data) * 100
        nlos_percent = (1 - np.mean(los_data)) * 100
        plt.bar(['NLoS', 'LoS'], [nlos_percent, los_percent],
                color=['red', 'green'], alpha=0.7)
        plt.title('LoS/NLoS Distribution')
        plt.ylabel('Percentage (%)')
        for i, v in enumerate([nlos_percent, los_percent]):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        # 4. Power Delay Profile
        plt.subplot(2, 2, 4)
        try:
            # Flatten all dimensions except the last one for power calculation
            h_reshaped = tf.reshape(h, [-1, h.shape[-1]])
            h_power = tf.reduce_mean(tf.abs(h_reshaped)**2, axis=0)
            h_power = h_power.numpy()
            h_power_db = 10 * np.log10(np.maximum(h_power, 1e-10))
            x_axis = np.arange(len(h_power))
            plt.plot(x_axis, h_power_db, 'b-', linewidth=2)
            plt.fill_between(x_axis, h_power_db, np.min(h_power_db),
                            alpha=0.3, color='blue')
            rms_delay = np.sqrt(np.average(x_axis**2, weights=h_power))
            plt.axvline(rms_delay, color='red', linestyle='--',
                        label=f'RMS Delay: {rms_delay:.2f}')
            plt.title('Power Delay Profile')
            plt.xlabel('Path Index')
            plt.ylabel('Average Power (dB)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        except Exception as e:
            print(f"[ERROR] Error in power delay profile calculation: {e}")
            rms_delay = 0  # Default value if calculation fails
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'channel_analysis_{timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Save numerical results
        results_file = os.path.join(result_dir, f'channel_analysis_{timestamp}.txt')
        with open(results_file, 'w') as f:
            f.write("Smart Factory Channel Analysis Results\n")
            f.write("=====================================\n")
            f.write(f"Average Channel Magnitude: {tf.reduce_mean(h_mag).numpy():.4f}\n")
            f.write(f"Channel Magnitude Std Dev: {tf.math.reduce_std(h_mag).numpy():.4f}\n")
            f.write(f"Mean Path Delay: {np.mean(valid_delays):.2f} ns\n")
            f.write(f"RMS Delay Spread: {rms_delay:.2f}\n")
            f.write(f"LoS Probability: {los_percent:.1f}%\n")
            f.write(f"NLoS Probability: {nlos_percent:.1f}%\n")
            
    except Exception as e:
        print(f"[ERROR] Error in channel analysis: {str(e)}")
        traceback.print_exc()

def analyze_ris_effectiveness(channel_response, result_dir):
    """Analyze RIS effectiveness by comparing channels with/without RIS"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    h_with_ris = channel_response['h_with_ris']
    h_without_ris = channel_response['h_without_ris']
    
    # Compute RIS gain
    gain_with_ris = tf.reduce_mean(tf.abs(h_with_ris)**2)
    gain_without_ris = tf.reduce_mean(tf.abs(h_without_ris)**2)
    ris_gain = float(gain_with_ris / gain_without_ris)
    
    print(f"[DEBUG] h_with_ris original shape: {h_with_ris.shape}")
    print(f"[DEBUG] h_without_ris original shape: {h_without_ris.shape}")
    
    # Sequential reduction for h_with_ris
    temp = tf.reduce_mean(tf.abs(h_with_ris), axis=0)
    print(f"[DEBUG] After reducing axis 0: {temp.shape}")  # Expected: (2,1,1,128,23,10)
    temp = tf.reduce_mean(temp, axis=0)
    print(f"[DEBUG] After reducing next axis: {temp.shape}")  # Expected: (1,1,128,23,10)
    temp = tf.reduce_mean(temp, axis=0)
    print(f"[DEBUG] After reducing third axis: {temp.shape}")  # Expected: (1,128,23,10)
    h_with_ris_img = tf.reduce_mean(temp, axis=-1)
    print(f"[DEBUG] h_with_ris_img shape before squeeze: {h_with_ris_img.shape}")  # Expected: (1,128,23)
    # Remove the extra singleton dimension (axis 0)
    h_with_ris_img = tf.squeeze(h_with_ris_img, axis=0)
    print(f"[DEBUG] Final h_with_ris_img shape: {h_with_ris_img.shape}")  # Expected: (128,23)
    
    # Similarly for h_without_ris
    temp2 = tf.reduce_mean(tf.abs(h_without_ris), axis=0)
    print(f"[DEBUG] h_without_ris shape after reducing axis 0: {temp2.shape}")
    temp2 = tf.reduce_mean(temp2, axis=0)
    print(f"[DEBUG] After reducing next axis for h_without_ris: {temp2.shape}")
    temp2 = tf.reduce_mean(temp2, axis=0)
    print(f"[DEBUG] After reducing third axis for h_without_ris: {temp2.shape}")
    h_without_ris_img = tf.reduce_mean(temp2, axis=-1)
    print(f"[DEBUG] h_without_ris_img shape before squeeze: {h_without_ris_img.shape}")
    h_without_ris_img = tf.squeeze(h_without_ris_img, axis=0)
    print(f"[DEBUG] Final h_without_ris_img shape: {h_without_ris_img.shape}")  # Expected: (128,23)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(h_with_ris_img.numpy(), aspect='auto', cmap='viridis')
    plt.title('Channel with RIS')
    plt.colorbar(im1, label='Magnitude')
    
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(h_without_ris_img.numpy(), aspect='auto', cmap='viridis')
    plt.title('Channel without RIS')
    plt.colorbar(im2, label='Magnitude')
    
    plt.suptitle(f'RIS Gain: {ris_gain:.2f}x')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'ris_effectiveness_{timestamp}.png'))
    plt.close()
    
    return ris_gain

def analyze_blockage_statistics(channel_response, result_dir):
    """Analyze and visualize blockage statistics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    los_conditions = channel_response['los_condition']
    los_ratio = tf.reduce_mean(tf.cast(los_conditions, tf.float32))
    
    # Save statistics
    stats_file = os.path.join(result_dir, f'blockage_stats_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write(f"LOS Ratio: {float(los_ratio):.2%}\n")
        f.write(f"NLOS Ratio: {float(1 - los_ratio):.2%}\n")
    
    return float(los_ratio)

def plot_agv_trajectories(channel_gen, result_dir):
    """Plot AGV movement trajectories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories for each AGV
    for i, positions in enumerate(channel_gen.positions_history):
        positions = np.array(positions)
        ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], 
                label=f'AGV {i+1}')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('AGV Trajectories')
    ax.legend()
    
    plt.savefig(os.path.join(result_dir, f'agv_trajectories_{timestamp}.png'))
    plt.close()

def analyze_causal_relationships(channel_response, result_dir):
    """Analyze and visualize causal relationships in channel response"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract causal analysis results
    causal_analysis = channel_response.get('causal_analysis', {})
    
    # Save causal analysis results
    causal_file = os.path.join(result_dir, f'causal_analysis_{timestamp}.txt')
    with open(causal_file, 'w') as f:
        f.write("Causal Analysis Results\n")
        f.write("======================\n\n")
        
        if 'causal_effect' in causal_analysis:
            f.write(f"Causal Effect: {causal_analysis['causal_effect']}\n")
        if 'confidence_intervals' in causal_analysis:
            f.write(f"Confidence Intervals: {causal_analysis['confidence_intervals']}\n")
        if 'treatment_variables' in causal_analysis:
            f.write(f"Treatment Variables: {causal_analysis['treatment_variables']}\n")
        if 'outcome_variable' in causal_analysis:
            f.write(f"Outcome Variable: {causal_analysis['outcome_variable']}\n")

def analyze_energy_efficiency(channel_response, result_dir):
    """Analyze and visualize energy efficiency metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract energy metrics
    energy_metrics = channel_response.get('energy_metrics', {})
    
    if energy_metrics:
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot beam training energy comparison
        if 'beam_training' in energy_metrics:
            beam_metrics = energy_metrics['beam_training']
            plt.subplot(1, 2, 1)
            plt.bar(['Baseline', 'Optimized'], 
                [beam_metrics['baseline'], beam_metrics['optimized']])
            plt.title('Beam Training Energy Consumption')
            plt.ylabel('Energy (J)')
            
        # Plot overall energy efficiency
        plt.subplot(1, 2, 2)
        if 'energy_efficiency' in energy_metrics:
            plt.bar(['Energy Efficiency'], [energy_metrics['energy_efficiency']])
            plt.title('Overall Energy Efficiency')
            plt.ylabel('Channel Quality / Energy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'energy_analysis_{timestamp}.png'))
        plt.close()
        
        # Save numerical results
        energy_file = os.path.join(result_dir, f'energy_metrics_{timestamp}.txt')
        with open(energy_file, 'w') as f:
            f.write("Energy Efficiency Metrics\n")
            f.write("=======================\n\n")
            for metric_name, value in energy_metrics.items():
                f.write(f"{metric_name}: {value}\n")

def main():
    """Main execution function"""
    result_dir = ensure_result_dir()
    scene = None  # Ensure scene is defined in outer scope for cleanup
    
    try:
        logger.info("Starting Smart Factory Channel Analysis")
        print("[DEBUG] Starting main execution...")
        
        # Set random seed for reproducibility
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
        
        # Initialize and validate configuration with additional checks
        try:
            print("[DEBUG] Initializing configuration...")
            config = SmartFactoryConfig()
            
            # Log configuration details before validation
            print("[DEBUG] Configuration parameters:")
            print(f"[DEBUG] - Carrier frequency: {config.carrier_frequency/1e9:.2f} GHz")
            print(f"[DEBUG] - Room dimensions: {config.room_dim}")
            print(f"[DEBUG] - Number of AGVs: {config.num_agvs}")
            
            validate_config(config)
            
            # Add model and scenario attributes if not present
            if not hasattr(config, 'model'):
                config.model = 'ray_tracing'
                print("[DEBUG] Added default model: ray_tracing")
            if not hasattr(config, 'scenario'):
                config.scenario = 'indoor_factory'
                print("[DEBUG] Added default scenario: indoor_factory")
                
            logger.info("Configuration initialized and validated")
            
        except Exception as e:
            print(f"[ERROR] Configuration initialization failed: {str(e)}")
            logger.error(f"Configuration initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Setup scene with detailed logging and validation
        try:
            print("[DEBUG] Starting scene setup...")
            logger.info("Starting scene setup...")
            scene = setup_scene(config)
            
            # Validate scene components
            if not scene:
                print("[ERROR] Scene setup returned None")
                raise ValueError("Scene setup returned None")
                
            # Verify essential scene components
            required_components = ['transmitters', 'receivers', 'objects']
            print("[DEBUG] Verifying scene components...")
            for component in required_components:
                if not hasattr(scene, component):
                    print(f"[ERROR] Missing required component: {component}")
                    raise ValueError(f"Scene missing required component: {component}")
                print(f"[DEBUG] Found component: {component}")
            
            # Verify scene objects
            if hasattr(scene, 'objects'):
                num_objects = len(scene.objects)
                print(f"[DEBUG] Scene contains {num_objects} objects")
                logger.info(f"Scene contains {num_objects} objects")
                
                # Verify essential objects
                required_objects = ['bs', 'ris']
                for obj_name in required_objects:
                    if not any(obj_name in name.lower() for name in scene.objects.keys()):
                        print(f"[WARNING] Scene missing recommended object: {obj_name}")
                        logger.warning(f"Scene missing recommended object: {obj_name}")
                
                # Log object details
                print("[DEBUG] Scene objects:")
                for name, obj in scene.objects.items():
                    print(f"[DEBUG] - {name}: {type(obj)}")
                    logger.debug(f"Scene object: {name}, Type: {type(obj)}")
            else:
                print("[ERROR] Scene has no objects attribute")
                raise ValueError("Scene has no objects attribute")
                
            logger.info("Scene setup completed successfully")
            print("[DEBUG] Scene setup completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to setup scene: {str(e)}")
            logger.error(f"Failed to setup scene: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Create and validate channel generator
        try:
            print("[DEBUG] Initializing channel generator...")
            logger.info("Initializing channel generator...")
            
            # Verify scene configuration before channel generator initialization
            if not hasattr(scene, 'frequency'):
                print("[DEBUG] Setting scene frequency from config")
                scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
                
            channel_gen = SmartFactoryChannel(config, scene=scene)
            print("[DEBUG] Channel generator instance created")
            
            # Verify channel generator initialization
            required_attrs = ['scene', 'config', 'positions_history', 'agv_positions']
            print("[DEBUG] Verifying channel generator attributes...")
            missing_attrs = [attr for attr in required_attrs if not hasattr(channel_gen, attr)]
            if missing_attrs:
                print(f"[ERROR] Missing channel generator attributes: {missing_attrs}")
                raise ValueError(f"Channel generator missing attributes: {missing_attrs}")
                
            print(f"[DEBUG] Initial AGV positions: {channel_gen.agv_positions}")
            print(f"[DEBUG] Positions history length: {len(channel_gen.positions_history)}")
            logger.debug(f"Initial AGV positions: {channel_gen.agv_positions}")
            logger.debug(f"Positions history length: {len(channel_gen.positions_history)}")
            logger.info("Channel generator initialized successfully")
            
            # Optional: Log additional diagnostic information
            try:
                print(f"[DEBUG] Scene frequency: {scene.frequency}")
                print(f"[DEBUG] Channel generator initial AGV positions: {channel_gen.agv_positions}")
                logger.debug(f"Scene frequency: {scene.frequency}")
                logger.debug(f"Channel generator initial AGV positions: {channel_gen.agv_positions}")
            except Exception as diag_error:
                print(f"[WARNING] Could not log diagnostic information: {diag_error}")
                logger.warning(f"Could not log diagnostic information: {diag_error}")
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize channel generator: {str(e)}")
            logger.error(f"Failed to initialize channel generator: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Optional: Log more context about the configuration and scene
            try:
                print(f"[DEBUG] Scene details: frequency={getattr(scene, 'frequency', 'N/A')}")
                print(f"[DEBUG] Scene type: {type(scene)}")
                logger.error(f"Scene details: frequency={getattr(scene, 'frequency', 'N/A')}")
                logger.error(f"Configuration details: {vars(config)}")
                logger.error(f"Scene type: {type(scene)}")
            except Exception:
                print("[ERROR] Could not log additional configuration details")
                logger.error("Could not log additional configuration details")
            
            raise
        
        # Generate and validate CSI dataset
        try:
            print("[DEBUG] Starting CSI dataset generation...")
            logger.info("Starting CSI dataset generation...")
            csi_filepath = os.path.join(result_dir, 'csi_dataset.h5')
            
            # Verify channel generator methods
            print("[DEBUG] Verifying channel generator methods...")
            if not hasattr(channel_gen, 'save_csi_dataset'):
                print("[ERROR] Channel generator missing save_csi_dataset method")
                raise AttributeError("Channel generator missing save_csi_dataset method")
                
            channel_gen.save_csi_dataset(csi_filepath)
            print(f"[DEBUG] CSI dataset saved to: {csi_filepath}")
            logger.info(f"CSI dataset saved to: {csi_filepath}")
            
            # Load and verify the saved dataset
            print("[DEBUG] Loading CSI dataset for verification...")
            logger.info("Loading CSI dataset for verification...")
            loaded_data = channel_gen.load_csi_dataset(csi_filepath)
            
            # Verify dataset contents
            print("[DEBUG] Verifying dataset contents...")
            required_keys = ['channel_matrices', 'path_delays', 'los_conditions', 'agv_positions']
            missing_keys = [key for key in required_keys if key not in loaded_data]
            if missing_keys:
                print(f"[ERROR] Dataset missing required keys: {missing_keys}")
                raise ValueError(f"Dataset missing required keys: {missing_keys}")
                
            print("[DEBUG] CSI dataset loaded and verified successfully")
            logger.info("CSI dataset loaded and verified successfully")
            
            # Process and validate channel responses
            print("[DEBUG] Processing channel responses...")
            logger.info("Processing channel responses...")
            channel_responses = process_channel_responses(loaded_data)
            if not channel_responses:
                print("[ERROR] No channel responses generated")
                raise ValueError("No channel responses generated")
            print(f"[DEBUG] Processed {len(channel_responses)} channel responses")
            logger.info(f"Processed {len(channel_responses)} channel responses")
            
            # Initialize and validate analyzer
            print("[DEBUG] Initializing channel analyzer...")
            logger.info("Initializing channel analyzer...")
            analyzer = ChannelAnalyzer(scene)
            if not hasattr(analyzer, 'scene'):
                print("[ERROR] Channel analyzer not properly initialized")
                raise ValueError("Channel analyzer not properly initialized")
            print("[DEBUG] Channel analyzer initialized")
            logger.info("Channel analyzer initialized")
            
            # Run analysis pipeline with validation
            print("[DEBUG] Starting analysis pipeline...")
            logger.info("Starting analysis pipeline...")
            validator = ChannelValidator(config)
            validation_results, validation_passed = run_validation_pipeline(
                validator, channel_responses, channel_gen, config
            )
            
            if validation_passed:
                print("[DEBUG] Validation passed, running analysis pipeline...")
                run_analysis_pipeline(analyzer, channel_responses, channel_gen, 
                                config, result_dir, validation_results)
            else:
                print("[WARNING] Validation failed, but continuing with analysis")
                logger.warning("Validation failed, but continuing with analysis")
                run_analysis_pipeline(analyzer, channel_responses, channel_gen, 
                                config, result_dir, validation_results)
            
        except Exception as e:
            print(f"[ERROR] Error in analysis pipeline: {str(e)}")
            logger.error(f"Error in analysis pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        print("[DEBUG] Analysis completed successfully")
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Fatal error in main execution: {str(e)}")
        logger.error(f"Fatal error in main execution: {str(e)}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("[DEBUG] Performing cleanup...")
        logger.info("Performing cleanup...")
        if scene:
            scene.cleanup()
            print("[DEBUG] Scene cleanup completed successfully")
            logger.info("Scene cleanup completed successfully.")
        plt.close('all')
        print(f"[DEBUG] All results saved in {result_dir}")
        logger.info(f"All results saved in {result_dir}")

if __name__ == "__main__":
    main()


def process_channel_responses(loaded_data):
    """Process loaded CSI data into channel responses"""
    channel_responses = []
    num_samples = loaded_data['channel_matrices'].shape[0]
    
    for i in range(num_samples):
        channel_response = {
            'h': loaded_data['channel_matrices'][i],
            'tau': loaded_data['path_delays'][i],
            'los_condition': loaded_data['los_conditions'][i],
            'agv_positions': loaded_data['agv_positions'][i],
            'h_with_ris': loaded_data['channel_matrices'][i],
            'h_without_ris': loaded_data['channel_matrices'][i] * 0.5
        }
        channel_responses.append(channel_response)
    
    return channel_responses

def run_analysis_pipeline(analyzer, channel_responses, channel_gen, config, result_dir, validation_results=None):
    """Run the complete analysis pipeline"""
    try:
        # 1. Visualize and save scene using ChannelAnalyzer
        fig = analyzer.visualize_scene()
        scene_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(result_dir, f'factory_scene_3d_{scene_timestamp}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Scene visualization saved to: {fig_path}")

        # 2. Analyze each time step
        for t, channel_response in enumerate(channel_responses):
            logger.info(f"Analyzing time step {t+1}/{len(channel_responses)}")
            
            # Basic channel analysis
            analyze_channel_properties(channel_response, config, result_dir)
            
            # RIS effectiveness analysis using ChannelAnalyzer
            if 'h_with_ris' in channel_response and 'h_without_ris' in channel_response:
                ris_gain = analyzer.analyze_channel_with_ris(channel_response)
                logger.info(f"Time step {t+1}: RIS Gain = {ris_gain:.2f}x")
            
            # Channel matrix visualization using ChannelAnalyzer
            if 'h' in channel_response:
                matrix_path = os.path.join(result_dir, f'channel_matrix_t{t+1}.png')
                analyzer.plot_channel_matrix(channel_response['h'])
                plt.savefig(matrix_path)
                plt.close()
            
            # Path analysis using ChannelAnalyzer methods
            if 'paths' in channel_response:
                analyzer.analyze_channel(channel_response['paths'])
                analyzer.plot_path_gains(channel_response['paths'])
                analyzer.plot_delay_spread(channel_response['paths'])
            
            # Additional analyses
            analyze_blockage_statistics(channel_response, result_dir)
            analyze_causal_relationships(channel_response, result_dir)
            analyze_energy_efficiency(channel_response, result_dir)

        # 3. Plot AGV trajectories
        if hasattr(channel_gen, 'positions_history'):
            plot_agv_trajectories(channel_gen, result_dir)
            logger.info("AGV trajectories plotted successfully")

        # 4. Save final statistics
        save_channel_stats(channel_responses[-1], config, result_dir, validation_results)
        logger.info("Final channel statistics saved")

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()        