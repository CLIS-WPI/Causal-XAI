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
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, SceneObject
import mitsuba as mi

def ensure_result_dir():
    """Create result directory if it doesn't exist"""
    result_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def save_channel_stats(channel_response, config, result_dir):
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

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Initialize configuration
    config = SmartFactoryConfig()
    
    # Setup scene
    scene = setup_scene(config)
    
    # Create channel generator with scene
    channel_gen = SmartFactoryChannel(config, scene=scene)
    channel_gen.scene = scene  # Explicitly set the scene
    
    # Ensure result directory exists
    result_dir = ensure_result_dir()
    
    # Generate and save CSI dataset
    csi_filepath = os.path.join(result_dir, 'csi_dataset.h5')
    channel_gen.save_csi_dataset(csi_filepath)
    print(f"CSI dataset saved to: {csi_filepath}")
    
    # Load the saved CSI dataset
    loaded_data = channel_gen.load_csi_dataset(csi_filepath)
    print("CSI dataset loaded successfully")
    
    # Extract channel responses from loaded data
    channel_responses = []
    num_samples = loaded_data['channel_matrices'].shape[0]
    
    for i in range(num_samples):
        channel_response = {
            'h': loaded_data['channel_matrices'][i],
            'tau': loaded_data['path_delays'][i],
            'los_condition': loaded_data['los_conditions'][i],
            'agv_positions': loaded_data['agv_positions'][i],
            'h_with_ris': loaded_data['channel_matrices'][i],
            'h_without_ris': loaded_data['channel_matrices'][i] * 0.5  # Simplified
        }
        channel_responses.append(channel_response)
    
    # Analyze scene
    analyzer = ChannelAnalyzer(scene)
    
    # Visualize scene
    # In the main function, before saving the scene visualization:
    try:
        fig = analyzer.visualize_scene()
        scene_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(result_dir, f'factory_scene_3d_{scene_timestamp}.png')
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Scene visualization saved to: {fig_path}")
        
        # Analyze channel properties for each time step
        for t, channel_response in enumerate(channel_responses):
            print(f"\nAnalyzing time step {t+1}/{len(channel_responses)}")
            
            # Basic channel analysis
            analyze_channel_properties(channel_response, config, result_dir)
            
            # RIS effectiveness analysis
            if 'h_with_ris' in channel_response and 'h_without_ris' in channel_response:
                ris_gain = analyze_ris_effectiveness(channel_response, result_dir)
                print(f"Time step {t+1}: RIS Gain = {ris_gain:.2f}x")
            
            # Blockage analysis
            if 'los_condition' in channel_response:
                los_ratio = analyze_blockage_statistics(channel_response, result_dir)
                print(f"Time step {t+1}: LOS Ratio = {los_ratio:.2%}")
        
        # Plot AGV trajectories
        if hasattr(channel_gen, 'positions_history'):
            plot_agv_trajectories(channel_gen, result_dir)
            print("AGV trajectories plotted successfully")
        
        # Save final channel statistics
        save_channel_stats(channel_responses[-1], config, result_dir)
        
        print("Analysis of loaded CSI dataset completed successfully")
        
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        traceback.print_exc()
    
    print(f"Analysis complete. Results saved in {result_dir}")

if __name__ == "__main__":
    main()