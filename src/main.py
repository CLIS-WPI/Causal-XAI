from config import SmartFactoryConfig
from channel_generator import SmartFactoryChannel
from channel_analyzer import ChannelAnalyzer
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import traceback  # Add this import
from mpl_toolkits.mplot3d import Axes3D
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, SceneObject

def setup_scene(config):
    """Set up the smart factory simulation scene"""
    # Create empty scene
    scene = load_scene("__empty__")
    
    # Configure BS antenna array (16x4 UPA at 28 GHz)
    scene.tx_array = PlanarArray(
        num_rows=16,
        num_cols=4,
        vertical_spacing=0.5*3e8/28e9,  # Half wavelength at 28 GHz
        horizontal_spacing=0.5*3e8/28e9,
        pattern="tr38901",  # This is a valid pattern
        polarization="VH"   # Using dual polarization for base station
    )
    
    # Configure AGV antenna array (1x1)
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5*3e8/28e9,
        horizontal_spacing=0.5*3e8/28e9,
        pattern="iso",      # Changed from "omni" to "iso" for isotropic pattern
        polarization="V"    # Using vertical polarization for AGV
    )
    
    # Add base station
    tx = Transmitter(
        name="bs",
        position=[10.0, 0.5, 4.5],  # Ceiling mounted
        orientation=[0.0, 0.0, 0.0]
    )
    scene.add(tx)
    
    # Add RIS
    ris = RIS(
        name="ris",
        position=[10.0, 19.5, 2.5],  # North wall
        orientation=[0.0, 0.0, 0.0],
        num_rows=8,
        num_cols=8,
        element_spacing=0.5*3e8/28e9  # Half wavelength at 28 GHz
    )
    scene.add(ris)
    
    # Add metallic shelves with fixed positions
    shelf_positions = [
        [5.0, 5.0, 0.0],
        [15.0, 5.0, 0.0],
        [10.0, 10.0, 0.0],
        [5.0, 15.0, 0.0],
        [15.0, 15.0, 0.0]
    ]
    
    for i, position in enumerate(shelf_positions):
        shelf = SceneObject(
            name=f"shelf_{i}",
            position=position,
            size=[2.0, 1.0, 4.0],  # Length x Width x Height
            material="metal"
        )
        scene.add(shelf)
    
    # Add initial AGV positions
    initial_positions = [
        [12.0, 5.0, 0.5],   # AGV1
        [8.0, 15.0, 0.5]    # AGV2
    ]
    
    for i, pos in enumerate(initial_positions):
        rx = Receiver(
            name=f"agv_{i}",
            position=pos,
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(rx)
    
    return scene

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
        print(f"Channel response shape: {h.shape}")
        
        # Properly reshape and normalize the channel response
        h_mag = tf.abs(h)
        h_mag_normalized = h_mag / tf.reduce_max(h_mag)  # Normalize for better visualization
        
        try:
            # Reduce dimensions while preserving time and spatial information
            h_2d = tf.reduce_mean(h_mag_normalized, axis=[1, 2])  # Average over antennas
            h_2d = h_2d.numpy()
        except Exception as e:
            print(f"Error in channel magnitude calculation: {e}")
            h_2d = tf.reduce_mean(h_mag_normalized, axis=-1).numpy()
        
        plt.subplot(2, 2, 1)
        im = plt.imshow(h_2d, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Normalized Magnitude')
        plt.title(f'Channel Magnitude\n({config.scenario} scenario)')
        plt.xlabel('Time Steps')
        plt.ylabel('AGV Index')
        
        # 2. Path Delay Analysis
        plt.subplot(2, 2, 2)
        delays_ns = channel_response['tau'].numpy().flatten() * 1e9  # Convert to ns
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
        los_data = channel_response['los_condition'].numpy().flatten()
        los_data = los_data.astype(np.int32)
        
        # Calculate percentages
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
            # Calculate power delay profile
            h_power = tf.reduce_mean(tf.abs(h)**2, axis=[0,1,2])  # Average over batch and antennas
            h_power = h_power.numpy()
            h_power_db = 10 * np.log10(np.maximum(h_power, 1e-10))
            
            # Plot with enhanced visualization
            plt.plot(range(len(h_power)), h_power_db, 'b-', linewidth=2)
            plt.fill_between(range(len(h_power)), h_power_db, 
                        min(h_power_db), alpha=0.3, color='blue')
            plt.title('Power Delay Profile')
            plt.xlabel('Path Index')
            plt.ylabel('Average Power (dB)')
            plt.grid(True, alpha=0.3)
            
            # Add RMS delay spread
            rms_delay = np.sqrt(np.average((range(len(h_power)))**2, weights=h_power))
            plt.axvline(rms_delay, color='red', linestyle='--', 
                    label=f'RMS Delay: {rms_delay:.2f}')
            plt.legend()
            
        except Exception as e:
            print(f"Error in power delay profile calculation: {e}")
        
        # Adjust layout and save
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
        print(f"Error in channel analysis: {str(e)}")
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
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(tf.abs(h_with_ris[0]).numpy())
    plt.title('Channel with RIS')
    plt.colorbar(label='Magnitude')
    
    plt.subplot(1, 2, 2)
    plt.imshow(tf.abs(h_without_ris[0]).numpy())
    plt.title('Channel without RIS')
    plt.colorbar(label='Magnitude')
    
    plt.suptitle(f'RIS Gain: {ris_gain:.2f}x')
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
    channel_gen = SmartFactoryChannel(config, scene_provided=True)
    
    # Generate channel for multiple time steps
    channel_responses = []
    for t in range(config.num_time_steps):
        # Update AGV positions if needed
        if hasattr(channel_gen, 'update_agv_positions'):
            channel_gen.update_agv_positions(t)
        
        # Generate channel response
        channel_response = channel_gen.generate_channel()
        channel_responses.append(channel_response)
    
    # Ensure result directory exists
    result_dir = ensure_result_dir()
    
    # Analyze scene
    analyzer = ChannelAnalyzer(scene)
    
    # Visualize scene
    try:
        fig = analyzer.visualize_scene()
        scene_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(result_dir, f'factory_scene_3d_{scene_timestamp}.png')
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
        
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
    
    print(f"Analysis complete. Results saved in {result_dir}")

if __name__ == "__main__":
    main()