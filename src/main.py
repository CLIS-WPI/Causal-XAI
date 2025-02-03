from config import SmartFactoryConfig
from channel_generator import SmartFactoryChannel
from channel_analyzer import ChannelAnalyzer
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
from sionna.rt import Scene, Transmitter, Receiver, RIS, SceneObject

def setup_scene(config):
    """
    Set up the simulation scene based on configuration
    """
    # Create empty scene
    scene = load_scene("__empty__")
    
    # Configure antenna arrays
    scene.tx_array = PlanarArray(
        num_rows=config.bs_array[0],
        num_cols=config.bs_array[1],
        vertical_spacing=0.7,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH"
    )
    
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross"
    )
    
    # Add transmitter (base station)
    tx = Transmitter(
        name="bs",
        position=config.bs_position,
        orientation=config.bs_orientation
    )
    scene.add(tx)
    
    # Add receivers (AGVs)
    for i in range(config.num_agvs):
        rx = Receiver(
            name=f"agv_{i}",
            position=[10.0 + i*2, 10.0, config.agv_height],  # Example positions
            orientation=[0, 0, 0]
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
    """Analyze and plot channel properties"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot channel magnitude
    plt.figure(figsize=(10, 6))
    h = channel_response['h']
    print(f"Channel response shape: {h.shape}")  # Debug print
    
    # Properly reshape the channel response for visualization
    h_mag = tf.abs(h)
    # Reduce across all dimensions except the first and last
    try:
        h_2d = tf.reduce_mean(h_mag, axis=list(range(1, len(h.shape)-1)))
        h_2d = h_2d.numpy()
    except Exception as e:
        print(f"Error in channel magnitude calculation: {e}")
        h_2d = tf.reduce_mean(h_mag, axis=-1).numpy()  # Fallback to simpler reduction
    
    plt.imshow(h_2d, aspect='auto')
    plt.colorbar(label='Average Magnitude')
    plt.title(f'Channel Magnitude ({config.scenario} scenario)')
    plt.xlabel('Time Steps')
    plt.ylabel('Batch')
    plt.savefig(os.path.join(result_dir, f'channel_magnitude_{timestamp}.png'))
    plt.close()
    
    # Plot delay distribution
    plt.figure(figsize=(10, 6))
    delays_ns = channel_response['tau'].numpy().flatten() * 1e9  # Convert to ns
    valid_delays = delays_ns[~np.isnan(delays_ns)]
    if len(valid_delays) > 0:
        plt.hist(valid_delays, bins=min(50, len(valid_delays)), density=True)
        plt.title('Path Delay Distribution')
        plt.xlabel('Delay (ns)')
        plt.ylabel('Density')
        plt.savefig(os.path.join(result_dir, f'delay_distribution_{timestamp}.png'))
    plt.close()
    
    # Plot LoS condition distribution
    plt.figure(figsize=(10, 6))
    los_data = channel_response['los_condition'].numpy().flatten()
    # Explicitly convert boolean data to integers before plotting
    los_data = los_data.astype(np.int32)
    plt.hist(los_data, bins=[-0.5, 0.5, 1.5], rwidth=0.8)  # Use explicit bin edges
    plt.xticks([0, 1], ['NLoS', 'LoS'])
    plt.title('LoS/NLoS Distribution')
    plt.xlabel('Channel State')
    plt.ylabel('Count')
    plt.savefig(os.path.join(result_dir, f'los_distribution_{timestamp}.png'))
    plt.close()
    
    # Power delay profile
    plt.figure(figsize=(10, 6))
    try:
        # Average power over all dimensions except paths
        h_power = tf.reduce_mean(tf.abs(h)**2, axis=[0,1,2,3,4])  
        h_power = h_power.numpy()
        h_power_db = 10 * np.log10(np.maximum(h_power, 1e-10))  # Use maximum for numerical stability
        plt.plot(range(len(h_power)), h_power_db)
        plt.title('Power Delay Profile')
        plt.xlabel('Path Index')
        plt.ylabel('Average Power (dB)')
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, f'power_delay_profile_{timestamp}.png'))
    except Exception as e:
        print(f"Error in power delay profile calculation: {e}")
    plt.close()

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
    
    # Set up the scene in channel generator if it has a setup method
    if hasattr(channel_gen, 'setup_scene'):
        channel_gen.setup_scene(scene)
    
    # Generate channel
    channel_response = channel_gen.generate_channel()
    
    # Ensure result directory exists
    result_dir = ensure_result_dir()
    
    # Analyze scene if available
    if hasattr(channel_gen, 'scene'):
        analyzer = ChannelAnalyzer(channel_gen.scene)
        try:
            fig = analyzer.visualize_scene()
            scene_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(result_dir, f'scene_3d_{scene_timestamp}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Scene visualization saved to: {fig_path}")
        except Exception as e:
            print(f"Error visualizing scene: {str(e)}")
    
    # Analyze channel properties
    analyze_channel_properties(channel_response, config, result_dir)
    
    # New analyses
    try:
        # Analyze RIS effectiveness
        ris_gain = analyze_ris_effectiveness(channel_response, result_dir)
        print(f"RIS Gain: {ris_gain:.2f}x")
        
        # Analyze blockage statistics
        los_ratio = analyze_blockage_statistics(channel_response, result_dir)
        print(f"LOS Ratio: {los_ratio:.2%}")
        
        # Plot AGV trajectories
        if hasattr(channel_gen, 'positions_history'):
            plot_agv_trajectories(channel_gen, result_dir)
            print("AGV trajectories plotted successfully")
    except Exception as e:
        print(f"Error in additional analyses: {str(e)}")
    
    # Save channel statistics
    save_channel_stats(channel_response, config, result_dir)
    
    print(f"Analysis complete. Results saved in {result_dir}")

if __name__ == "__main__":
    main()