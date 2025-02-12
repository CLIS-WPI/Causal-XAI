# src/channel_analyzer.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import SmartFactoryConfig
from channel_generator import SmartFactoryChannel

# Create config and channel generator
config = SmartFactoryConfig()
channel_gen = SmartFactoryChannel(config)

# Generate channel data
channel_data = channel_gen.generate_channel()

# Now let's create some basic visualization functions:

# Visualization functions
def plot_channel_magnitude(channel_matrix):
    """Plot channel magnitude response"""
    plt.figure(figsize=(10,6))
    
    # Extract 2D slice from 7D tensor (1, 2, 1, 7, 128, 1, 1024)
    # Shape interpretation:
    # - batch_size (1)
    # - num_rx (2)
    # - num_rx_ant (1)
    # - num_tx (7)
    # - num_tx_ant (128)
    # - num_streams (1)
    # - num_subcarriers (1024)
    
    # Take first batch, first receiver, first rx antenna, first tx, all tx antennas, first stream, all subcarriers
    h_2d = tf.squeeze(channel_matrix[0, 0, 0, 0, :, 0, :])
    
    # Convert to numpy and calculate magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(h_2d.numpy()))
    
    # Plot
    plt.imshow(magnitude_db, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Transmit Antenna Index')
    plt.title('Channel Magnitude Response')
    plt.show()

# Visualization code
if 'h' in channel_data:
    print(f"Channel matrix shape: {channel_data['h'].shape}")
    plot_channel_magnitude(channel_data['h'])

def plot_path_delays(path_delays):
    """Plot histogram of path delays"""
    plt.figure(figsize=(10,6))
    delays = path_delays.numpy().flatten()
    # Remove any zero or invalid delays
    delays = delays[delays > 0]
    
    plt.hist(delays, bins=50)
    plt.xlabel('Delay (s)')
    plt.ylabel('Count')
    plt.title('Path Delay Distribution')
    plt.grid(True)
    plt.show()

# Visualization code
if 'h' in channel_data:
    print(f"Channel matrix shape: {channel_data['h'].shape}")
    plot_channel_magnitude(channel_data['h'])

if 'tau' in channel_data:
    print(f"Path delays shape: {channel_data['tau'].shape}")
    plot_path_delays(channel_data['tau'])

# Print channel statistics
print("\nChannel Statistics:")
h = channel_data['h']
print(f"Channel matrix shape: {h.shape}")
print(f"Maximum magnitude: {np.max(np.abs(h.numpy())):.2f}")
print(f"Minimum magnitude: {np.min(np.abs(h.numpy())):.2f}")