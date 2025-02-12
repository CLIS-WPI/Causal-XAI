# First, import necessary libraries
%matplotlib inline
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

def plot_channel_magnitude(channel_matrix):
    """Plot channel magnitude response"""
    plt.figure(figsize=(10,6))
    magnitude_db = 20 * np.log10(np.abs(channel_matrix.numpy()))
    plt.imshow(magnitude_db, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Antenna Index')
    plt.title('Channel Magnitude Response')
    plt.show()

def plot_path_delays(path_delays):
    """Plot histogram of path delays"""
    plt.figure(figsize=(10,6))
    plt.hist(path_delays.numpy().flatten(), bins=50)
    plt.xlabel('Delay (s)')
    plt.ylabel('Count')
    plt.title('Path Delay Distribution')
    plt.grid(True)
    plt.show()

# Use these functions to visualize your channel data
if 'channel_matrices' in channel_data:
    plot_channel_magnitude(channel_data['channel_matrices'])

if 'path_delays' in channel_data:
    plot_path_delays(channel_data['path_delays'])

# Print channel statistics
print("\nChannel Statistics:")
print(f"Channel matrix shape: {channel_data['channel_matrices'].shape}")
print(f"Maximum magnitude: {np.max(np.abs(channel_data['channel_matrices'].numpy())):.2f}")
print(f"Minimum magnitude: {np.min(np.abs(channel_data['channel_matrices'].numpy())):.2f}")