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
    # Taking first batch [0], first receiver [0], first receiver antenna [0], 
    # first transmitter [0], all transmit antennas [:], first stream [0], all subcarriers [:]
    h_2d = channel_matrix[0, 0, 0, 0, :, 0, :].numpy()
    
    # Calculate magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(h_2d))
    
    # Create the plot
    plt.clf()  # Clear the current figure
    plt.imshow(magnitude_db, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Transmit Antenna Index')
    plt.title('Channel Magnitude Response')
    plt.tight_layout()
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