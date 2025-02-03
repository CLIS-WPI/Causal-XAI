import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import Scene, PlanarArray, Antenna
from sionna.rt.previewer import InteractiveDisplay
from sionna.rt import load_scene
# Create an empty scene

scene = load_scene("__empty__")
class ChannelAnalyzer:
    """
    A class for analyzing and visualizing wireless channels in Sionna.
    """
    def __init__(self, scene, resolution=(1024, 768), fov=45):
        """
        Initialize the channel analyzer.

        Parameters:
        -----------
        scene : sionna.rt.Scene
            The scene to analyze
        resolution : tuple
            Display resolution (width, height)
        fov : float 
            Field of view in degrees
        """
        self.scene = scene
        self.resolution = resolution
        self.fov = fov
        self._preview = None

    def visualize_scene(self):
        """
        Create a static visualization of the scene using matplotlib
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get room dimensions from config
        room_dims = [20.0, 20.0, 5.0]  # Default dimensions
        
        # Create wireframe for room boundaries
        x = np.array([0, room_dims[0]])
        y = np.array([0, room_dims[1]])
        z = np.array([0, room_dims[2]])
        
        # Plot room edges
        for i in [0, 1]:
            for j in [0, 1]:
                ax.plot3D([x[0], x[1]], [y[i], y[i]], [z[j], z[j]], 'gray', alpha=0.5)
                ax.plot3D([x[i], x[i]], [y[0], y[1]], [z[j], z[j]], 'gray', alpha=0.5)
                ax.plot3D([x[i], x[i]], [y[j], y[j]], [z[0], z[1]], 'gray', alpha=0.5)
        
        # Plot transmitters if any exist
        if hasattr(self.scene, 'transmitters'):
            for tx_name, tx in self.scene.transmitters.items():
                pos = tx.position.numpy()
                ax.scatter(pos[0], pos[1], pos[2], c='red', marker='^', s=100, label=f'TX: {tx_name}')
        
        # Plot receivers if any exist
        if hasattr(self.scene, 'receivers'):
            for rx_name, rx in self.scene.receivers.items():
                pos = rx.position.numpy()
                ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='o', s=100, label=f'RX: {rx_name}')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Smart Factory Scene Overview')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Set axis limits
        ax.set_xlim([0, room_dims[0]])
        ax.set_ylim([0, room_dims[1]])
        ax.set_zlim([0, room_dims[2]])
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 0.5])  # Adjust the last number to change vertical stretch
        
        return fig

    def plot_channel_matrix(self, channel_matrix):
        """
        Visualize the channel matrix magnitude.
        
        Parameters:
        -----------
        channel_matrix : tf.Tensor
            Complex channel matrix
        """
        # Convert to magnitude in dB
        magnitude_db = 20 * np.log10(np.abs(channel_matrix))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(magnitude_db, aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Channel Matrix Magnitude')
        plt.xlabel('Receive Antenna')
        plt.ylabel('Transmit Antenna')
        plt.show()

    def plot_path_gains(self, paths):
        """
        Visualize path gains distribution.
        
        Parameters:
        -----------
        paths : sionna.rt.Paths
            Paths object containing path information
        """
        gains_db = 10 * np.log10(np.abs(paths.a))
        
        plt.figure(figsize=(10, 6))
        plt.hist(gains_db.flatten(), bins=50)
        plt.title('Path Gains Distribution')
        plt.xlabel('Path Gain (dB)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

    def plot_delay_spread(self, paths):
        """
        Visualize delay spread of paths.
        
        Parameters:
        -----------
        paths : sionna.rt.Paths
            Paths object containing path information
        """
        delays = paths.tau
        
        plt.figure(figsize=(10, 6))
        plt.hist(delays.flatten(), bins=50)
        plt.title('Path Delay Distribution')
        plt.xlabel('Delay (s)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

    def analyze_channel(self, paths):
        """
        Perform comprehensive channel analysis.
        
        Parameters:
        -----------
        paths : sionna.rt.Paths
            Paths object to analyze
        """
        print("Channel Analysis Summary:")
        print("-----------------------")
        
        # Number of paths
        print(f"Total number of paths: {paths.n_paths}")
        
        # Path types statistics
        if hasattr(paths, 'types'):
            unique_types, counts = np.unique(paths.types, return_counts=True)
            print("\nPath Types Distribution:")
            for t, c in zip(unique_types, counts):
                print(f"{t}: {c}")
        
        # Average path gain
        avg_gain_db = 10 * np.log10(np.mean(np.abs(paths.a)))
        print(f"\nAverage path gain: {avg_gain_db:.2f} dB")
        
        # Delay spread
        delay_spread = np.std(paths.tau)
        print(f"RMS delay spread: {delay_spread*1e9:.2f} ns")
        
        # Visualizations
        self.plot_path_gains(paths)
        self.plot_delay_spread(paths)

    def plot_coverage_map(self, coverage_map, tx_index=0):
        """
        Visualize coverage map for a specific transmitter.
        
        Parameters:
        -----------
        coverage_map : sionna.rt.CoverageMap
            Coverage map to visualize
        tx_index : int
            Index of transmitter to show coverage for
        """
        if self._preview is None:
            self.visualize_scene()
            
        self._preview.plot_coverage_map(
            coverage_map,
            tx=tx_index,
            db_scale=True
        )
