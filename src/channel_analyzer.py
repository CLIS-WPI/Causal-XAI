import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sionna.rt import Scene, PlanarArray, Antenna, RIS
from sionna.rt.previewer import InteractiveDisplay
from sionna.rt import load_scene

class ChannelAnalyzer:
    """
    A class for analyzing and visualizing wireless channels in Smart Factory scenarios.
    """
    def __init__(self, scene, resolution=(1024, 768), fov=45):
        """
        Initialize the channel analyzer.

        Parameters:
        -----------
        scene : sionna.rt.Scene
            The smart factory scene to analyze
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
        Create a static visualization of the smart factory scene using matplotlib
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get room dimensions
        room_dims = [20.0, 20.0, 5.0]  # Smart factory dimensions
        
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
        
        # Plot transmitters (Base Station)
        if hasattr(self.scene, 'transmitters'):
            for tx_name, tx in self.scene.transmitters.items():
                pos = tx.position.numpy()
                ax.scatter(pos[0], pos[1], pos[2], c='red', marker='^', s=100, 
                         label=f'BS: {tx_name}')
        
        # Plot receivers (AGVs)
        if hasattr(self.scene, 'receivers'):
            for rx_name, rx in self.scene.receivers.items():
                pos = rx.position.numpy()
                ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='o', s=100, 
                         label=f'AGV: {rx_name}')
        
        # Plot RIS elements
        if hasattr(self.scene, 'ris'):
            for ris_name, ris in self.scene.ris.items():
                pos = ris.position.numpy()
                ax.scatter(pos[0], pos[1], pos[2], c='green', marker='s', s=100, 
                         label=f'RIS: {ris_name}')
                
                # Plot RIS panel outline
                self._plot_ris_panel(ax, ris)
        
        # Plot shelves
        if hasattr(self.scene, 'objects'):
            for obj_name, obj in self.scene.objects.items():
                if 'shelf' in obj_name:
                    pos = obj.position.numpy()
                    size = obj.size.numpy()
                    self._plot_shelf(ax, pos, size)
        
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
        ax.set_box_aspect([1, 1, 0.5])
        
        return fig

    def _plot_ris_panel(self, ax, ris):
        """Plot RIS panel outline"""
        pos = ris.position.numpy()
        num_rows = ris.num_rows
        num_cols = ris.num_cols
        spacing = ris.element_spacing.numpy()
        
        width = num_cols * spacing
        height = num_rows * spacing
        
        # Plot RIS panel outline
        x = [pos[0] - width/2, pos[0] + width/2, pos[0] + width/2, pos[0] - width/2, pos[0] - width/2]
        y = [pos[1], pos[1], pos[1], pos[1], pos[1]]
        z = [pos[2] - height/2, pos[2] - height/2, pos[2] + height/2, pos[2] + height/2, pos[2] - height/2]
        
        ax.plot3D(x, y, z, 'g--', alpha=0.5)

    def _plot_shelf(self, ax, pos, size):
        """Plot shelf as a wireframe box"""
        x = pos[0]
        y = pos[1]
        z = pos[2]
        dx, dy, dz = size
        
        # Define vertices
        vertices = np.array([
            [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
        ])
        
        # Define edges
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        
        # Plot edges
        for edge in edges:
            ax.plot3D(
                vertices[edge, 0],
                vertices[edge, 1],
                vertices[edge, 2],
                'brown',
                alpha=0.5
            )

    def analyze_channel_with_ris(self, channel_response):
        """
        Analyze channel characteristics with RIS effects.
        
        Parameters:
        -----------
        channel_response : dict
            Dictionary containing channel responses with and without RIS
        """
        h_with_ris = channel_response['h_with_ris']
        h_without_ris = channel_response['h_without_ris']
        
        # Calculate RIS gain
        ris_gain = tf.reduce_mean(tf.abs(h_with_ris)**2) / tf.reduce_mean(tf.abs(h_without_ris)**2)
        
        plt.figure(figsize=(12, 6))
        
        # Plot channel magnitude with RIS
        plt.subplot(121)
        plt.imshow(tf.abs(h_with_ris[0]).numpy(), aspect='auto')
        plt.colorbar(label='Magnitude')
        plt.title('Channel with RIS')
        
        # Plot channel magnitude without RIS
        plt.subplot(122)
        plt.imshow(tf.abs(h_without_ris[0]).numpy(), aspect='auto')
        plt.colorbar(label='Magnitude')
        plt.title('Channel without RIS')
        
        plt.suptitle(f'RIS Gain: {float(ris_gain):.2f}x')
        plt.tight_layout()
        
        return ris_gain

    # ... (keep other existing methods) ...