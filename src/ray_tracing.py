# src/ray_tracing.py

import os
import tensorflow as tf
import numpy as np
from sionna.rt import Scene, load_scene, Camera
import matplotlib.pyplot as plt
from sionna.rt import Transmitter, Receiver
import mitsuba as mi

class RayTracingSimulator:
    def __init__(self):
        # Set the Mitsuba variant
        mi.set_variant('scalar_rgb')
        
        # Initialize scene
        self.scene = Scene()
        
        # Generate and load the scene configuration
        self.generate_scene_xml()
        self.scene = Scene("src/factory_scene.xml")
        
        # Setup transmitters, receivers and cameras
        self.setup_scene_components()

    def generate_scene_xml(self):
        """Generate the factory scene XML configuration"""
        scene_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <scene version="2.1.0">
            <!-- Your existing XML content -->
        </scene>"""
        
        # Create src directory if it doesn't exist
        os.makedirs('src', exist_ok=True)
        
        # Write the XML file
        with open('src/factory_scene.xml', 'w') as f:
            f.write(scene_xml)

    def setup_scene_components(self):
        """Setup transmitters, receivers and cameras"""
        # Create config instance
        config = SmartFactoryConfig()
        
        # Add transmitter (BS)
        tx = Transmitter(name="tx",
                        position=config.bs_position,  # [10.0, 10.0, 4.5]
                        orientation=config.bs_orientation)  # [0.0, 0.0, -90.0]
        self.scene.add(tx)

        # Add receivers (AGVs)
        for i in range(config.num_agvs):
            rx_name = f"rx_{i}"
            # For this example, we'll need to add AGV positions to config
            # You might want to calculate positions based on your requirements
            rx = Receiver(name=rx_name,
                        position=[15.0, 15.0, config.agv_height],  # Use AGV height from config
                        orientation=[0.0, 0.0, 0.0])
            self.scene.add(rx)

        # Set frequency from config
        self.scene.frequency = config.carrier_frequency

        # Setup cameras
        self.cameras = {
            "top": Camera("top_view", position=[10.0, 10.0, 20.0]),
            "side": Camera("side_view", position=[30.0, 10.0, 5.0]),
            "corner": Camera("corner_view", position=[20.0, 20.0, 10.0])
        }

        for cam in self.cameras.values():
            self.scene.add(cam)
            cam.look_at([10.0, 10.0, 0.0])

    def compute_ray_paths(self):
        """Compute ray paths using ray tracing"""
        return self.scene.compute_paths(
            max_depth=3,
            method="fibonacci",
            los=True,
            reflection=True,
            diffraction=True,
            scattering=False
        )

    def visualize_scene(self):
        """Visualize the scene from different camera views"""
        plt.figure(figsize=(15, 5))
        
        for i, (name, camera) in enumerate(self.cameras.items(), 1):
            plt.subplot(1, 3, i)
            img = self.scene.render(camera)
            plt.imshow(img)
            plt.title(f"{name} view")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def analyze_paths(self, paths):
        """Analyze and visualize ray paths"""
        print("\nPath Information:")
        print(f"Number of paths: {paths.n_paths}")
        if paths.n_paths > 0:
            print(f"Path lengths: {paths.lengths}")
            print(f"Number of interactions: {paths.n_interactions}")
        
        # Visualize ray paths
        plt.figure(figsize=(10, 10))
        paths.show()
        plt.title("Ray Paths")
        plt.show()

        # Get and plot channel impulse response
        a, tau = paths.cir()
        plt.figure(figsize=(10, 5))
        plt.stem(tau[0,0,0].numpy(), np.abs(a[0,0,0].numpy()))
        plt.xlabel('Delay [s]')
        plt.ylabel('Magnitude')
        plt.title('Channel Impulse Response')
        plt.grid(True)
        plt.show()

def main():
    # Create simulator instance
    simulator = RayTracingSimulator()
    
    # Compute paths
    paths = simulator.compute_ray_paths()
    
    # Visualize scene
    simulator.visualize_scene()
    
    # Analyze paths
    simulator.analyze_paths(paths)

if __name__ == "__main__":
    main()