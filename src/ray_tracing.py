import os
import tensorflow as tf
import numpy as np
from sionna.rt import Scene, load_scene, Camera, PlanarArray
from sionna.rt import Transmitter, Receiver
import matplotlib.pyplot as plt
import mitsuba as mi

class RayTracingSimulator:
    def __init__(self):
        """Initialize the ray tracing simulator"""
        # Set the Mitsuba variant
        mi.set_variant('scalar_rgb')
        
        # Generate the scene XML first
        self.generate_scene_xml()
        
        # Initialize scene with empty scene first
        self.scene = Scene("__empty__")
        
        # Setup scene components
        self.setup_scene_components()

    def generate_scene_xml(self):
        """Generate the factory scene XML configuration"""
        scene_xml = """<?xml version="1.0" encoding="UTF-8"?>
<scene version="2.1.0">
    <integrator type="path">
        <integer name="max_depth" value="12"/>
    </integrator>

    <emitter type="constant" id="World">
        <rgb value="1.0 1.0 1.0" name="radiance"/>
    </emitter>

    <bsdf type="twosided" id="mat-itu_concrete">
        <bsdf type="diffuse">
            <rgb value="0.6 0.6 0.6" name="reflectance"/>
        </bsdf>
    </bsdf>

    <shape type="rectangle" id="floor">
        <transform name="to_world">
            <scale value="20 20 1"/>
            <translate value="10 10 0"/>
        </transform>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="rectangle" id="ceiling">
        <transform name="to_world">
            <scale value="20 20 1"/>
            <translate value="10 10 5"/>
            <rotate x="1" angle="180"/>
        </transform>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="rectangle" id="wall_north">
        <transform name="to_world">
            <scale value="20 5 1"/>
            <translate value="10 20 2.5"/>
            <rotate x="1" angle="-90"/>
        </transform>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="rectangle" id="wall_south">
        <transform name="to_world">
            <scale value="20 5 1"/>
            <translate value="10 0 2.5"/>
            <rotate x="1" angle="90"/>
        </transform>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="rectangle" id="wall_east">
        <transform name="to_world">
            <scale value="20 5 1"/>
            <translate value="20 10 2.5"/>
            <rotate y="1" angle="-90"/>
        </transform>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="rectangle" id="wall_west">
        <transform name="to_world">
            <scale value="20 5 1"/>
            <translate value="0 10 2.5"/>
            <rotate y="1" angle="90"/>
        </transform>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>
</scene>"""
        
        # Create src directory if it doesn't exist
        os.makedirs('src', exist_ok=True)
        
        # Write the XML file
        with open('src/factory_scene.xml', 'w') as f:
            f.write(scene_xml)

    def setup_scene_components(self):
        """Setup transmitters, receivers and cameras"""
        # Add transmitter (BS)
        tx = Transmitter(
            name="tx",
            position=[10.0, 10.0, 4.5],  # BS position
            orientation=[0.0, 0.0, -90.0]  # Facing down
        )
        self.scene.add(tx)

        # Add antenna array to transmitter
        tx_array = PlanarArray(
            num_rows=8,
            num_cols=8,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )
        tx.add_array(tx_array)

        # Add two receivers (AGVs)
        rx_positions = [
            [15.0, 15.0, 0.5],  # AGV 1 position
            [10.0, 15.0, 0.5]   # AGV 2 position
        ]
        
        rx_orientations = [
            [0.0, 0.0, 0.0],    # AGV 1 orientation
            [0.0, 0.0, 0.0]     # AGV 2 orientation
        ]

        for i in range(2):  # For 2 AGVs
            rx = Receiver(
                name=f"rx_{i}",
                position=rx_positions[i],
                orientation=rx_orientations[i]
            )
            # Add antenna array to receiver
            rx_array = PlanarArray(
                num_rows=2,
                num_cols=2,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V"
            )
            rx.add_array(rx_array)
            self.scene.add(rx)

        # Set carrier frequency
        self.scene.frequency = 28e9  # 28 GHz

        # Setup cameras
        camera_configs = {
            "top": {
                "position": [10.0, 10.0, 20.0],
                "look_at": [10.0, 10.0, 0.0]
            },
            "side": {
                "position": [30.0, 10.0, 5.0],
                "look_at": [10.0, 10.0, 0.0]
            },
            "corner": {
                "position": [20.0, 20.0, 10.0],
                "look_at": [10.0, 10.0, 0.0]
            }
        }

        self.cameras = {}
        for name, config in camera_configs.items():
            camera = Camera(
                f"{name}_view",
                position=config["position"]
            )
            self.cameras[name] = camera
            self.scene.add(camera)
            camera.look_at(config["look_at"])

    def compute_ray_paths(self):
        """Compute ray paths using ray tracing"""
        return self.scene.compute_paths(
            max_depth=3,
            method="fibonacci",
            num_samples=1000,
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