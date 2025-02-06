import tensorflow as tf
import sionna
from sionna.rt import (
    load_scene, 
    Scene,
    PlanarArray, 
    Transmitter, 
    Receiver, 
    RIS, 
    RadioMaterial, 
    SceneObject
)
from sionna.constants import SPEED_OF_LIGHT

print(f"[DEBUG] Using Sionna version: {sionna.__version__}")

def setup_scene(config):
    """Setup scene using Sionna's built-in objects
    
    Args:
        config: Configuration object containing scene parameters
        
    Returns:
        scene: Configured Sionna Scene object
    """
    try:
        print("[DEBUG] Starting scene setup...")
        
        # Initialize empty scene with proper dtype
        print(f"[DEBUG] Initializing empty scene with dtype: {config.dtype}")
        scene = load_scene("__empty__", dtype=config.dtype)
        scene.frequency = config.carrier_frequency
        print(f"[DEBUG] Scene initialized with frequency: {scene.frequency} Hz")
        
        # Configure antenna arrays
        print("[DEBUG] Configuring antenna arrays...")
        try:
            scene.tx_array = PlanarArray(
                num_rows=config.bs_array[0],
                num_cols=config.bs_array[1],
                vertical_spacing=config.bs_array_spacing,
                horizontal_spacing=config.bs_array_spacing,
                pattern="tr38901",
                polarization="VH",
                dtype=config.dtype
            )
            print("[DEBUG] TX array configured successfully")
            
            scene.rx_array = PlanarArray(
                num_rows=config.agv_array[0],
                num_cols=config.agv_array[1],
                vertical_spacing=config.agv_array_spacing,
                horizontal_spacing=config.agv_array_spacing,
                pattern="iso",
                polarization="V",
                dtype=config.dtype
            )
            print("[DEBUG] RX array configured successfully")
        except Exception as e:
            print(f"[ERROR] Failed to configure antenna arrays: {str(e)}")
            raise

        # Add base station
        print("[DEBUG] Adding base station...")
        try:
            tx = Transmitter(
                name="bs",
                position=config.bs_position,
                orientation=[0.0, 0.0, 0.0]
            )
            scene.add(tx)
            print(f"[DEBUG] Base station added at position: {config.bs_position}")
        except Exception as e:
            print(f"[ERROR] Failed to add base station: {str(e)}")
            raise
        
        # Add AGVs (mobile receivers)
        print("[DEBUG] Adding AGVs...")
        try:
            for i in range(config.num_agvs):
                position = [12.0 - i*4.0, 5.0 + i*10.0, 0.5]
                rx = Receiver(
                    name=f"agv_{i}",
                    position=position,
                    orientation=[0.0, 0.0, 0.0]
                )
                scene.add(rx)
                print(f"[DEBUG] AGV {i} added at position: {position}")
        except Exception as e:
            print(f"[ERROR] Failed to add AGVs: {str(e)}")
            raise
        
        # Add RIS
        print("[DEBUG] Adding RIS...")
        try:
            ris = RIS(
                name="ris",
                position=config.ris_position,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                num_modes=config.ris_modes,
                orientation=config.ris_orientation,
                dtype=config.dtype
            )
            scene.add(ris)
            print(f"[DEBUG] RIS added at position: {config.ris_position}")
        except Exception as e:
            print(f"[ERROR] Failed to add RIS: {str(e)}")
            raise
        
        # Add metallic shelves
        print("[DEBUG] Adding materials and shelves...")
        try:
            material = RadioMaterial(
                name="metal",
                relative_permittivity=1.0,
                conductivity=1e7,
                scattering_coefficient=0.1,
                xpd_coefficient=0.0
            )
            scene.add(material)
            print("[DEBUG] Metal material added")
            
            for i, pos in enumerate(config.scene_objects['shelf_positions']):
                shelf = SceneObject(
                    name=f"shelf_{i}",
                    position=pos,
                    size=config.scene_objects['shelf_dimensions'],
                    orientation=[0.0, 0.0, 0.0]
                )
                # First add to scene, then set material
                scene.add(shelf)
                shelf.radio_material = material
                print(f"[DEBUG] Shelf {i} added at position: {pos}")
        except Exception as e:
            print(f"[ERROR] Failed to add shelves: {str(e)}")
            raise
            
        # Add room boundaries
        print("[DEBUG] Adding room boundaries...")
        try:
            walls_material = RadioMaterial(
                name="walls",
                relative_permittivity=4.0,
                conductivity=0.01,
                scattering_coefficient=0.2,
                xpd_coefficient=0.5
            )
            scene.add(walls_material)
            print("[DEBUG] Wall material added")
            
            room_objects = [
                ("floor", [config.room_dim[0]/2, config.room_dim[1]/2, 0], 
                [config.room_dim[0], config.room_dim[1], 0.3], [0.0, 0.0, 0.0]),
                ("ceiling", [config.room_dim[0]/2, config.room_dim[1]/2, config.room_dim[2]], 
                [config.room_dim[0], config.room_dim[1], 0.3], [0.0, 0.0, 0.0]),
                ("wall_north", [config.room_dim[0]/2, config.room_dim[1], config.room_dim[2]/2], 
                [config.room_dim[0], 0.3, config.room_dim[2]], [0.0, 90.0, 0.0]),
                ("wall_south", [config.room_dim[0]/2, 0, config.room_dim[2]/2], 
                [config.room_dim[0], 0.3, config.room_dim[2]], [0.0, 90.0, 0.0]),
                ("wall_east", [config.room_dim[0], config.room_dim[1]/2, config.room_dim[2]/2], 
                [0.3, config.room_dim[1], config.room_dim[2]], [90.0, 0.0, 0.0]),
                ("wall_west", [0, config.room_dim[1]/2, config.room_dim[2]/2], 
                [0.3, config.room_dim[1], config.room_dim[2]], [90.0, 0.0, 0.0])
            ]
            
            for name, position, size, orientation in room_objects:
                obj = SceneObject(
                    name=name,
                    position=position,
                    size=size,
                    orientation=orientation
                )
                # First add to scene
                scene.add(obj)
                # Then set material
                obj.radio_material = walls_material
                print(f"[DEBUG] Room object {name} added")
        except Exception as e:
            print(f"[ERROR] Failed to add room boundaries: {str(e)}")
            raise
            
        print("[DEBUG] Scene setup completed successfully")
        return scene
        
    except Exception as e:
        print(f"[ERROR] Fatal error in scene setup: {str(e)}")
        raise RuntimeError(f"Error setting up scene: {str(e)}") from e