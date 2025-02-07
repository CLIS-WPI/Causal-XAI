import os
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

def setup_scene(config, ply_dir='src/meshes'):
    """
    Setup scene with PLY-based geometries
    """
    try:
        # Initialize empty scene correctly using load_scene
        scene = load_scene("__empty__", dtype=config.dtype)
        
        # Set frequency 
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        print(f"[DEBUG] Scene initialized with frequency: {scene.frequency/1e9} GHz")

        # Create radio materials using tf.float32 for all real values
        metal_material = RadioMaterial(
            name="shelf_metal",  
            relative_permittivity=tf.cast(1.0, tf.float32),
            conductivity=tf.cast(1e7, tf.float32),
            scattering_coefficient=tf.cast(0.1, tf.float32),
            xpd_coefficient=tf.cast(0.0, tf.float32)
        )
        scene.add(metal_material)

        concrete_material = RadioMaterial(
            name="wall_concrete",  
            relative_permittivity=tf.cast(4.0, tf.float32),
            conductivity=tf.cast(0.01, tf.float32),
            scattering_coefficient=tf.cast(0.2, tf.float32),
            xpd_coefficient=tf.cast(0.5, tf.float32)
        )
        scene.add(concrete_material)

        # Load PLY files for room boundaries
        room_objects = {
            'floor': os.path.join(ply_dir, 'floor.ply'),
            'ceiling': os.path.join(ply_dir, 'ceiling.ply'),
            'wall_xp': os.path.join(ply_dir, 'wall_xp.ply'),
            'wall_xm': os.path.join(ply_dir, 'wall_xm.ply'),
            'wall_yp': os.path.join(ply_dir, 'wall_yp.ply'),
            'wall_ym': os.path.join(ply_dir, 'wall_ym.ply')
        }

        # Load room boundaries from PLY files
        for name, ply_path in room_objects.items():
            try:
                room_obj = SceneObject(name=name)
                room_obj.load_from_file(ply_path)
                room_obj.radio_material = concrete_material
                print(f"Loaded room object: {name}")
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")

        # Load shelves from PLY files
        for i in range(5):  # 5 shelves as in config
            shelf_path = os.path.join(ply_dir, f'shelf_{i}.ply')
            try:
                shelf = SceneObject(name=f"shelf_{i}")
                shelf.load_from_file(shelf_path)
                shelf.radio_material = metal_material
                print(f"Loaded shelf {i}")
            except Exception as e:
                print(f"Error loading shelf_{i}: {str(e)}")

        # Add base station with correct type casting
        bs_pos = [tf.cast(x, tf.float32) for x in config.bs_position]
        tx = Transmitter(
            name="bs",
            position=bs_pos,
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(tx)

        # Configure antenna arrays
        scene.tx_array = PlanarArray(
            num_rows=config.bs_array[0],
            num_cols=config.bs_array[1],
            vertical_spacing=tf.cast(config.bs_array_spacing, tf.float32),
            horizontal_spacing=tf.cast(config.bs_array_spacing, tf.float32),
            pattern="tr38901",
            polarization="VH",
            dtype=config.dtype
        )

        scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=tf.cast(config.agv_array_spacing, tf.float32),
            horizontal_spacing=tf.cast(config.agv_array_spacing, tf.float32),
            pattern="iso",
            polarization="V",
            dtype=config.dtype
        )

        # Add AGVs
        for i in range(config.num_agvs):
            position = [12.0 - i*4.0, 5.0 + i*10.0, 0.5]
            position = [tf.cast(x, tf.float32) for x in position]
            rx = Receiver(
                name=f"agv_{i}",
                position=position,
                orientation=[0.0, 0.0, 0.0]
            )
            scene.add(rx)

        # Add RIS
        ris_pos = [tf.cast(x, tf.float32) for x in config.ris_position]
        ris = RIS(
            name="ris",
            position=ris_pos,
            num_rows=config.ris_elements[0],
            num_cols=config.ris_elements[1],
            orientation=config.ris_orientation,
            dtype=config.dtype
        )
        scene.add(ris)

        return scene

    except Exception as e:
        print(f"[CRITICAL ERROR] Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e

if __name__ == "__main__":
    from config import SmartFactoryConfig
    config = SmartFactoryConfig()
    setup_scene(config)