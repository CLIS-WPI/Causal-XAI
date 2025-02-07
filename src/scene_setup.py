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

def setup_scene(config):
    """
    Setup scene using factory_scene.xml which contains the PLY references
    """
    try:
        # Load scene from XML file that contains PLY references
        scene = load_scene('src/factory_scene.xml', dtype=config.dtype)
        
        # Set frequency 
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        print(f"[DEBUG] Scene initialized with frequency: {scene.frequency/1e9} GHz")

        # First, process existing scene objects to get the current max object ID
        max_object_id = -1
        for obj in scene.objects.values():
            if hasattr(obj, 'object_id'):
                max_object_id = max(max_object_id, obj.object_id)
        
        # Start new object IDs after the existing maximum
        current_object_id = max_object_id + 1
        
        # Add base station
        bs_pos = [tf.cast(x, tf.float32) for x in config.bs_position]
        tx = Transmitter(
            name="bs",
            position=bs_pos,
            orientation=[0.0, 0.0, 0.0]
        )
        tx.object_id = current_object_id
        current_object_id += 1
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
            rx.object_id = current_object_id
            current_object_id += 1
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
        ris.object_id = current_object_id
        current_object_id += 1
        scene.add(ris)

        # Print debug information
        print(f"[DEBUG] Total objects in scene: {len(scene.objects)}")
        print("[DEBUG] Scene setup completed successfully")
        
        return scene

    except Exception as e:
        raise RuntimeError(f"Failed to setup scene: {str(e)}") from e

    except Exception as e:
        print(f"[CRITICAL ERROR] Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e