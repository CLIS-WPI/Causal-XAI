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
        
        # Add base station with proper tensor formatting
        bs_pos = tf.constant(config.bs_position, dtype=tf.float32)
        bs_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        
        try:
            bs = Transmitter(
                name="bs",
                position=bs_pos,
                orientation=bs_orientation
            )
            bs.object_id = current_object_id
            current_object_id += 1
            scene.add(bs)
            print("[DEBUG] Base station added successfully")
            
            # Verify base station was added correctly
            if "bs" not in scene.transmitters:
                raise RuntimeError("Base station not found in scene.transmitters after adding")
                
        except Exception as e:
            raise RuntimeError(f"Failed to add base station: {str(e)}")

        # Modified verification step
        print("[DEBUG] Verifying scene objects...")
        required_objects = ['bs', 'ris']
        for obj_name in required_objects:
            # Check in both objects and transmitters dictionaries
            found = False
            if obj_name in scene.objects:
                found = True
            elif obj_name in scene.transmitters:
                found = True
            
            if not found:
                print(f"[DEBUG] Available objects: {list(scene.objects.keys())}")
                print(f"[DEBUG] Available transmitters: {list(scene.transmitters.keys())}")
                raise RuntimeError(f"Failed to add required object: {obj_name}")
                
        print("[DEBUG] Scene verification completed successfully")
        return scene

    except Exception as e:
        print(f"[CRITICAL ERROR] Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e
        
        try:
            bs = Transmitter(
                name="bs",  # This name must match what's expected in validation
                position=bs_pos,
                orientation=bs_orientation
            )
            bs.object_id = current_object_id
            current_object_id += 1
            scene.add(bs)
            print("[DEBUG] Base station added successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to add base station: {str(e)}")

        # Configure antenna arrays with proper tensor formatting
        try:
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
            print("[DEBUG] Antenna arrays configured successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to configure antenna arrays: {str(e)}")

        # Add AGVs with proper tensor formatting
        try:
            for i in range(config.num_agvs):
                agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, 0.5], dtype=tf.float32)
                agv_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
                
                rx = Receiver(
                    name=f"agv_{i}",
                    position=agv_pos,
                    orientation=agv_orientation
                )
                rx.object_id = current_object_id
                current_object_id += 1
                scene.add(rx)
            print(f"[DEBUG] {config.num_agvs} AGVs added successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to add AGVs: {str(e)}")

        # Add RIS with proper tensor formatting
        try:
            ris_pos = tf.constant(config.ris_position, dtype=tf.float32)
            ris_orientation = tf.constant(config.ris_orientation, dtype=tf.float32)
            
            ris = RIS(
                name="ris",
                position=ris_pos,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                orientation=ris_orientation,
                dtype=config.dtype
            )
            ris.object_id = current_object_id
            current_object_id += 1
            scene.add(ris)
            print("[DEBUG] RIS added successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to add RIS: {str(e)}")

        # Print debug information
        print(f"[DEBUG] Total objects in scene: {len(scene.objects)}")
        print("[DEBUG] Scene setup completed successfully")
        
        # Verify required objects are present
        required_objects = ['bs', 'ris']
        for obj_name in required_objects:
            if not any(obj.name == obj_name for obj in scene.objects.values()):
                raise RuntimeError(f"Failed to add required object: {obj_name}")
            
        # Final verification of scene configuration
        if not hasattr(scene, 'tx_array') or not hasattr(scene, 'rx_array'):
            raise RuntimeError("Scene is missing antenna array configuration")
            
        return scene

    except Exception as e:
        print(f"[CRITICAL ERROR] Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e