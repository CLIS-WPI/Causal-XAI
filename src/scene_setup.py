import tensorflow as tf
import sionna
from sionna.rt import (
    load_scene, Scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial, SceneObject
)

print(f"[DEBUG] Using Sionna version: {sionna.__version__}")

def setup_scene(config):
    """Setup the factory scene with transmitters, receivers, and RIS"""
    try:
        # Load scene and set frequency
        scene = load_scene('src/factory_scene.xml', dtype=config.dtype)
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        print(f"[DEBUG] Scene initialized with frequency: {scene.frequency/1e9} GHz")
        print(f"[DEBUG] Initial scene objects: {list(scene.objects.keys())}")

        # Add base station
        try:
            bs_pos = tf.constant(config.bs_position, dtype=tf.float32)
            bs_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            
            print("[DEBUG] Scene objects before BS addition:", list(scene.objects.keys()))
            print("[DEBUG] Current object IDs:", [(name, getattr(obj, 'object_id', None)) 
                                                for name, obj in scene.objects.items()])
            
            bs = Transmitter(
                name="bs",
                position=bs_pos,
                orientation=bs_orientation,
                dtype=config.dtype
            )
            
            # Set object_id before adding to scene
            bs.object_id = len(scene.objects)
            print(f"[DEBUG] Base station object_id set to: {getattr(bs, 'object_id', None)}")
            
            # Add to scene
            scene.add(bs)
            
            # Explicitly add to scene objects if not already added
            if 'bs' not in scene.objects:
                scene._scene_objects['bs'] = bs
            
            print(f"[DEBUG] Base station added successfully at position {bs_pos}")
            print(f"[DEBUG] Base station registered as transmitter: {bs.name in scene.transmitters}")
            print(f"[DEBUG] Base station registered as object: {bs.name in scene.objects}")
            print(f"[DEBUG] Scene objects after BS addition: {list(scene.objects.keys())}")
            
        except Exception as e:
            print(f"[ERROR] Failed to add base station: {str(e)}")
            raise RuntimeError(f"Failed to add base station: {str(e)}")

        # Configure antenna arrays
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

        # Add RIS
        try:
            print("[DEBUG] Scene objects before RIS addition:", list(scene.objects.keys()))
            print("[DEBUG] Current object IDs:", [(name, getattr(obj, 'object_id', None)) 
                                                for name, obj in scene.objects.items()])
            
            ris_pos = tf.constant(config.ris_position, dtype=tf.float32)
            ris_orientation = tf.constant(config.ris_orientation, dtype=tf.float32)
            
            ris = RIS(
                name="ris",
                position=ris_pos,
                orientation=ris_orientation,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                dtype=config.dtype
            )
            
            # Set object_id before adding to scene
            ris.object_id = len(scene.objects)
            print(f"[DEBUG] RIS object_id set to: {getattr(ris, 'object_id', None)}")
            
            # Add to scene
            scene.add(ris)
            
            # Explicitly add to scene objects if not already added
            if 'ris' not in scene.objects:
                scene._scene_objects['ris'] = ris
            
            print("[DEBUG] RIS added successfully")
            print(f"[DEBUG] RIS registered as RIS: {'ris' in scene.ris}")
            print(f"[DEBUG] RIS registered as object: {'ris' in scene.objects}")
            print(f"[DEBUG] Scene objects after RIS addition: {list(scene.objects.keys())}")
            
        except Exception as e:
            print(f"[ERROR] Failed to add RIS: {str(e)}")
            raise RuntimeError(f"Failed to add RIS: {str(e)}")

        # Add AGVs
        try:
            for i in range(config.num_agvs):
                agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, 0.5], dtype=tf.float32)
                agv_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
                
                rx = Receiver(
                    name=f"agv_{i}",
                    position=agv_pos,
                    orientation=agv_orientation,
                    dtype=config.dtype
                )
                scene.add(rx)
            print(f"[DEBUG] {config.num_agvs} AGVs added successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to add AGVs: {str(e)}")

        verify_scene(scene)
        return scene

    except Exception as e:
        print(f"[CRITICAL ERROR] Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e

def verify_scene(scene):
    """Verify all required components are present in the scene"""
    print("\n[DEBUG] Starting scene verification...")
    print(f"[DEBUG] Available objects: {list(scene.objects.keys())}")
    print(f"[DEBUG] Available transmitters: {list(scene.transmitters.keys())}")
    print(f"[DEBUG] Available RIS: {list(scene.ris.keys()) if hasattr(scene, 'ris') else 'No RIS'}")
    print(f"[DEBUG] Object IDs: {[(name, getattr(obj, 'object_id', None)) for name, obj in scene.objects.items()]}")
    
    # Verify base station
    if "bs" not in scene.transmitters:
        raise RuntimeError("Base station not found in transmitters")
    if "bs" not in scene.objects:
        print("[WARNING] Base station not in scene objects, attempting to add...")
        scene._scene_objects['bs'] = scene.transmitters['bs']
        if "bs" not in scene.objects:
            raise RuntimeError("Base station not found in scene objects")
        print("[DEBUG] Base station added to scene objects")
    
    # Verify RIS
    if not scene.ris or "ris" not in scene.ris:
        raise RuntimeError("RIS not found in scene")
    
    # Verify antenna arrays
    if not hasattr(scene, 'tx_array') or not hasattr(scene, 'rx_array'):
        raise RuntimeError("Scene is missing antenna array configuration")
    
    print("[DEBUG] Scene verification completed successfully")