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

        # Add base station
        try:
            bs_pos = tf.constant(config.bs_position, dtype=tf.float32)
            bs_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            
            bs = Transmitter(
                name="bs",
                position=bs_pos,
                orientation=bs_orientation
            )
            scene.add(bs)
            print("[DEBUG] Base station added successfully")
        except Exception as e:
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
            ris_pos = tf.constant(config.ris_position, dtype=tf.float32)
            ris_orientation = tf.constant(config.ris_orientation, dtype=tf.float32)
            
            print(f"[DEBUG] RIS configuration: position={config.ris_position}, orientation={config.ris_orientation}, elements={config.ris_elements}")
            
            ris = RIS(
                name="ris",
                position=ris_pos,
                orientation=ris_orientation,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                dtype=config.dtype
            )
            scene.add(ris)
            print("[DEBUG] RIS added successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to add RIS: {str(e)}")

        # Add AGVs
        try:
            for i in range(config.num_agvs):
                agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, 0.5], dtype=tf.float32)
                agv_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
                
                rx = Receiver(
                    name=f"agv_{i}",
                    position=agv_pos,
                    orientation=agv_orientation
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
    print("[DEBUG] Verifying scene configuration...")
    print(f"[DEBUG] Available objects: {list(scene.objects.keys())}")
    print(f"[DEBUG] Available transmitters: {list(scene.transmitters.keys())}")
    if hasattr(scene, 'ris'):
        print(f"[DEBUG] Available RIS: {list(scene.ris.keys())}")
    
    # Verify transmitter
    if "bs" not in scene.transmitters:
        raise RuntimeError("Base station not found in scene")
        
    # Verify RIS - check in scene.ris dictionary
    if not scene.ris or "ris" not in scene.ris:
        raise RuntimeError("RIS not found in scene")
        
    # Verify antenna arrays
    if not hasattr(scene, 'tx_array') or not hasattr(scene, 'rx_array'):
        raise RuntimeError("Scene is missing antenna array configuration")
        
    print("[DEBUG] Scene verification completed successfully")