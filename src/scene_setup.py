import tensorflow as tf
import sionna
from sionna.rt import RadioMaterial
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
            
            # Debug current scene state
            print("[DEBUG] Scene objects before BS addition:", list(scene.objects.keys()))
            print("[DEBUG] Current object IDs:", [(name, getattr(obj, 'object_id', None)) 
                                                for name, obj in scene.objects.items()])
            
            # Validate current object IDs
            current_ids = [getattr(obj, 'object_id', None) for obj in scene.objects.values()]
            print(f"[DEBUG] Current object IDs in use: {current_ids}")
            
            # Create metal material
            metal_material = RadioMaterial("itu_metal", dtype=config.dtype)

            # Create base station
            bs = Transmitter(
                name="bs",
                position=bs_pos,
                orientation=bs_orientation,
                dtype=config.dtype
            )

            # Add velocity property
            bs.velocity = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

            # Calculate next available object ID
            next_id = max(current_ids + [-1]) + 1 if current_ids else 0
            bs.object_id = next_id
            
            print(f"[DEBUG] Next available object_id: {next_id}")
            print(f"[DEBUG] Base station assigned object_id: {getattr(bs, 'object_id', None)}")

            # Set radio material
            bs.radio_material = metal_material

            # Validate object ID before adding to scene
            if hasattr(bs, 'object_id'):
                print(f"[DEBUG] Validating BS object_id: {bs.object_id}")
                if bs.object_id in current_ids:
                    raise ValueError(f"Duplicate object ID {bs.object_id} detected")
            else:
                raise ValueError("Base station missing object_id attribute")

            # Add to scene
            scene.add(bs)
            
            # Verify successful addition
            print(f"[DEBUG] Base station added successfully at position {bs_pos}")
            print(f"[DEBUG] Base station registered as transmitter: {bs.name in scene.transmitters}")
            print(f"[DEBUG] Base station registered as object: {bs.name in scene.objects}")
            
            # Verify final state
            print(f"[DEBUG] Scene objects after BS addition: {list(scene.objects.keys())}")
            print(f"[DEBUG] Final object IDs: {[(name, getattr(obj, 'object_id', None)) for name, obj in scene.objects.items()]}")
            
            # Validate no duplicate IDs exist
            final_ids = [getattr(obj, 'object_id', None) for obj in scene.objects.values()]
            if len(final_ids) != len(set(final_ids)):
                raise ValueError("Duplicate object IDs detected after BS addition")
            
            # Validate max ID doesn't exceed object count
            max_id = max(final_ids) if final_ids else -1
            if max_id >= len(scene.objects):
                raise ValueError(f"Maximum object ID ({max_id}) exceeds total objects ({len(scene.objects)})")

        except Exception as e:
            print(f"[ERROR] Failed to add base station: {str(e)}")
            print(f"[DEBUG] Current scene state at failure:")
            print(f"[DEBUG] Objects: {list(scene.objects.keys())}")
            print(f"[DEBUG] Object IDs: {[(name, getattr(obj, 'object_id', None)) for name, obj in scene.objects.items()]}")
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
            print(f"[DEBUG] Creating RIS at position {ris_pos} with orientation {ris_orientation}")
            
            # Create RIS instance
            ris = RIS(
                name="ris",
                position=ris_pos,
                orientation=ris_orientation,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                dtype=config.dtype
            )
            print(f"[DEBUG] RIS instance created with dimensions: {config.ris_elements}")
            
            # Set scene reference first (required for proper initialization)
            ris.scene = scene
            print("[DEBUG] Scene reference set for RIS")
            
            # Get or create radio material before setting object ID
            metal_material = scene.get("itu_metal")
            if metal_material is None:
                print("[DEBUG] Creating new itu_metal material")
                metal_material = RadioMaterial("itu_metal", dtype=config.dtype)
                metal_material.scene = scene
                scene.add(metal_material)
            else:
                print("[DEBUG] Using existing itu_metal material")

            # Set radio material for RIS
            ris.radio_material = metal_material
            print(f"[DEBUG] Radio material set for RIS: {getattr(ris.radio_material, 'name', None)}")
            
            # Add RIS to scene (this will automatically handle object ID assignment)
            scene.add(ris)
            print("[DEBUG] RIS added to scene")
            print(f"[DEBUG] RIS object ID after scene addition: {getattr(ris, 'object_id', None)}")

            # Verify RIS addition
            print("\n[DEBUG] Scene state after RIS addition:")
            print(f"[DEBUG] Scene objects: {list(scene.objects.keys())}")
            print(f"[DEBUG] Scene RIS: {list(scene.ris.keys())}")
            print(f"[DEBUG] RIS object ID: {getattr(ris, 'object_id', None)}")
            print(f"[DEBUG] RIS radio material: {getattr(ris.radio_material, 'name', None)}")

            # Calculate total objects and verify IDs
            total_objects = len(scene.objects)  # This already includes RIS and BS
            all_ids = [getattr(obj, 'object_id', None) for obj in scene.objects.values()]
            valid_ids = [id for id in all_ids if id is not None]
            unique_ids = set(valid_ids)
            
            print("\n[DEBUG] Final verification:")
            print(f"[DEBUG] All object IDs: {valid_ids}")
            print(f"[DEBUG] Unique object IDs: {list(unique_ids)}")
            print(f"[DEBUG] Total objects in scene: {total_objects}")
            
            # Validate IDs
            if len(valid_ids) != total_objects:
                raise ValueError(f"Missing object IDs: found {len(valid_ids)} IDs for {total_objects} objects")
            
            if len(valid_ids) != len(unique_ids):
                duplicate_ids = [id for id in valid_ids if valid_ids.count(id) > 1]
                raise ValueError(f"Duplicate object IDs detected: {duplicate_ids}")
            
            max_id = max(valid_ids) if valid_ids else -1
            if max_id >= total_objects:
                raise ValueError(f"Maximum object ID ({max_id}) exceeds total objects ({total_objects})")
            
            print("[DEBUG] All RIS validations passed successfully")
            print(f"[DEBUG] Object ID validation complete: {len(valid_ids)}/{total_objects} objects have valid IDs")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to add RIS: {str(e)}")
            print(f"[DEBUG] Current scene state at failure:")
            print(f"[DEBUG] Objects: {list(scene.objects.keys())}")
            print(f"[DEBUG] Object IDs: {[(name, getattr(obj, 'object_id', None)) for name, obj in scene.objects.items()]}")
            print(f"[DEBUG] RIS state: {getattr(ris, '__dict__', 'Not created')}")
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
    
    # Verify base station
    if "bs" not in scene.transmitters:
        raise RuntimeError("Base station not found in transmitters")
    if "bs" not in scene.objects:
        print("[WARNING] Base station not in scene objects, attempting to add...")
        scene._scene_objects['bs'] = scene.transmitters['bs']
    
    # Verify RIS
    if not hasattr(scene, 'ris') or "ris" not in scene.ris:
        raise RuntimeError("RIS not found in scene.ris")
    if "ris" not in scene.objects:
        print("[WARNING] RIS not in scene objects, attempting to add...")
        scene._scene_objects['ris'] = scene.ris['ris']
    
    # Verify RIS radio material
    ris = scene.ris.get('ris')
    if ris is None or not hasattr(ris, 'radio_material'):
        raise RuntimeError("RIS missing radio material")
    
    # Verify antenna arrays
    if not hasattr(scene, 'tx_array') or not hasattr(scene, 'rx_array'):
        raise RuntimeError("Scene is missing antenna array configuration")
    
    print("[DEBUG] Scene verification completed successfully")