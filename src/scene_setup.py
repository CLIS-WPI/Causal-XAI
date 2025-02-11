import tensorflow as tf
import sionna
import logging
from sionna.rt import RadioMaterial
from sionna.rt import (
    load_scene, Scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial, SceneObject
)
from scene_manager import SceneManager
print(f"[DEBUG] Using Sionna version: {sionna.__version__}")

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def _debug_object_state(obj, name):
    """Helper to debug object state with detailed logs and safe attribute checking."""
    print(f"[DEBUG PRINT] Debugging {name} state...")
    logger.debug(f"{name} properties:")
    
    # Position check
    try:
        if hasattr(obj, 'position'):
            pos = obj.position.numpy() if hasattr(obj.position, 'numpy') else obj.position
            logger.debug(f"- Position: {pos}")
            print(f"[DEBUG PRINT] {name} position: {pos}")
    except Exception as e:
        logger.debug(f"- Position: Unable to get position: {str(e)}")
        print(f"[DEBUG PRINT] Error getting position: {str(e)}")

    # Orientation check
    try:
        if hasattr(obj, 'orientation'):
            orient = obj.orientation.numpy() if hasattr(obj.orientation, 'numpy') else obj.orientation
            logger.debug(f"- Orientation: {orient}")
            print(f"[DEBUG PRINT] {name} orientation: {orient}")
    except Exception as e:
        logger.debug(f"- Orientation: Unable to get orientation: {str(e)}")
        print(f"[DEBUG PRINT] Error getting orientation: {str(e)}")

    # Array configuration check
    try:
        if hasattr(obj, 'array'):
            logger.debug(f"- Array config: {obj.array}")
            print(f"[DEBUG PRINT] {name} array config: {obj.array}")
            if hasattr(obj.array, 'num_rows'):
                print(f"[DEBUG PRINT] Array rows: {obj.array.num_rows}")
            if hasattr(obj.array, 'num_cols'):
                print(f"[DEBUG PRINT] Array columns: {obj.array.num_cols}")
    except Exception as e:
        logger.debug(f"- Array config: Unable to get array config: {str(e)}")
        print(f"[DEBUG PRINT] Error getting array config: {str(e)}")

    # Safe dtype check
    try:
        if hasattr(obj, '_dtype'):
            logger.debug(f"- dtype: {obj._dtype}")
            print(f"[DEBUG PRINT] {name} _dtype: {obj._dtype}")
        elif hasattr(obj, 'dtype'):
            logger.debug(f"- dtype: {obj.dtype}")
            print(f"[DEBUG PRINT] {name} dtype: {obj.dtype}")
        else:
            logger.debug("- dtype: Not available")
            print(f"[DEBUG PRINT] {name} dtype not available")
    except Exception as e:
        logger.debug(f"- dtype: Unable to get dtype: {str(e)}")
        print(f"[DEBUG PRINT] Error getting dtype: {str(e)}")

    print(f"[DEBUG PRINT] {name} memory address: {hex(id(obj))}")
    print(f"[DEBUG PRINT] Finished debugging {name} state")
    
def _add_room_boundaries(scene, config):
    """Add walls, floor and ceiling to the scene with proper material handling."""
    print("[DEBUG PRINT] Starting room boundaries setup...")
    logger.debug("Starting room boundaries setup")
    
    try:
        # Check if concrete material already exists in scene
        concrete_name = "concrete"
        print(f"[DEBUG PRINT] Checking for existing {concrete_name} material...")
        
        if concrete_name in scene.radio_materials:
            print(f"[DEBUG PRINT] Reusing existing {concrete_name} material")
            logger.debug(f"Reusing existing {concrete_name} material")
            concrete = scene.radio_materials[concrete_name]
        else:
            print(f"[DEBUG PRINT] Creating new {concrete_name} material...")
            logger.debug(f"Creating new {concrete_name} material")
            concrete = RadioMaterial(
                name=concrete_name,
                relative_permittivity=4.5,
                conductivity=0.01,
                dtype=scene.dtype
            )
            print(f"[DEBUG PRINT] {concrete_name} material created at {hex(id(concrete))}")
            
            # Add material to scene
            print(f"[DEBUG PRINT] Adding {concrete_name} material to scene...")
            scene.add(concrete)
            print(f"[DEBUG PRINT] {concrete_name} material added successfully")
        
        # Room dimensions validation and setup
        try:
            length, width, height = config.room_dim
            if any(dim <= 0 for dim in [length, width, height]):
                raise ValueError("Room dimensions must be positive")
            print(f"[DEBUG PRINT] Room dimensions: {length}x{width}x{height}")
            logger.debug(f"Room dimensions set to {length}x{width}x{height}")
        except Exception as dim_error:
            print(f"[DEBUG PRINT] Error in room dimensions setup: {str(dim_error)}")
            logger.error(f"Room dimensions error: {str(dim_error)}")
            raise
        
        # Define surfaces with their properties
        surfaces = [
            ("floor", [length/2, width/2, 0]),
            ("ceiling", [length/2, width/2, height]),
            ("wall_front", [length/2, 0, height/2]),
            ("wall_back", [length/2, width, height/2]),
            ("wall_left", [0, width/2, height/2]),
            ("wall_right", [length, width/2, height/2])
        ]
        
        # Create all surfaces
        print("[DEBUG PRINT] Creating room boundaries...")
        logger.debug("Creating room boundaries")
        
        for name, position in surfaces:
            try:
                print(f"[DEBUG PRINT] Creating {name}...")
                logger.debug(f"Creating {name}")
                
                # Convert position to tensor
                pos = tf.constant(position, dtype=tf.float32)
                print(f"[DEBUG PRINT] Position for {name}: {pos.numpy()}")
                
                # Create surface
                surface = Transmitter(
                    name=name,
                    position=pos,
                    orientation=tf.constant([0, 0, 0], dtype=tf.float32),
                    dtype=scene.dtype
                )
                
                # Set scene reference
                print(f"[DEBUG PRINT] Setting scene reference for {name}")
                surface.scene = scene
                
                # Set material
                print(f"[DEBUG PRINT] Assigning material to {name}")
                surface.radio_material = concrete
                
                # Add to scene
                print(f"[DEBUG PRINT] Adding {name} to scene")
                scene.add(surface)
                
                print(f"[DEBUG PRINT] Successfully added {name} at position: {pos.numpy()}")
                logger.debug(f"Added {name} at position {pos.numpy()}")
                
            except Exception as surface_error:
                print(f"[DEBUG PRINT] Error creating {name}: {str(surface_error)}")
                logger.error(f"Failed to create {name}: {str(surface_error)}")
                raise
        
        print("[DEBUG PRINT] Room boundaries setup completed successfully")
        logger.info("Room boundaries setup completed successfully")
        
    except Exception as e:
        print(f"[DEBUG PRINT] Critical error in room boundaries setup: {str(e)}")
        print("[DEBUG PRINT] Stack trace:")
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to setup room boundaries: {str(e)}")
        raise RuntimeError(f"Room boundaries setup failed: {str(e)}") from e
    
    finally:
        print("[DEBUG PRINT] Room boundaries setup process finished")
        logger.debug("Room boundaries setup process finished")

def setup_scene(config):
    """Setup the factory scene with transmitters, receivers, and RIS."""
    print("[DEBUG PRINT] Starting scene setup...")
    logger.debug("Starting scene setup...")
    
    try:
        # Validate config prerequisites
        print("[DEBUG PRINT] Validating configuration...")
        required_attrs = ['dtype', 'carrier_frequency', 'bs_position', 'ris_position', 
                        'ris_elements', 'num_agvs', 'agv_height', 'room_dim']
        missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing_attrs:
            error_msg = f"Missing required config attributes: {missing_attrs}"
            print(f"[DEBUG PRINT] Configuration error: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"[DEBUG PRINT] Using dtype: {config.dtype}")
        print(f"[DEBUG PRINT] Carrier frequency: {config.carrier_frequency}")
        logger.debug(f"Using dtype: {config.dtype}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency}")

        # Create scene with explicit error checking
        print("[DEBUG PRINT] Creating new scene...")
        try:
            scene = Scene(env_filename="__empty__", dtype=config.dtype)
            if not hasattr(scene, '_dtype'):
                raise AttributeError("Scene dtype not properly initialized")
            print(f"[DEBUG PRINT] Scene created with dtype: {scene._dtype}")
            logger.debug(f"Scene created with dtype: {scene._dtype}")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to create scene: {str(e)}")
            logger.error(f"Failed to create scene: {str(e)}")
            raise

        # Set frequency with validation
        try:
            print("[DEBUG PRINT] Setting scene frequency...")
            scene.frequency = tf.cast(config.carrier_frequency, scene.dtype.real_dtype)
            print(f"[DEBUG PRINT] Scene frequency set to {scene.frequency.numpy()} Hz")
            logger.debug(f"Scene frequency set to {scene.frequency.numpy()} Hz")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to set scene frequency: {str(e)}")
            logger.error(f"Failed to set scene frequency: {str(e)}")
            raise

        # Initialize scene manager with validation
        print("[DEBUG PRINT] Initializing scene manager...")
        try:
            manager = SceneManager(scene, config)
            print("[DEBUG PRINT] Scene manager initialized successfully")
            logger.debug("Scene manager state initialized")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to initialize scene manager: {str(e)}")
            logger.error(f"Failed to initialize scene manager: {str(e)}")
            raise

        # Add base station with detailed debug and timeout
        print("[DEBUG PRINT] Adding base station...")
        try:
            bs_position = tf.constant(config.bs_position, dtype=tf.float32)
            bs_orientation = tf.constant(config.bs_orientation, dtype=tf.float32)
            print(f"[DEBUG PRINT] BS position: {bs_position.numpy()}")
            print(f"[DEBUG PRINT] BS orientation: {bs_orientation.numpy()}")
            
            bs = manager.add_transmitter(
                name="bs",
                position=bs_position,
                orientation=bs_orientation,
                dtype=config.dtype
            )
            _debug_object_state(bs, "Base station")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to add base station: {str(e)}")
            logger.error(f"Failed to add base station: {str(e)}")
            raise

        # Add RIS with detailed debug and fallback mechanism
        print("[DEBUG PRINT] Adding RIS...")
        try:
            ris_position = tf.constant(config.ris_position, dtype=tf.float32)
            ris_orientation = tf.constant(config.ris_orientation, dtype=tf.float32)
            print(f"[DEBUG PRINT] RIS position: {ris_position.numpy()}")
            print(f"[DEBUG PRINT] RIS orientation: {ris_orientation.numpy()}")
            print(f"[DEBUG PRINT] RIS elements: {config.ris_elements}")
            
            ris = manager.add_ris(
                name="ris",
                position=ris_position,
                orientation=ris_orientation,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                dtype=config.dtype
            )
            _debug_object_state(ris, "RIS")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to add RIS: {str(e)}")
            print("[DEBUG PRINT] Attempting fallback RIS configuration...")
            logger.error(f"Failed to add RIS: {str(e)}")
            logger.warning("Attempting fallback RIS configuration...")
            try:
                ris = manager.add_ris(
                    name="fallback_ris",
                    position=ris_position,
                    orientation=ris_orientation,
                    num_rows=4,
                    num_cols=4,
                    dtype=config.dtype
                )
                print("[DEBUG PRINT] Fallback RIS added successfully")
                logger.debug("Fallback RIS added successfully")
            except Exception as fallback_e:
                print(f"[DEBUG PRINT] Fallback RIS configuration failed: {fallback_e}")
                logger.error(f"Fallback RIS configuration failed: {fallback_e}")
                raise

        # Add AGVs with enhanced logging and error handling
        agv_count = config.num_agvs
        print(f"[DEBUG PRINT] Adding {agv_count} AGVs...")
        try:
            for i in range(agv_count):
                print(f"[DEBUG PRINT] Adding AGV_{i}...")
                agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, config.agv_height], 
                                    dtype=tf.float32)
                print(f"[DEBUG PRINT] AGV_{i} position: {agv_pos.numpy()}")
                
                rx = manager.add_receiver(
                    name=f"agv_{i}",
                    position=agv_pos,
                    orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32),
                    dtype=config.dtype
                )
                _debug_object_state(rx, f"AGV_{i}")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to add AGV {i}: {str(e)}")
            logger.error(f"Failed to add AGV {i}: {str(e)}")
            raise

        # Add room boundaries with validation
        print("[DEBUG PRINT] Adding room boundaries...")
        try:
            _add_room_boundaries(scene, config)
            print("[DEBUG PRINT] Room boundaries added successfully")
            logger.debug("Room boundaries added successfully")
        except Exception as e:
            print(f"[DEBUG PRINT] Failed to add room boundaries: {str(e)}")
            logger.error(f"Failed to add room boundaries: {str(e)}")
            raise

        # Validate final scene state
        print("[DEBUG PRINT] Validating final scene state...")
        try:
            if not manager.validate_scene():
                raise RuntimeError("Scene validation failed")
            
            # Final debug info
            print("[DEBUG PRINT] Final scene state:")
            print(f"[DEBUG PRINT] - Number of transmitters: {len(scene.transmitters)}")
            print(f"[DEBUG PRINT] - Number of receivers: {len(scene.receivers)}")
            print(f"[DEBUG PRINT] - Number of RIS: {len(scene.ris)}")
            print(f"[DEBUG PRINT] - Number of objects: {len(scene.objects)}")
            
            logger.debug("Final scene state:")
            logger.debug(f"- Number of transmitters: {len(scene.transmitters)}")
            logger.debug(f"- Number of receivers: {len(scene.receivers)}")
            logger.debug(f"- Number of RIS: {len(scene.ris)}")
            logger.debug(f"- Number of objects: {len(scene.objects)}")
        except Exception as e:
            print(f"[DEBUG PRINT] Scene validation failed: {str(e)}")
            logger.error(f"Scene validation failed: {str(e)}")
            raise

        print("[DEBUG PRINT] Scene setup completed successfully")
        logger.info("Scene setup completed successfully")
        return scene

    except Exception as e:
        print(f"[DEBUG PRINT] Scene setup failed: {str(e)}")
        print("[DEBUG PRINT] Stack trace:")
        import traceback
        traceback.print_exc()
        logger.error(f"Scene setup failed: {str(e)}")
        logger.exception("Detailed error trace:")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e