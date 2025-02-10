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

def setup_scene(config):
    """Setup the factory scene with transmitters, receivers, and RIS"""
    
    # Debug helper function
    def _debug_object_state(obj, name):
        """Helper to debug object state"""
        logger.debug(f"{name} properties:")
        logger.debug(f"- Position: {obj.position.numpy()}")
        logger.debug(f"- Orientation: {obj.orientation.numpy()}")
        if hasattr(obj, 'array'):
            logger.debug(f"- Array config: {obj.array}")
        logger.debug(f"- dtype: {obj.dtype}")

    try:
        # Validate config prerequisites
        logger.debug("Validating configuration...")
        required_attrs = ['dtype', 'carrier_frequency', 'bs_position', 'ris_position', 
                        'ris_elements', 'num_agvs', 'agv_height']
        missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing_attrs:
            raise ValueError(f"Missing required config attributes: {missing_attrs}")
        
        logger.debug(f"Using dtype: {config.dtype}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency}")

        # Create scene with explicit error checking
        logger.info("Creating new scene...")
        try:
            scene = Scene(env_filename="__empty__", dtype=config.dtype)
            if not hasattr(scene, '_dtype'):
                raise AttributeError("Scene dtype not properly initialized")
            logger.debug(f"Scene created with dtype: {scene._dtype}")
        except Exception as e:
            logger.error(f"Failed to create scene: {str(e)}")
            raise

        # Set frequency with validation
        try:
            scene.frequency = tf.cast(config.carrier_frequency, scene.dtype.real_dtype)
            logger.debug(f"Scene frequency set to {scene.frequency.numpy()} Hz")
        except Exception as e:
            logger.error(f"Failed to set scene frequency: {str(e)}")
            raise

        # Initialize scene manager with validation
        logger.info("Initializing scene manager...")
        try:
            manager = SceneManager(scene, config)
            logger.debug("Scene manager state initialized")
        except Exception as e:
            logger.error(f"Failed to initialize scene manager: {str(e)}")
            raise

        # Add base station with position validation
        logger.info("Adding base station...")
        try:
            bs_position = tf.constant(config.bs_position, dtype=tf.float32)
            bs_orientation = tf.constant(config.bs_orientation, dtype=tf.float32)
            logger.debug(f"BS position: {bs_position.numpy()}")
            logger.debug(f"BS orientation: {bs_orientation.numpy()}")
            
            bs = manager.add_transmitter(
                name="bs",
                position=bs_position,
                orientation=bs_orientation,
                dtype=config.dtype
            )
            _debug_object_state(bs, "Base station")
        except Exception as e:
            logger.error(f"Failed to add base station: {str(e)}")
            raise

        # Add RIS with validation
        logger.info("Adding RIS...")
        try:
            ris_position = tf.constant(config.ris_position, dtype=tf.float32)
            ris_orientation = tf.constant(config.ris_orientation, dtype=tf.float32)
            logger.debug(f"RIS position: {ris_position.numpy()}")
            logger.debug(f"RIS orientation: {ris_orientation.numpy()}")
            logger.debug(f"RIS elements: {config.ris_elements}")
            
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
            logger.error(f"Failed to add RIS: {str(e)}")
            raise

        # Add AGVs with position validation
        logger.info(f"Adding {config.num_agvs} AGVs...")
        try:
            for i in range(config.num_agvs):
                agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, config.agv_height], 
                                    dtype=tf.float32)
                logger.debug(f"AGV_{i} position: {agv_pos.numpy()}")
                
                rx = manager.add_receiver(
                    name=f"agv_{i}",
                    position=agv_pos,
                    orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32),
                    dtype=config.dtype
                )
                _debug_object_state(rx, f"AGV_{i}")
        except Exception as e:
            logger.error(f"Failed to add AGV {i}: {str(e)}")
            raise

        # Add room boundaries with validation
        logger.info("Adding room boundaries...")
        try:
            _add_room_boundaries(scene, config)
            logger.debug("Room boundaries added successfully")
        except Exception as e:
            logger.error(f"Failed to add room boundaries: {str(e)}")
            raise

        # Validate final scene state
        logger.info("Validating scene...")
        try:
            if not manager.validate_scene():
                raise RuntimeError("Scene validation failed")
            
            # Final debug info
            logger.debug(f"Final scene state:")
            logger.debug(f"- Number of transmitters: {len(scene.transmitters)}")
            logger.debug(f"- Number of receivers: {len(scene.receivers)}")
            logger.debug(f"- Number of RIS: {len(scene.ris)}")
            logger.debug(f"- Number of objects: {len(scene.objects)}")
        except Exception as e:
            logger.error(f"Scene validation failed: {str(e)}")
            raise

        logger.info("Scene setup completed successfully")
        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}")
        # Log full stack trace for debugging
        logger.exception("Detailed error trace:")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e

def _add_room_boundaries(self):
    """Add walls, floor and ceiling to the scene"""
    try:
        # Create concrete material
        concrete = RadioMaterial(
            name="concrete",
            relative_permittivity=4.5,
            conductivity=0.01,
            dtype=self._scene.dtype
        )
        
        # Add material to scene
        self._scene.add(concrete)
        
        # Room dimensions
        length, width, height = self._config.room_dim
        
        # Create walls using Transmitter objects (as placeholders for surfaces)
        # This is a workaround since Sionna doesn't have a direct wall object
        
        # Floor
        floor = Transmitter(
            name="floor",
            position=tf.constant([length/2, width/2, 0], dtype=tf.float32),
            orientation=tf.constant([0, 0, 0], dtype=tf.float32),
            dtype=self._scene.dtype
        )
        floor.scene = self._scene
        floor.radio_material = concrete
        self._scene.add(floor)
        
        # Ceiling
        ceiling = Transmitter(
            name="ceiling",
            position=tf.constant([length/2, width/2, height], dtype=tf.float32),
            orientation=tf.constant([0, 0, 0], dtype=tf.float32),
            dtype=self._scene.dtype
        )
        ceiling.scene = self._scene
        ceiling.radio_material = concrete
        self._scene.add(ceiling)
        
        # Walls
        wall_configs = [
            ("wall_front", [length/2, 0, height/2]),
            ("wall_back", [length/2, width, height/2]),
            ("wall_left", [0, width/2, height/2]),
            ("wall_right", [length, width/2, height/2])
        ]
        
        for name, position in wall_configs:
            wall = Transmitter(
                name=name,
                position=tf.constant(position, dtype=tf.float32),
                orientation=tf.constant([0, 0, 0], dtype=tf.float32),
                dtype=self._scene.dtype
            )
            wall.scene = self._scene
            wall.radio_material = concrete
            self._scene.add(wall)
            
        logger.info("Room boundaries added successfully")
            
    except Exception as e:
        logger.error(f"Failed to add room boundaries: {str(e)}")
        raise