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
    try:
        logger.info("Creating new scene...")
        # Create new scene instead of loading
        scene = Scene()
        scene.dtype = config.dtype
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        logger.info("Scene created successfully")

        # Initialize scene manager
        logger.info("Initializing scene manager...")
        manager = SceneManager(scene, config)
        logger.info("Scene manager initialized")

        # Add base station
        logger.info("Adding base station...")
        bs = manager.add_transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32),
            dtype=config.dtype
        )
        logger.info("Base station added successfully")

        # Add RIS
        logger.info("Adding RIS...")
        ris = manager.add_ris(
            name="ris",
            position=tf.constant(config.ris_position, dtype=tf.float32),
            orientation=tf.constant(config.ris_orientation, dtype=tf.float32),
            num_rows=config.ris_elements[0],
            num_cols=config.ris_elements[1],
            dtype=config.dtype
        )
        logger.info("RIS added successfully")

        # Add AGVs
        logger.info(f"Adding {config.num_agvs} AGVs...")
        for i in range(config.num_agvs):
            agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, config.agv_height], dtype=tf.float32)
            manager.add_receiver(
                name=f"agv_{i}",
                position=agv_pos,
                orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32),
                dtype=config.dtype
            )
        logger.info("AGVs added successfully")

        # Add walls and floor
        logger.info("Adding room boundaries...")
        _add_room_boundaries(scene, config)
        logger.info("Room boundaries added")

        # Validate scene
        logger.info("Validating scene...")
        if not manager.validate_scene():
            raise RuntimeError("Scene validation failed")
        logger.info("Scene validation successful")

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e

def _add_room_boundaries(scene, config):
    """Add walls, floor and ceiling to the scene"""
    try:
        # Create concrete material
        concrete = RadioMaterial(
            name="concrete",
            relative_permittivity=4.5,
            conductivity=0.01,
            dtype=config.dtype
        )
        
        # Add to scene
        scene.add(concrete)
        
        # Room dimensions
        length, width, height = config.room_dim
        
        # Add floor
        floor = SceneObject(
            name="floor",
            position=[length/2, width/2, 0],
            size=[length, width, 0.2],
            radio_material=concrete,
            dtype=config.dtype
        )
        scene.add(floor)
        
        # Add ceiling
        ceiling = SceneObject(
            name="ceiling",
            position=[length/2, width/2, height],
            size=[length, width, 0.2],
            radio_material=concrete,
            dtype=config.dtype
        )
        scene.add(ceiling)
        
        # Add walls
        walls = [
            # Front wall
            SceneObject(
                name="wall_front",
                position=[length/2, 0, height/2],
                size=[length, 0.2, height],
                radio_material=concrete,
                dtype=config.dtype
            ),
            # Back wall
            SceneObject(
                name="wall_back",
                position=[length/2, width, height/2],
                size=[length, 0.2, height],
                radio_material=concrete,
                dtype=config.dtype
            ),
            # Left wall
            SceneObject(
                name="wall_left",
                position=[0, width/2, height/2],
                size=[0.2, width, height],
                radio_material=concrete,
                dtype=config.dtype
            ),
            # Right wall
            SceneObject(
                name="wall_right",
                position=[length, width/2, height/2],
                size=[0.2, width, height],
                radio_material=concrete,
                dtype=config.dtype
            )
        ]
        
        for wall in walls:
            scene.add(wall)
            
    except Exception as e:
        logger.error(f"Failed to add room boundaries: {str(e)}")
        raise