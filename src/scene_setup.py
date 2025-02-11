import tensorflow as tf
import sionna
import logging
from sionna.rt import Scene, RadioMaterial
from scene_manager import SceneManager

# Setup logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _debug_object_state(obj, name):
    """Helper to debug object state"""
    logger.debug(f"Debugging {name} state:")
    
    if hasattr(obj, 'position'):
        logger.debug(f"- Position: {obj.position.numpy()}")
    if hasattr(obj, 'orientation'):
        logger.debug(f"- Orientation: {obj.orientation.numpy()}")
    if hasattr(obj, 'array'):
        logger.debug(f"- Array config: {obj.array}")
    if hasattr(obj, 'dtype'):
        logger.debug(f"- dtype: {obj.dtype}")

def setup_scene(config):
    """Setup the factory scene using Sionna's built-in management"""
    logger.info("Starting scene setup...")
    
    try:
        # Validate config
        required_attrs = ['dtype', 'carrier_frequency', 'bs_position', 'ris_position', 
                        'ris_elements', 'num_agvs', 'agv_height', 'room_dim']
        missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing_attrs:
            raise ValueError(f"Missing required config attributes: {missing_attrs}")

        # Create scene
        scene = Scene(env_filename="__empty__", dtype=config.dtype)
        scene.frequency = tf.cast(config.carrier_frequency, scene.dtype.real_dtype)
        
        # Initialize scene manager
        manager = SceneManager(scene, config)
        
        # Add base station
        bs = manager.add_transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )
        _debug_object_state(bs, "Base station")

        # Add RIS
        ris = manager.add_ris(
            name="ris",
            position=tf.constant(config.ris_position, dtype=tf.float32),
            orientation=tf.constant(config.ris_orientation, dtype=tf.float32),
            num_rows=config.ris_elements[0],
            num_cols=config.ris_elements[1]
        )
        _debug_object_state(ris, "RIS")

        # Add AGVs
        for i in range(config.num_agvs):
            agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, config.agv_height], 
                                dtype=tf.float32)
            rx = manager.add_receiver(
                name=f"agv_{i}",
                position=agv_pos,
                orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            )
            _debug_object_state(rx, f"AGV_{i}")

        # Log final scene state
        logger.info("Scene setup completed successfully")
        logger.info(f"- Transmitters: {len(scene.transmitters)}")
        logger.info(f"- Receivers: {len(scene.receivers)}")
        logger.info(f"- RIS: {len(scene.ris)}")
        logger.info(f"- Objects: {len(scene.objects)}")

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Scene setup failed: {str(e)}") from e