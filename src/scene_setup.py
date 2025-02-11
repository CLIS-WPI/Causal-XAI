import tensorflow as tf
import sionna
import logging
from sionna.rt import Scene, RadioMaterial
from scene_manager import SceneManager
from config import SmartFactoryConfig

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

def setup_scene(config: SmartFactoryConfig):
    """Setup the factory scene with ray tracing"""
    logger.info("Starting scene setup with ray tracing...")
    
    try:
        # Create scene - remove max_depth parameter
        scene = Scene(env_filename="__empty__", dtype=config.dtype)
        
        # Initialize scene manager
        manager = SceneManager(scene, config)
        
        # Add base station
        bs = manager.add_transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )
        _debug_object_state(bs, "Base station")

        # Set transmitter array for the scene
        scene.tx_array = bs.array

        # Add RIS
        ris = manager.add_ris(
            name="ris",
            position=tf.constant(config.ris_position, dtype=tf.float32),
            orientation=tf.constant(config.ris_orientation, dtype=tf.float32)
        )
        _debug_object_state(ris, "RIS")

        # Add AGVs and set receiver array
        agv_array = None
        for i in range(config.num_agvs):
            agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, config.agv_height], 
                                dtype=tf.float32)
            rx = manager.add_receiver(
                name=f"agv_{i}",
                position=agv_pos,
                orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            )
            _debug_object_state(rx, f"AGV_{i}")
            
            # Store array configuration from first AGV
            if i == 0:
                agv_array = rx.array
        
        # Set receiver array for the scene
        scene.rx_array = agv_array

        # Configure ray-tracing parameters
        scene.synthetic_array = True
        
        # Set ray tracing specific parameters
        scene.los = config.ray_tracing['los']
        scene.reflection = config.ray_tracing['reflection']
        scene.diffraction = config.ray_tracing['diffraction']
        scene.scattering = config.ray_tracing['scattering']
        
        # Log final scene state
        logger.info("Scene setup completed successfully")
        logger.info(f"- Transmitters: {len(scene.transmitters)}")
        logger.info(f"- Receivers: {len(scene.receivers)}")
        logger.info(f"- RIS: {len(scene.ris)}")
        logger.info(f"- Objects: {len(scene.objects)}")
        logger.info(f"- Ray tracing enabled with:")
        logger.info(f"  - LOS: {config.ray_tracing['los']}")
        logger.info(f"  - Reflection: {config.ray_tracing['reflection']}")
        logger.info(f"  - Diffraction: {config.ray_tracing['diffraction']}")
        logger.info(f"  - Scattering: {config.ray_tracing['scattering']}")

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Scene setup failed: {str(e)}") from e