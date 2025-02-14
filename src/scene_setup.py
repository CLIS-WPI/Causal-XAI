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
        # Debug print scene configuration
        logger.debug("=== Scene Configuration ===")
        logger.debug(f"Room dimensions: {config.room_dim}")
        logger.debug(f"Number of AGVs: {config.num_agvs}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency} Hz")
        
        # Create scene with only supported initialization parameters
        scene = Scene(
            env_filename="__empty__", 
            dtype=config.dtype
        )
        
        # Set synthetic array property
        scene.synthetic_array = True
        
        # Initialize scene manager
        manager = SceneManager(scene, config)
        
        # Debug print BS configuration
        logger.debug("\n=== Base Station Configuration ===")
        logger.debug(f"BS position: {config.bs_position}")
        logger.debug(f"BS orientation: {config.bs_orientation}")
        
        # Add base station
        bs = manager.add_transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )
        _debug_object_state(bs, "Base station")

        # Set transmitter array for the scene
        scene.tx_array = bs.array
        logger.debug(f"BS array configuration: {bs.array}")

        # Debug print AGV configurations
        logger.debug("\n=== AGV Configurations ===")
        
        # Add AGVs and set receiver array
        agv_array = None
        for i in range(config.num_agvs):
            logger.debug(f"\nAGV_{i} Configuration:")
            logger.debug(f"Position: {config.agv_positions[i]}")
            logger.debug(f"Orientation: {config.agv_orientations[i]}")
            
            # Calculate and log distance to BS
            bs_to_agv = tf.norm(tf.constant(config.agv_positions[i]) - tf.constant(config.bs_position))
            logger.debug(f"Distance to BS: {bs_to_agv:.2f} meters")
            
            agv_pos = tf.constant(config.agv_positions[i], dtype=tf.float32)
            rx = manager.add_receiver(
                name=f"agv_{i}",
                position=agv_pos,
                orientation=tf.constant(config.agv_orientations[i], dtype=tf.float32)
            )
            _debug_object_state(rx, f"AGV_{i}")
            
            if i == 0:
                agv_array = rx.array
                logger.debug(f"AGV array configuration: {rx.array}")

        # Set receiver array for the scene
        scene.rx_array = agv_array

        # Debug print ray tracing configuration
        logger.debug("\n=== Ray Tracing Configuration ===")
        logger.debug(f"Max depth: {config.ray_tracing.get('max_depth', 'Not set')}")
        logger.debug(f"Method: {config.ray_tracing.get('method', 'Not set')}")
        logger.debug(f"Number of samples: {config.ray_tracing.get('num_samples', 'Not set')}")
        
        # Set propagation mechanisms
        scene.los = True
        scene.reflection = True
        scene.diffraction = True
        scene.scattering = True
        
        # Set ray tracing parameters with additional path detection settings
        scene.max_depth = config.ray_tracing['max_depth']
        scene.num_samples = config.ray_tracing['num_samples']
        scene.method = config.ray_tracing['method']
        
        # Set visibility parameters
        scene.test_medium = True  # Enable visibility testing through materials
        scene.delete_duplicates = True  # Remove duplicate paths
        
        # Set frequency
        scene.frequency = config.carrier_frequency
        
        # Log final scene state
        logger.info("\n=== Final Scene State ===")
        logger.info(f"- Transmitters: {len(scene.transmitters)}")
        logger.info(f"- Receivers: {len(scene.receivers)}")
        logger.info(f"- Objects: {len(scene.objects)}")
        logger.info(f"- Ray tracing enabled with:")
        logger.info(f"  - LOS: {scene.los}")
        logger.info(f"  - Reflection: {scene.reflection}")
        logger.info(f"  - Diffraction: {scene.diffraction}")
        logger.info(f"  - Scattering: {scene.scattering}")
        logger.info(f"  - Method: {scene.method}")
        logger.info(f"  - Max depth: {scene.max_depth}")
        logger.info(f"  - Num samples: {scene.num_samples}")
        logger.info(f"  - Test medium: {scene.test_medium}")

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Scene setup failed: {str(e)}") from e