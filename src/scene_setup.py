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
        # Load scene
        scene = load_scene('src/factory_scene.xml', dtype=config.dtype)
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)

        # Initialize scene manager
        manager = SceneManager(scene, config)

        # Add base station
        bs = manager.add_transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32),
            dtype=config.dtype
        )

        # Add RIS
        ris = manager.add_ris(
            name="ris",
            position=tf.constant(config.ris_position, dtype=tf.float32),
            orientation=tf.constant(config.ris_orientation, dtype=tf.float32),
            num_rows=config.ris_elements[0],
            num_cols=config.ris_elements[1],
            dtype=config.dtype
        )

        # Add AGVs
        for i in range(config.num_agvs):
            agv_pos = tf.constant([12.0 - i*4.0, 5.0 + i*10.0, 0.5], dtype=tf.float32)
            manager.add_receiver(
                name=f"agv_{i}",
                position=agv_pos,
                orientation=tf.constant([0.0, 0.0, 0.0], dtype=tf.float32),
                dtype=config.dtype
            )

        # Validate scene
        if not manager.validate_scene():
            raise RuntimeError("Scene validation failed")

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e