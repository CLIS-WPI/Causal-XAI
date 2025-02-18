#src/scene_setup.py
import tensorflow as tf
import sionna
import logging
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray
from config import SmartFactoryConfig
import numpy as np
import os

logger = logging.getLogger(__name__)

def _debug_object_state(obj, name):
    """Helper to debug object state"""
    logger.debug(f"Debugging {name} state:")
    if hasattr(obj, 'position'):
        logger.debug(f"- Position: {obj.position.numpy()}")
    if hasattr(obj, 'orientation'):
        logger.debug(f"- Orientation: {obj.orientation.numpy()}")
    if hasattr(obj, 'dtype'):
        logger.debug(f"- dtype: {obj.dtype}")

def setup_scene(config: SmartFactoryConfig):
    """
    Sets up the factory scene by:
    - Loading passive geometry (walls, shelves) from factory_scene.xml
    - Adding a base station and AGVs in Python
    - Configuring ray tracing parameters
    """
    logger.info("Starting scene setup with ray tracing...")
    try:
        # Debug prints for the scene configuration
        logger.debug("=== Scene Configuration ===")
        logger.debug(f"Room dimensions: {config.room_dim}")
        logger.debug(f"Number of AGVs: {config.num_agvs}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency} Hz")

        # 1) Load the scene from an XML file that references your PLY files
        xml_path = os.path.join(os.path.dirname(__file__), "factory_scene.xml")
        scene = Scene(
            env_filename=xml_path,
            dtype=config.dtype
        )

        # 2) Optionally enable synthetic array usage
        scene.synthetic_array = True

        # --- Add the Base Station (Transmitter) ---
        logger.debug("\n=== Base Station Configuration ===")
        logger.debug(f"BS position: {config.bs_position}")
        logger.debug(f"BS orientation: {config.bs_orientation}")

        bs = Transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )

        # Create a PlanarArray for the base station
        # In setup_scene function, change:
        array = PlanarArray(
            num_rows=config.bs_array['num_rows'],      # Use dictionary access
            num_cols=config.bs_array['num_cols'],      # Use dictionary access
            vertical_spacing=config.bs_array.get('vertical_spacing', 0.7),
            horizontal_spacing=config.bs_array.get('horizontal_spacing', 0.5),
            pattern=config.bs_array.get('pattern', "tr38901"),
            polarization=config.bs_array.get('polarization', "VH")
        )
        # Attach array to the transmitter
        bs.array = array

        # Add BS to scene
        scene.add(bs)
        _debug_object_state(bs, "Base station")

        # **Important**: For older Sionna, we must set scene.tx_array so it knows
        # which array to use for the transmitter.
        scene.tx_array = array

        # --- Add AGVs (Receivers) ---
        logger.debug("\n=== AGV Configurations ===")
        agv_array_for_scene = None

        for i in range(config.num_agvs):
            logger.debug(f"\nAGV_{i} Configuration:")
            logger.debug(f"Position: {config.agv_positions[i]}")
            logger.debug(f"Orientation: {config.agv_orientations[i]}")

            dist_bs = tf.norm(
                tf.constant(config.agv_positions[i])
                - tf.constant(config.bs_position)
            )
            logger.debug(f"Distance to BS: {dist_bs:.2f} meters")

            rx = Receiver(
                name=f"agv_{i}",
                position=tf.constant(config.agv_positions[i], dtype=tf.float32),
                orientation=tf.constant(config.agv_orientations[i], dtype=tf.float32)
            )

            # Assign a PlanarArray to each AGV's receiver
            rx_array = PlanarArray(
                num_rows=config.agv_array['num_rows'],
                num_cols=config.agv_array['num_cols'],
                vertical_spacing=config.agv_array.get('vertical_spacing', 0.5),
                horizontal_spacing=config.agv_array.get('horizontal_spacing', 0.5),
                pattern=config.agv_array.get('pattern', "tr38901"),
                polarization=config.agv_array.get('polarization', "VH")
            )
            rx.array = rx_array

            # Add AGV to the scene
            scene.add(rx)
            _debug_object_state(rx, f"AGV_{i}")

            # First AGV sets receiver array
            if i == 0:
                agv_array_for_scene = rx_array

        # Log final AGV positions
        logger.info("\n=== Final AGV Positions in Scene ===")
        for rx in scene.receivers.values():
            logger.info(f"Receiver {rx.name} at {rx.position.numpy()}")

        # Log all scene objects
        logger.info("\n=== Scene Objects ===")
        logger.info(f"Scene objects: {scene.objects.keys()}")


        # If we created at least one AGV, set scene.rx_array from the first one
        if agv_array_for_scene is not None:
            scene.rx_array = agv_array_for_scene
        else:
            logger.warning("No AGVs were added, so no receiver array is set.")

        # --- Configure Ray Tracing Parameters ---
        logger.debug("\n=== Ray Tracing Configuration ===")
        logger.debug(f"Max depth: {config.ray_tracing.get('max_depth', 'Not set')}")
        logger.debug(f"Method: {config.ray_tracing.get('method', 'Not set')}")
        logger.debug(f"Number of samples: {config.ray_tracing.get('num_samples', 'Not set')}")

        scene.los = True
        scene.reflection = True
        scene.diffraction = True
        scene.scattering = True

        scene.max_depth = config.ray_tracing['max_depth']
        scene.num_samples = config.ray_tracing['num_samples']
        scene.method = config.ray_tracing['method']

        scene.test_medium = True
        scene.delete_duplicates = True
        scene.frequency = config.carrier_frequency

        # --- Log Final Scene State ---
        logger.info("\n=== Final Scene State ===")
        logger.info(f"- Transmitters: {len(scene.transmitters)}")
        logger.info(f"- Receivers: {len(scene.receivers)}")
        logger.info(f"- Objects: {len(scene.objects)}")
        logger.info("Ray tracing enabled with:")
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

def verify_geometry(scene):
    """Verify that scene contains expected geometry"""
    logger.info("Verifying scene geometry...")
    
    # Check number of objects
    logger.info(f"Number of objects in scene: {len(scene.objects)}")
    
    # List all objects
    for obj_name, obj in scene.objects.items():  # Access both name and object
        logger.info(f"Object name: {obj_name}")
        if hasattr(obj, 'vertices'):
            logger.info(f"  - Vertices: {len(obj.vertices)}")
        if hasattr(obj, 'faces'):
            logger.info(f"  - Faces: {len(obj.faces)}")
        if hasattr(obj, 'material'):
            logger.info(f"  - Material: {obj.material}")
            
    # List loaded materials
    logger.info("\nAvailable materials:")
    for mat_name in scene.radio_materials:
        logger.info(f"  - {mat_name}")
        
    # Verify specific objects are present
    expected_objects = [
        'floor', 'ceiling',
        'wall_xp', 'wall_xm', 'wall_yp', 'wall_ym',
        'shelf_0', 'shelf_1', 'shelf_2', 'shelf_3', 'shelf_4'
    ]
    
    missing_objects = [obj for obj in expected_objects if obj not in scene.objects]
    if missing_objects:
        logger.warning(f"Missing expected objects: {missing_objects}")
    else:
        logger.info("All expected objects are present in the scene")

def verify_los_paths(scene):
    """
    Check and log basic LOS path info from the base station to each receiver.
    This does NOT handle shelf blocking in older Sionna versions because geometry
    is loaded from XML. The core ray tracer will do LOS computations.
    """
    logger = logging.getLogger(__name__)
    bs_pos = scene.transmitters['bs'].position

    logger.debug(f"\n=== LOS Path Verification ===")
    logger.debug(f"BS Position: {bs_pos.numpy()}")

    for name, rx in scene.receivers.items():
        rx_pos = rx.position
        distance = tf.norm(rx_pos - bs_pos)
        logger.debug(f"\nChecking {name}:")
        logger.debug(f"- Position: {rx_pos.numpy()}")
        logger.debug(f"- Distance to BS: {distance.numpy():.2f}m")

        # Calculate vertical angle
        height_diff = bs_pos[2] - rx_pos[2]
        horizontal_dist = tf.norm(rx_pos[:2] - bs_pos[:2])
        vertical_angle = tf.math.atan2(height_diff, horizontal_dist) * 180.0 / np.pi
        logger.debug(f"- Vertical angle: {vertical_angle.numpy():.1f}°")

        if height_diff > 0:
            logger.debug("Likely clear LOS above the AGV")
        else:
            logger.warning("Possibly blocked (receiver is higher than or level with the BS)")
