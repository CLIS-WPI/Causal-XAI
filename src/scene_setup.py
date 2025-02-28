#src/scene_setup.py
import mitsuba
import tensorflow as tf
import sionna
import logging
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray, RadioMaterial
from config import SmartFactoryConfig
import numpy as np
import os
from tensorflow import autograph

logger = logging.getLogger(__name__)

def _debug_object_state(obj, name):
    logger.debug(f"Debugging {name} state:")
    if hasattr(obj, 'position'):
        logger.debug(f"- Position: {obj.position.numpy()}")
    if hasattr(obj, 'orientation'):
        logger.debug(f"- Orientation: {obj.orientation.numpy()}")
    if hasattr(obj, 'dtype'):
        logger.debug(f"- dtype: {obj.dtype}")


@autograph.experimental.do_not_convert
def setup_scene(config: SmartFactoryConfig):
    try:
        logger.debug("=== Scene Configuration ===")
        logger.debug(f"Room dimensions: {config.room_dim}")
        logger.debug(f"Number of AGVs: {config.num_agvs}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency} Hz")

        scene_path = os.path.join(os.path.dirname(__file__), 'factory_scene.xml')
        logger.debug(f"Loading scene from: {scene_path}")
        scene = Scene(scene_path)
        logger.info("Scene loaded from XML")

        # Debug: Log initial scene state after loading XML
        logger.debug("Initial scene state after XML load:")
        logger.debug(f"Objects: {list(scene.objects.keys())}")
        logger.debug(f"Transmitters: {list(scene.transmitters.keys())}")
        logger.debug(f"Receivers: {list(scene.receivers.keys())}")

        # Define radio materials with unique names
        logger.debug("Defining radio materials...")
        concrete = RadioMaterial(
            name="factory_concrete",
            relative_permittivity=5.31,
            conductivity=0.0147,
            scattering_coefficient=0.1,
            xpd_coefficient=0.5
        )
        scene.add(concrete)
        logger.info("Defined radio material: factory_concrete with permittivity=5.31, conductivity=0.0147")

        metal = RadioMaterial(
            name="factory_metal",
            relative_permittivity=1.0,
            conductivity=1e7,
            scattering_coefficient=0.05,
            xpd_coefficient=0.7
        )
        scene.add(metal)
        logger.info("Defined radio material: factory_metal with permittivity=1.0, conductivity=1e7")

        logger.debug("Registered radio materials: " + ", ".join(scene.radio_materials.keys()))

        scene.synthetic_array = True
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)

        logger.debug("\n=== Base Station Configuration ===")
        logger.debug(f"BS position: {config.bs_position}")
        logger.debug(f"BS orientation: {config.bs_orientation}")

        bs = Transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )
        array = PlanarArray(
            num_rows=config.bs_array['num_rows'],
            num_cols=config.bs_array['num_cols'],
            vertical_spacing=config.bs_array.get('vertical_spacing', 0.7),
            horizontal_spacing=config.bs_array.get('horizontal_spacing', 0.5),
            pattern=config.bs_array.get('pattern', "tr38901"),
            polarization=config.bs_array.get('polarization', "VH")
        )
        bs.array = array
        scene.add(bs)
        _debug_object_state(bs, "Base station")

        print("\n=== CRITICAL TRANSMITTER DEBUG ===")
        print(f"Number of transmitters in scene: {len(scene.transmitters)}")
        print(f"Available transmitter keys: {list(scene.transmitters.keys())}")
        if len(scene.transmitters) > 0:
            print(f"First transmitter position: {scene.transmitters['bs'].position}")
        else:
            print("WARNING: No transmitters found in scene!")
        print("===================================\n")

        scene.tx_array = array

        logger.debug("\n=== AGV Configurations ===")
        agv_array_for_scene = None

        for i in range(config.num_agvs):
            logger.debug(f"\nAGV_{i} Configuration:")
            logger.debug(f"Position: {config.agv_positions[i]}")
            logger.debug(f"Orientation: {config.agv_orientations[i]}")
            dist_bs = tf.norm(tf.constant(config.agv_positions[i]) - tf.constant(config.bs_position))
            logger.debug(f"Distance to BS: {dist_bs:.2f} meters")

            rx = Receiver(
                name=f"rx_agv_{i}",  # Changed to avoid conflict with agv_robot_{i}
                position=tf.constant(config.agv_positions[i], dtype=tf.float32),
                orientation=tf.constant(config.agv_orientations[i], dtype=tf.float32)
            )
            rx_array = PlanarArray(
                num_rows=config.agv_array['num_rows'],
                num_cols=config.agv_array['num_cols'],
                vertical_spacing=config.agv_array.get('vertical_spacing', 0.5),
                horizontal_spacing=config.agv_array.get('horizontal_spacing', 0.5),
                pattern=config.agv_array.get('pattern', "tr38901"),
                polarization=config.agv_array.get('polarization', "VH")
            )
            rx.array = rx_array
            logger.debug(f"Adding receiver {rx.name} to scene...")
            scene.add(rx)
            _debug_object_state(rx, f"AGV_{i}")
            if i == 0:
                agv_array_for_scene = rx_array

        # Debug: Log receiver positions after setup
        for rx_name, rx in scene.receivers.items():
            logger.debug(f"Receiver {rx_name} initialized at {rx.position.numpy()}")

        logger.info("\n=== Final AGV Positions in Scene ===")
        for rx in scene.receivers.values():
            logger.info(f"Receiver {rx.name} at {rx.position.numpy()}")

        logger.info("\n=== Scene Objects ===")
        logger.info(f"Scene objects: {scene.objects.keys()}")

        if agv_array_for_scene is not None:
            scene.rx_array = agv_array_for_scene
        else:
            logger.warning("No AGVs were added, so no receiver array is set.")

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
