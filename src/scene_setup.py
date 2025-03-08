#src/scene_setup.py
import mitsuba
import tensorflow as tf
import sionna
import logging
import os
import numpy as np
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray, RadioMaterial
from config import SmartFactoryConfig

logger = logging.getLogger(__name__)

def _debug_object_state(obj, name):
    logger.debug(f"Debugging {name} state:")
    if hasattr(obj, 'position'):
        logger.debug(f"- Position: {obj.position.numpy()}")
    if hasattr(obj, 'orientation'):
        logger.debug(f"- Orientation: {obj.orientation.numpy()}")
    if hasattr(obj, 'dtype'):
        logger.debug(f"- dtype: {obj.dtype}")

def setup_scene(config: SmartFactoryConfig):
    """Set up the smart factory scene using PLY files and configuration parameters."""
    try:
        # Set Mitsuba variant
        mitsuba.set_variant('cuda_ad_rgb')
        logger.debug(f"Mitsuba variant set to: {mitsuba.variant()}")

        # Create empty Sionna scene
        scene = Scene()
        logger.debug("=== Scene Configuration ===")
        logger.debug(f"Room dimensions: {config.room_dim}")
        logger.debug(f"Number of AGVs: {config.num_agvs}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency} Hz")

        # Define radio materials from config
        concrete_material = RadioMaterial(
            name="factory_concrete",
            relative_permittivity=config.materials['concrete']['relative_permittivity'],
            conductivity=config.materials['concrete']['conductivity'],
            scattering_coefficient=config.materials['concrete']['scattering_coefficient'],
            xpd_coefficient=config.materials['concrete']['xpd_coefficient']
        )
        metal_material = RadioMaterial(
            name="factory_metal",
            relative_permittivity=config.materials['metal']['relative_permittivity'],
            conductivity=config.materials['metal']['conductivity'],
            scattering_coefficient=config.materials['metal']['scattering_coefficient'],
            xpd_coefficient=config.materials['metal']['xpd_coefficient']
        )
        scene.add(concrete_material)
        scene.add(metal_material)
        logger.info(f"Defined radio materials: factory_concrete "
                    f"(permittivity={config.materials['concrete']['relative_permittivity']}, "
                    f"conductivity={config.materials['concrete']['conductivity']}), "
                    f"factory_metal (permittivity={config.materials['metal']['relative_permittivity']}, "
                    f"conductivity={config.materials['metal']['conductivity']})")

        # Load PLY files using Mitsuba and add to Sionna scene
        meshes_dir = os.path.join(os.path.dirname(__file__), "meshes")
        if not os.path.exists(meshes_dir):
            logger.error(f"Meshes directory not found: {meshes_dir}. Run sionna_ply_generator.py first.")
            raise FileNotFoundError(f"Meshes directory not found: {meshes_dir}")

        material_map = {}
        for ply_file in os.listdir(meshes_dir):
            if ply_file.endswith(".ply"):
                full_path = os.path.join(meshes_dir, ply_file)
                name = ply_file[:-4]  # Remove .ply extension
                material_name = ("factory_concrete" if "wall" in name or "floor" in name or "ceiling" in name
                                 else "factory_metal")
                
                # Load PLY file as a Mitsuba shape
                shape = mitsuba.load_dict({
                    'type': 'ply',
                    'filename': full_path,
                    'to_world': mitsuba.ScalarTransform4f(),  # Identity transform
                    'bsdf': {'type': 'null'}  # No BSDF needed for ray tracing
                })
                
                # Add the shape to the Sionna scene
                scene.add(shape)
                # Store material mapping (since we can't set it directly)
                material_map[name] = material_name
                logger.debug(f"Loaded {ply_file} as {name} with intended material {material_name}")

        # Debug scene objects
        logger.debug(f"Objects loaded: {list(scene.objects.keys())}")
        logger.debug(f"Material mapping: {material_map}")

        # Set scene frequency
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        scene.synthetic_array = True

        # Add base station (transmitter)
        logger.debug("\n=== Base Station Configuration ===")
        logger.debug(f"BS position: {config.bs_position}")
        logger.debug(f"BS orientation: {config.bs_orientation}")
        bs = Transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )
        bs_array = PlanarArray(
            num_rows=config.bs_array['num_rows'],
            num_cols=config.bs_array['num_cols'],
            vertical_spacing=config.bs_array.get('vertical_spacing', 0.7),
            horizontal_spacing=config.bs_array.get('horizontal_spacing', 0.5),
            pattern=config.bs_array.get('pattern', "tr38901"),
            polarization=config.bs_array.get('polarization', "VH")
        )
        bs.array = bs_array
        scene.add(bs)
        _debug_object_state(bs, "Base station")
        scene.tx_array = bs_array

        # Debug transmitters
        print("\n=== CRITICAL TRANSMITTER DEBUG ===")
        print(f"Number of transmitters in scene: {len(scene.transmitters)}")
        print(f"Available transmitter keys: {list(scene.transmitters.keys())}")
        if 'bs' in scene.transmitters:
            print(f"BS position: {scene.transmitters['bs'].position.numpy()}")
        else:
            print("WARNING: No base station transmitter found!")
        print("===================================\n")

        # Add AGVs (receivers)
        logger.debug("\n=== AGV Configurations ===")
        agv_array_for_scene = None
        for i in range(config.num_agvs):
            agv_id = f"agv_{i}"
            initial_pos = config.agv_trajectories[f"agv_{i+1}"][0]
            logger.debug(f"\nAGV_{i} Configuration:")
            logger.debug(f"Initial Position: {initial_pos}")
            logger.debug(f"Orientation: {config.agv_orientations[i]}")
            dist_bs = tf.norm(tf.constant(initial_pos, dtype=tf.float32) - tf.constant(config.bs_position, dtype=tf.float32))
            logger.debug(f"Distance to BS: {dist_bs:.2f} meters")

            rx = Receiver(
                name=f"rx_{agv_id}",
                position=tf.constant(initial_pos, dtype=tf.float32),
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
            scene.add(rx)
            _debug_object_state(rx, f"AGV_{i}")
            if i == 0:
                agv_array_for_scene = rx_array

        logger.info("\n=== Final AGV Positions in Scene ===")
        for rx_name, rx in scene.receivers.items():
            logger.info(f"Receiver {rx_name} at {rx.position.numpy()}")

        if agv_array_for_scene is not None:
            scene.rx_array = agv_array_for_scene
        else:
            logger.warning("No AGVs were added, so no receiver array is set.")

        # Configure ray tracing
        logger.debug("\n=== Ray Tracing Configuration ===")
        logger.debug(f"Max depth: {config.ray_tracing['max_depth']}")
        logger.debug(f"Method: {config.ray_tracing['method']}")
        logger.debug(f"Number of samples: {config.ray_tracing['num_samples']}")
        scene.configure_ray_tracing(
            los=True,
            reflection=True,
            diffraction=True,
            scattering=True,
            max_depth=config.ray_tracing['max_depth'],
            num_samples=config.ray_tracing['num_samples'],
            method=config.ray_tracing['method']
        )
        scene.test_medium = True
        scene.delete_duplicates = True

        # Final scene state
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
        logger.info("Scene setup completed")

        # Warning about material assignment
        logger.warning("Material assignment is not directly supported in this version of Sionna. "
                       "Objects added without explicit material binding; ray tracing may use default material.")

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Scene setup failed: {str(e)}") from e

def verify_geometry(scene):
    """Verify that scene contains expected geometry."""
    logger.info("Verifying scene geometry...")
    logger.info(f"Number of objects in scene: {len(scene.objects)}")

    for obj_name, obj in scene.objects.items():
        logger.info(f"Object name: {obj_name}")
        if hasattr(obj, 'vertices'):
            logger.info(f"  - Vertices: {len(obj.vertices)}")
        if hasattr(obj, 'faces'):
            logger.info(f"  - Faces: {len(obj.faces)}")
        if hasattr(obj, 'material'):
            logger.info(f"  - Material: {obj.material.name}")
        else:
            logger.warning(f"  - No material assigned to {obj_name}")

    logger.info("\nAvailable materials:")
    for mat_name in scene.radio_materials:
        logger.info(f"  - {mat_name}")

    expected_objects = [
        'floor', 'ceiling', 'wall_xp', 'wall_xm', 'wall_yp', 'wall_ym',
        'shelf_0', 'shelf_1', 'shelf_2', 'shelf_3', 'shelf_4',
        'agv_robot_0', 'agv_robot_1', 'base_station'
    ]
    missing_objects = [obj for obj in expected_objects if obj not in scene.objects]
    if missing_objects:
        logger.warning(f"Missing expected objects: {missing_objects}")
    else:
        logger.info("All expected objects are present in the scene")

def verify_los_paths(scene):
    """Check basic LOS path info from BS to each receiver."""
    logger.debug("\n=== LOS Path Verification ===")
    if 'bs' not in scene.transmitters:
        logger.error("Base station transmitter not found!")
        return
    bs_pos = scene.transmitters['bs'].position
    logger.debug(f"BS Position: {bs_pos.numpy()}")

    for name, rx in scene.receivers.items():
        rx_pos = rx.position
        distance = tf.norm(rx_pos - bs_pos)
        logger.debug(f"\nChecking {name}:")
        logger.debug(f"- Position: {rx_pos.numpy()}")
        logger.debug(f"- Distance to BS: {distance.numpy():.2f}m")

        height_diff = bs_pos[2] - rx_pos[2]
        horizontal_dist = tf.norm(rx_pos[:2] - bs_pos[:2])
        vertical_angle = tf.math.atan2(height_diff, horizontal_dist) * 180.0 / np.pi
        logger.debug(f"- Vertical angle: {vertical_angle.numpy():.1f}°")

        if height_diff > 0:
            logger.debug("Likely clear LOS above the AGV")
        else:
            logger.warning("Possibly blocked (receiver is higher than or level with the BS)")