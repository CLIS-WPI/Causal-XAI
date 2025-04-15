#src/scene_setup.py
# Core libraries
import mitsuba
import tensorflow as tf
import logging
import os
import numpy as np

# Sionna imports
from sionna.rt.scenes import Scene
from sionna.rt.components import Transmitter, Receiver, SceneObject
from sionna.rt.antenna import PlanarArray
from sionna.rt.materials import RadioMaterial

# Local imports
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
        scene = Scene(dtype=tf.complex64)
        if not hasattr(scene, '_dtype'):
            scene._dtype = tf.complex64

        logger.debug("=== Scene Configuration ===")
        logger.debug(f"Room dimensions: {config.room_dim}")
        logger.debug(f"Number of AGVs: {config.num_agvs}")
        logger.debug(f"Carrier frequency: {config.carrier_frequency} Hz")

        # Define radio materials
        scene.add(RadioMaterial(
            name="factory_concrete",
            relative_permittivity=config.materials['concrete']['relative_permittivity'],
            conductivity=config.materials['concrete']['conductivity'],
            scattering_coefficient=config.materials['concrete'].get('scattering_coefficient', 0.7),
            xpd_coefficient=config.materials['concrete'].get('xpd_coefficient', 8.0)
        ))
        scene.add(RadioMaterial(
            name="factory_metal",
            relative_permittivity=config.materials['metal']['relative_permittivity'],
            conductivity=config.materials['metal']['conductivity'],
            scattering_coefficient=config.materials['metal'].get('scattering_coefficient', 0.3),
            xpd_coefficient=config.materials['metal'].get('xpd_coefficient', 15.0)
        ))
        logger.info(f"Defined radio materials: factory_concrete "
                    f"(permittivity={config.materials['concrete']['relative_permittivity']}, "
                    f"conductivity={config.materials['concrete']['conductivity']}), "
                    f"factory_metal (permittivity={config.materials['metal']['relative_permittivity']}, "
                    f"conductivity={config.materials['metal']['conductivity']})")

        # Build Mitsuba XML from PLY files
        meshes_dir = os.path.join(os.path.dirname(__file__), "meshes")
        if not os.path.exists(meshes_dir):
            logger.error(f"Meshes directory not found: {meshes_dir}. Run sionna_ply_generator.py first.")
            raise FileNotFoundError(f"Meshes directory not found: {meshes_dir}")

        xml_content = '<?xml version="1.0"?>\n<scene version="3.0.0">\n'
        material_map = {}
        for idx, ply_file in enumerate(os.listdir(meshes_dir)):
            if ply_file.endswith(".ply"):
                full_path = os.path.join(meshes_dir, ply_file)
                name = ply_file[:-4]
                material_name = "factory_concrete" if "wall" in name or "floor" in name or "ceiling" in name else "factory_metal"

                xml_content += f'    <shape type="ply" id="{name}">\n'
                xml_content += f'        <string name="filename" value="{full_path}"/>\n'
                xml_content += '        <bsdf type="null"/>\n'
                xml_content += '    </shape>\n'
                material_map[name] = material_name
                logger.debug(f"Added {ply_file} to XML with ID {name} and intended material {material_name}")
        xml_content += '</scene>'

        # Write temporary XML file
        temp_xml_path = os.path.join(meshes_dir, "temp_scene.xml")
        with open(temp_xml_path, 'w') as f:
            f.write(xml_content)

        # Load Mitsuba scene and set it
        mitsuba_scene = mitsuba.load_file(temp_xml_path)
        scene._scene = mitsuba_scene
        logger.debug(f"Loaded Mitsuba scene from {temp_xml_path} and set scene._scene")

        # Clean up temporary file
        os.remove(temp_xml_path)

        # Manually populate scene.objects from mitsuba_scene shapes
        scene.objects.clear()  # Ensure it’s empty before adding
        for shape in scene._scene.shapes():
            name = shape.id()
            if name:
                scene_obj = SceneObject(name, scene)  # Minimal args
                scene_obj._shape = shape  # Set shape manually
                scene.objects[name] = scene_obj
                logger.debug(f"Registered object {name} in scene.objects")

        # Assign RadioMaterial to scene objects
        for obj_name, obj in scene.objects.items():
            base_name = obj_name.split(':')[0]  # Handle Mitsuba suffixes
            if base_name in material_map:
                obj.radio_material = material_map[base_name]
                logger.debug(f"Assigned {material_map[base_name]} to object {obj_name}")
            else:
                logger.warning(f"No material mapping for {obj_name}, defaulting to factory_concrete")
                obj.radio_material = "factory_concrete"

        logger.debug(f"Objects in scene: {list(scene.objects.keys())}")
        logger.debug(f"Material mapping applied: {material_map}")

        # Set scene properties
        scene.synthetic_array = True
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)

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
        scene.tx_array = bs_array
        _debug_object_state(bs, "Base station")

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
        scene.los = True
        scene.reflection = True
        scene.diffraction = True
        scene.scattering = True
        scene.max_depth = config.ray_tracing['max_depth']
        scene.num_samples = config.ray_tracing['num_samples']
        scene.method = config.ray_tracing['method']
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

        return scene

    except Exception as e:
        logger.error(f"Scene setup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Scene setup failed: {str(e)}") from e

def verify_geometry(scene):
    """Verify that scene contains expected geometry."""
    logger.info("Verifying scene geometry...")
    logger.info(f"Number of objects in scene: {len(scene.objects)}")

    for obj_name, obj in scene.objects.items():
        logger.info(f"Object: {obj_name}")
        if hasattr(obj, 'radio_material'):
            logger.info(f"  - Material: {obj.radio_material.name}")
        else:
            logger.warning(f"  - No RadioMaterial assigned")
        if hasattr(obj, 'vertices'):
            logger.info(f"  - Vertices: {len(obj.vertices)}")
        if hasattr(obj, 'faces'):
            logger.info(f"  - Faces: {len(obj.faces)}")

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