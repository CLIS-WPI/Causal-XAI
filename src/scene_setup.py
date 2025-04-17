# This script sets up a smart factory simulation for beam switching using Sionna's Ray Tracing.
# It defines a scene with PLY-based objects, applies radio materials, and configures transmitters/receivers.
# The goal is to simulate beamforming in an indoor environment with AGVs and base stations.
# This script sets up a smart factory simulation for beam switching using Sionna's Ray Tracing.
# It defines a scene with PLY-based objects, applies radio materials, and configures transmitters/receivers.
# The goal is to simulate beamforming in an indoor environment with AGVs and base stations.

import mitsuba
import tensorflow as tf
import os
import logging
from typing import Dict
import xml.etree.ElementTree as ET

# Sionna imports
# --- Note: Added Scene import based on previous Pylance fix ---
from sionna.rt import Scene, load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial

# Import configuration
from config import SmartFactoryConfig # Assuming you have this config file

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Note: Changed return type hint based on previous Pylance fix ---
def setup_scene(config: SmartFactoryConfig) -> Scene:
    """
    Set up the ray tracing scene for a smart factory environment.

    Args:
        config (SmartFactoryConfig): Configuration object containing simulation parameters.

    Returns:
        sionna.rt.Scene: Configured Sionna scene ready for ray tracing and beamforming.
    """
    try:
        # Set Mitsuba variant for GPU-accelerated ray tracing
        #mitsuba.set_variant('cuda_ad_rgb')
        mitsuba.set_variant('scalar_rgb')
        logger.debug(f"Mitsuba variant set to: {mitsuba.variant()}")

        # Define the directory containing PLY files
        meshes_dir = os.path.join(os.path.dirname(__file__), "meshes")
        if not os.path.exists(meshes_dir):
            raise FileNotFoundError(f"Meshes directory not found: {meshes_dir}")
        logger.debug(f"Using meshes directory: {meshes_dir}")

        # Build XML content for the scene
        xml_content = '<?xml version="1.0"?>\n<scene version="3.0.0">\n'

        # Add integrator
        xml_content += '    <integrator type="path">\n'
        xml_content += '        <integer name="max_depth" value="12"/>\n' # This depth is for Mitsuba rendering, Sionna uses scene.max_depth
        xml_content += '    </integrator>\n'

        # Add constant emitter (optional for basic scene loading, but good practice)
        xml_content += '    <emitter type="constant" id="World">\n'
        xml_content += '        <rgb name="radiance" value="1.0 1.0 1.0"/>\n'
        xml_content += '    </emitter>\n'

        # --- Step 1: Define a standard Mitsuba material for initial loading ---
        # ******** CHANGE APPLIED: Initial BSDF definition removed ********
        # xml_content += '    <bsdf type="twosided" id="base_material">\n'
        # xml_content += '        <bsdf type="diffuse">\n'
        # xml_content += '            <rgb name="reflectance" value="0.8 0.8 0.8"/>\n'
        # xml_content += '        </bsdf>\n'
        # xml_content += '    </bsdf>\n'
        # ******************************************************************

        # Load all PLY files and map their materials conceptually
        material_map = {} # Keep track of which Sionna material to apply later
        for ply_file in os.listdir(meshes_dir):
            if ply_file.endswith(".ply"):
                full_path = os.path.join(meshes_dir, ply_file)
                # Ensure path separators are correct for XML (usually forward slashes)
                full_path = full_path.replace("\\", "/")
                name = ply_file[:-4]  # Remove .ply extension

                # Assign conceptual material type based on object name
                material_type = "concrete" if "wall" in name or "floor" in name or "ceiling" in name else "metal"

                # Add shape to XML, WITHOUT referencing any initial BSDF
                xml_content += f'    <shape type="ply" id="mesh-{name}">\n'
                xml_content += f'        <string name="filename" value="{full_path}"/>\n'
                xml_content += '        <boolean name="face_normals" value="true"/>\n'
                # ******** CHANGE APPLIED: Reference to base_material removed ********
                # --- Reference the standard material ID ---
                # xml_content += '        <ref id="base_material" name="bsdf"/>\n'
                # *********************************************************************
                xml_content += '    </shape>\n'

                material_map[name] = material_type # Store for later use with Sionna materials
                logger.debug(f"Added {ply_file} to XML (no initial material ref), conceptual type: {material_type}")

        xml_content += '</scene>'

        # Write temporary XML file for scene loading
        temp_xml_path = os.path.join(meshes_dir, "temp_scene.xml")
        with open(temp_xml_path, 'w') as f:
            f.write(xml_content)
        logger.debug(f"Wrote temporary XML (without initial materials) to {temp_xml_path}")

        # Optional: Add XML validation again if needed
        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()
            logger.debug("Starting XML attribute validation...")
            # Basic validation still useful, even without materials defined here
            for elem in root.iter():
                for attr_name, attr_value in elem.attrib.items():
                    if attr_value is None:
                        logger.error(f"Found None value in attribute '{attr_name}' of element '{elem.tag}'")
                        raise ValueError(f"None value found in attribute '{attr_name}' of element '{elem.tag}'")
            logger.debug("Finished XML attribute validation")
        except ET.ParseError as xml_err:
            logger.error(f"XML parsing failed: {xml_err}")
            raise

        # --- Load the scene using the XML (now without initial material defs/refs) ---
        # Sionna's load_scene / process_xml will likely attach default Holders
        scene = load_scene(filename=temp_xml_path, merge_shapes=False)
        logger.debug("Loaded scene geometry using load_scene (expecting default Holders)")

        # Remove temporary XML file
        os.remove(temp_xml_path)
        logger.debug("Removed temporary XML file")

        # --- Step 2: Define Sionna RadioMaterial objects (Unchanged) ---
        concrete_material = RadioMaterial(
            name="factory_concrete",
            conductivity=config.materials['concrete']['conductivity'],
            relative_permittivity=config.materials['concrete']['relative_permittivity'],
            scattering_coefficient=config.materials['concrete'].get('scattering_coefficient', 0.3),
            thickness=0.1
        )
        metal_material = RadioMaterial(
            name="factory_metal",
            conductivity=config.materials['metal']['conductivity'],
            relative_permittivity=config.materials['metal']['relative_permittivity'],
            scattering_coefficient=config.materials['metal'].get('scattering_coefficient', 0.8),
            thickness=0.01
        )
        logger.info("Defined Sionna radio materials: factory_concrete, factory_metal")

        # --- Step 3: Apply Sionna RadioMaterial objects to the loaded scene (Unchanged) ---
        # This step is now crucial as it assigns the *actual* materials
        for obj_name in scene.objects:
            if ':' in obj_name:
                 base_name_part = obj_name.split(':')[0]
            else:
                 base_name_part = obj_name

            if base_name_part.startswith('mesh-'):
                 base_name = base_name_part.replace('mesh-', '')
            else:
                 base_name = base_name_part

            if base_name in material_map:
                material_to_apply = concrete_material if material_map[base_name] == "concrete" else metal_material
                try:
                    # This should now correctly set the material within the Holder
                    scene.set_material(obj_name, material_to_apply)
                    logger.debug(f"Applied Sionna '{material_map[base_name]}' material ({material_to_apply.name}) to {obj_name}")
                except Exception as e:
                    logger.error(f"Failed to apply material to {obj_name}: {e}")
            else:
                logger.warning(f"No material mapping found for base name '{base_name}' derived from scene object '{obj_name}'. Defaulting to concrete.")
                try:
                    scene.set_material(obj_name, concrete_material)
                except Exception as e:
                    logger.error(f"Failed to apply default material to {obj_name}: {e}")


        # Set scene frequency (Unchanged)
        scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        logger.debug(f"Set scene frequency to {config.carrier_frequency / 1e9:.2f} GHz")

        # Add base station (transmitter) (Unchanged)
        tx = Transmitter(
            name="bs",
            position=tf.constant(config.bs_position, dtype=tf.float32),
            orientation=tf.constant(config.bs_orientation, dtype=tf.float32)
        )
        tx_array = PlanarArray(
            num_rows=config.bs_array['num_rows'],
            num_cols=config.bs_array['num_cols'],
            vertical_spacing=config.bs_array['vertical_spacing'],
            horizontal_spacing=config.bs_array['horizontal_spacing'],
            pattern=config.bs_array['pattern'],
            polarization=config.bs_array['polarization']
        )
        tx.array = tx_array
        scene.add(tx)
        logger.debug("Added base station transmitter")

        # Add AGVs (receivers) (Unchanged)
        for i in range(config.num_agvs):
            agv_id = f"agv_{i}"
            trajectory_key = f"agv_{i+1}" if f"agv_{i+1}" in config.agv_trajectories else agv_id
            if trajectory_key not in config.agv_trajectories:
                 logger.error(f"Trajectory key '{trajectory_key}' not found in config.agv_trajectories")
                 continue
            initial_pos = config.agv_trajectories[trajectory_key][0]

            if i >= len(config.agv_orientations):
                 logger.error(f"Orientation index {i} out of bounds for config.agv_orientations")
                 continue

            rx = Receiver(
                name=f"rx_{agv_id}",
                position=tf.constant(initial_pos, dtype=tf.float32),
                orientation=tf.constant(config.agv_orientations[i], dtype=tf.float32)
            )
            rx_array = PlanarArray(
                num_rows=config.agv_array['num_rows'],
                num_cols=config.agv_array['num_cols'],
                vertical_spacing=config.agv_array['vertical_spacing'],
                horizontal_spacing=config.agv_array['horizontal_spacing'],
                pattern=config.agv_array['pattern'],
                polarization=config.agv_array['polarization']
            )
            rx.array = rx_array
            scene.add(rx)
            logger.debug(f"Added receiver for {agv_id} at position {initial_pos}")

        # Configure ray tracing parameters (Unchanged)
        scene.los = config.ray_tracing['los']
        scene.reflection = config.ray_tracing['reflection']
        scene.diffraction = config.ray_tracing['diffraction']
        scene.scattering = config.ray_tracing['scattering']
        scene.max_depth = config.ray_tracing['max_depth']
        scene.num_samples = config.ray_tracing['num_samples'] * config.bs_array['num_rows'] * config.bs_array['num_cols'] # Adjust samples as per Sionna docs
        scene.method = config.ray_tracing['method']
        logger.debug(f"Configured ray tracing parameters (num_samples adjusted to: {scene.num_samples})")

        logger.info("Scene setup completed successfully")
        return scene

    except FileNotFoundError as e:
        logger.error(f"Scene setup failed: {str(e)}")
        raise
    except KeyError as e:
        logger.error(f"Scene setup failed: Missing key in configuration - {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Scene setup failed with unexpected error: {str(e)}")
        logger.exception("Traceback:")
        raise
