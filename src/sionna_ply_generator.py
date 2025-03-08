#sionna_ply_generator.py
import mitsuba
import os
import tensorflow as tf
import numpy as np
import struct
import logging
from config import SmartFactoryConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SionnaPLYGenerator:
    """
    Generate PLY files for Sionna ray tracing simulations.
    
    Materials:
    - Uses 'factory_concrete' and 'factory_metal' mapped from config.static_scene and config.scene_objects.
    """

    @staticmethod
    def generate_factory_geometries(config, output_dir):
        """Generate PLY files for factory scenario using configuration."""
        try:
            logger.debug("Starting PLY file generation...")
            os.makedirs(output_dir, exist_ok=True)

            # Generate floor (z=0) with material from config
            SionnaPLYGenerator._generate_horizontal_surface(
                filename=os.path.join(output_dir, "floor.ply"),
                width=config.room_dim[0],
                depth=config.room_dim[1],
                z=0.0,  # Floor at ground level
                material_type=config.static_scene['floor_material']  # 'concrete'
            )

            # Generate ceiling (z=room_dim[2]) with material from config
            SionnaPLYGenerator._generate_horizontal_surface(
                filename=os.path.join(output_dir, "ceiling.ply"),
                width=config.room_dim[0],
                depth=config.room_dim[1],
                z=config.room_dim[2],  # Ceiling at room height
                material_type=config.static_scene['ceiling_material']  # 'concrete'
            )

            # Generate walls with material from config
            wall_configs = {
                'wall_xp': {'x': config.room_dim[0], 'y': config.room_dim[1]/2, 'orientation': 'yz'},
                'wall_xm': {'x': 0.0, 'y': config.room_dim[1]/2, 'orientation': 'yz'},
                'wall_yp': {'x': config.room_dim[0]/2, 'y': config.room_dim[1], 'orientation': 'xz'},
                'wall_ym': {'x': config.room_dim[0]/2, 'y': 0.0, 'orientation': 'xz'}
            }
            for wall_name, wall_config in wall_configs.items():
                SionnaPLYGenerator._generate_vertical_wall(
                    filename=os.path.join(output_dir, f"{wall_name}.ply"),
                    width=config.room_dim[0] if wall_config['orientation'] == 'xz' else config.room_dim[1],
                    height=config.room_dim[2],
                    x=wall_config['x'],
                    y=wall_config['y'],
                    orientation=wall_config['orientation'],
                    material_type=config.static_scene['material']  # 'concrete'
                )

            # Generate shelves with material from config
            shelf_positions = config.scene_objects['shelf_positions']
            shelf_dimensions = config.scene_objects['shelf_dimensions']
            for i, pos in enumerate(shelf_positions):
                shelf_dims = shelf_dimensions[i] if isinstance(shelf_dimensions, list) and i < len(shelf_dimensions) else shelf_dimensions
                SionnaPLYGenerator._generate_shelf_ply(
                    filename=os.path.join(output_dir, f"shelf_{i}.ply"),
                    dims=shelf_dims,
                    position=pos,
                    material_type=config.scene_objects['shelf_material']  # 'metal'
                )

            # Generate AGVs with material from config
            robot_dims = config.agv_dimensions
            for i in range(config.num_agvs):
                initial_pos = config.agv_trajectories[f"agv_{i+1}"][0]  # Use initial trajectory position
                SionnaPLYGenerator._generate_robot_ply(
                    filename=os.path.join(output_dir, f"agv_robot_{i}.ply"),
                    dims=robot_dims,
                    position=initial_pos,
                    material_type=config.scene_objects.get('agv_material', 'metal')  # Default to 'metal'
                )

            # Generate base station with dimensions and material from config
            bs_dims = config.scene_objects.get('bs_dimensions', [0.2, 0.2, 0.1])  # Fallback if not in config
            SionnaPLYGenerator._generate_modem_ply(
                filename=os.path.join(output_dir, "base_station.ply"),
                dims=bs_dims,
                position=config.bs_position,
                material_type=config.scene_objects.get('bs_material', 'metal')  # Default to 'metal'
            )

            logger.info("PLY file generation completed successfully")

        except Exception as e:
            logger.error(f"Error generating PLY files: {str(e)}")
            raise

    @staticmethod
    def _generate_horizontal_surface(filename, width, depth, z, material_type='concrete'):
        """Generate a horizontal surface (floor or ceiling) PLY."""
        try:
            # Map config material names to Sionna material IDs
            material_ids = {'concrete': 0, 'metal': 1}
            if material_type not in material_ids:
                raise ValueError(f"Invalid material type: {material_type}")
            material_id = material_ids[material_type]

            with open(filename, 'wb') as f:
                f.write(b'ply\n')
                f.write(b'format binary_little_endian 1.0\n')
                f.write(b'element vertex 4\n')
                f.write(b'property float x\n')
                f.write(b'property float y\n')
                f.write(b'property float z\n')
                f.write(b'element face 2\n')
                f.write(b'property list uchar int vertex_indices\n')
                f.write(b'property int material_index\n')
                f.write(b'end_header\n')

                vertices = [
                    (0.0, 0.0, z),      # Bottom-left
                    (width, 0.0, z),    # Bottom-right
                    (width, depth, z),  # Top-right
                    (0.0, depth, z)     # Top-left
                ]
                for vertex in vertices:
                    for value in vertex:
                        f.write(struct.pack('<f', float(value)))

                # Two triangles
                f.write(struct.pack('<B', 3))  # First triangle
                f.write(struct.pack('<3i', 0, 1, 2))
                f.write(struct.pack('<i', material_id))
                f.write(struct.pack('<B', 3))  # Second triangle
                f.write(struct.pack('<3i', 0, 2, 3))
                f.write(struct.pack('<i', material_id))

            logger.debug(f"Generated horizontal surface: {filename} with material: {material_type}")

        except Exception as e:
            logger.error(f"Error generating horizontal surface PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_vertical_wall(filename, width, height, x, y, orientation, material_type='concrete'):
        """Generate a vertical wall PLY."""
        try:
            material_ids = {'concrete': 0, 'metal': 1}
            if material_type not in material_ids:
                raise ValueError(f"Invalid material type: {material_type}")
            material_id = material_ids[material_type]

            with open(filename, 'wb') as f:
                f.write(b'ply\n')
                f.write(b'format binary_little_endian 1.0\n')
                f.write(b'element vertex 4\n')
                f.write(b'property float x\n')
                f.write(b'property float y\n')
                f.write(b'property float z\n')
                f.write(b'element face 2\n')
                f.write(b'property list uchar int vertex_indices\n')
                f.write(b'property int material_index\n')
                f.write(b'end_header\n')

                if orientation == 'xz':
                    vertices = [
                        (x, y, 0.0),      # Bottom-left
                        (x+width, y, 0.0),  # Bottom-right
                        (x+width, y, height), # Top-right
                        (x, y, height)    # Top-left
                    ]
                else:  # 'yz'
                    vertices = [
                        (x, y, 0.0),      # Bottom-left
                        (x, y+width, 0.0),  # Bottom-right
                        (x, y+width, height), # Top-right
                        (x, y, height)    # Top-left
                    ]

                for vertex in vertices:
                    for value in vertex:
                        f.write(struct.pack('<f', float(value)))

                f.write(struct.pack('<B', 3))
                f.write(struct.pack('<3i', 0, 1, 2))
                f.write(struct.pack('<i', material_id))
                f.write(struct.pack('<B', 3))
                f.write(struct.pack('<3i', 0, 2, 3))
                f.write(struct.pack('<i', material_id))

            logger.debug(f"Generated vertical wall: {filename} with material: {material_type}")

        except Exception as e:
            logger.error(f"Error generating vertical wall PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_shelf_ply(filename, dims, position, material_type='metal'):
        """Generate a shelf PLY (box geometry)."""
        try:
            material_ids = {'concrete': 0, 'metal': 1}
            if material_type not in material_ids:
                raise ValueError(f"Invalid material type: {material_type}")
            material_id = material_ids[material_type]

            width, depth, height = dims
            x, y, z = position

            vertices = [
                (x, y, z),          # 0: Bottom-left-front
                (x+width, y, z),    # 1: Bottom-right-front
                (x+width, y, z+height),  # 2: Top-right-front
                (x, y, z+height),   # 3: Top-left-front
                (x, y+depth, z),    # 4: Bottom-left-back
                (x+width, y+depth, z),  # 5: Bottom-right-back
                (x+width, y+depth, z+height),  # 6: Top-right-back
                (x, y+depth, z+height)  # 7: Top-left-back
            ]

            faces = [
                (0, 1, 2), (0, 2, 3),  # Front
                (5, 4, 7), (5, 7, 6),  # Back
                (3, 2, 6), (3, 6, 7),  # Top
                (4, 5, 1), (4, 1, 0),  # Bottom
                (4, 0, 3), (4, 3, 7),  # Left
                (1, 5, 6), (1, 6, 2)   # Right
            ]

            with open(filename, 'wb') as f:
                f.write(b'ply\n')
                f.write(b'format binary_little_endian 1.0\n')
                f.write(f'element vertex {len(vertices)}\n'.encode())
                f.write(b'property float x\n')
                f.write(b'property float y\n')
                f.write(b'property float z\n')
                f.write(f'element face {len(faces)}\n'.encode())
                f.write(b'property list uchar int vertex_indices\n')
                f.write(b'property int material_index\n')
                f.write(b'end_header\n')

                for vertex in vertices:
                    for coord in vertex:
                        f.write(struct.pack('<f', float(coord)))

                for face in faces:
                    f.write(struct.pack('<B', 3))
                    f.write(struct.pack('<3i', *face))
                    f.write(struct.pack('<i', material_id))

            logger.debug(f"Generated shelf PLY: {filename} with material: {material_type}")

        except Exception as e:
            logger.error(f"Error generating shelf PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_robot_ply(filename, dims, position, material_type='metal'):
        """Generate an AGV robot PLY (box geometry)."""
        try:
            material_ids = {'concrete': 0, 'metal': 1}
            if material_type not in material_ids:
                raise ValueError(f"Invalid material type: {material_type}")
            material_id = material_ids[material_type]

            width, depth, height = dims
            x, y, z = position

            vertices = [
                (x, y, z),          # 0
                (x+width, y, z),    # 1
                (x+width, y, z+height),  # 2
                (x, y, z+height),   # 3
                (x, y+depth, z),    # 4
                (x+width, y+depth, z),  # 5
                (x+width, y+depth, z+height),  # 6
                (x, y+depth, z+height)  # 7
            ]

            faces = [
                (0, 1, 2), (0, 2, 3),  # Front
                (5, 4, 7), (5, 7, 6),  # Back
                (3, 2, 6), (3, 6, 7),  # Top
                (4, 5, 1), (4, 1, 0),  # Bottom
                (4, 0, 3), (4, 3, 7),  # Left
                (1, 5, 6), (1, 6, 2)   # Right
            ]

            with open(filename, 'wb') as f:
                f.write(b'ply\n')
                f.write(b'format binary_little_endian 1.0\n')
                f.write(f'element vertex {len(vertices)}\n'.encode())
                f.write(b'property float x\n')
                f.write(b'property float y\n')
                f.write(b'property float z\n')
                f.write(f'element face {len(faces)}\n'.encode())
                f.write(b'property list uchar int vertex_indices\n')
                f.write(b'property int material_index\n')
                f.write(b'end_header\n')

                for vertex in vertices:
                    for coord in vertex:
                        f.write(struct.pack('<f', float(coord)))

                for face in faces:
                    f.write(struct.pack('<B', 3))
                    f.write(struct.pack('<3i', *face))
                    f.write(struct.pack('<i', material_id))

            logger.debug(f"Generated robot PLY: {filename} with material: {material_type}")

        except Exception as e:
            logger.error(f"Error generating robot PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_modem_ply(filename, dims, position, material_type='metal'):
        """Generate a base station PLY."""
        try:
            material_ids = {'concrete': 0, 'metal': 1}
            if material_type not in material_ids:
                raise ValueError(f"Invalid material type: {material_type}")
            material_id = material_ids[material_type]

            width, depth, height = dims
            x, y, z = position

            vertices = [
                (x-width/2, y-depth/2, z),
                (x+width/2, y-depth/2, z),
                (x+width/2, y+depth/2, z),
                (x-width/2, y+depth/2, z),
                (x-width/2, y-depth/2, z+height),
                (x+width/2, y-depth/2, z+height),
                (x+width/2, y+depth/2, z+height),
                (x-width/2, y+depth/2, z+height)
            ]

            faces = [
                (0, 1, 2), (0, 2, 3),  # Bottom
                (4, 5, 6), (4, 6, 7),  # Top
                (0, 1, 5), (0, 5, 4),  # Front
                (2, 3, 7), (2, 7, 6),  # Back
                (0, 3, 7), (0, 7, 4),  # Left
                (1, 2, 6), (1, 6, 5)   # Right
            ]

            with open(filename, 'wb') as f:
                f.write(b'ply\n')
                f.write(b'format binary_little_endian 1.0\n')
                f.write(f'element vertex {len(vertices)}\n'.encode())
                f.write(b'property float x\n')
                f.write(b'property float y\n')
                f.write(b'property float z\n')
                f.write(f'element face {len(faces)}\n'.encode())
                f.write(b'property list uchar int vertex_indices\n')
                f.write(b'property int material_index\n')
                f.write(b'end_header\n')

                for vertex in vertices:
                    for coord in vertex:
                        f.write(struct.pack('<f', float(coord)))

                for face in faces:
                    f.write(struct.pack('<B', 3))
                    f.write(struct.pack('<3i', *face))
                    f.write(struct.pack('<i', material_id))

            logger.debug(f"Generated modem PLY: {filename} with material: {material_type}")

        except Exception as e:
            logger.error(f"Error generating modem PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def verify_ply_file(filename):
        """Verify that a PLY file was generated correctly."""
        try:
            with open(filename, 'rb') as f:
                header = f.readline().decode().strip()
                if header != 'ply':
                    logger.error(f"Invalid PLY header in {filename}")
                    return False
                format_line = f.readline().decode().strip()
                if not format_line.startswith('format binary_little_endian'):
                    logger.error(f"Invalid format specification in {filename}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Error verifying PLY file {filename}: {str(e)}")
            return False

    @staticmethod
    def validate_config(config):
        """Validate configuration parameters."""
        required_fields = ['room_dim', 'scene_objects', 'bs_position', 'agv_trajectories', 'num_agvs', 'agv_dimensions']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required configuration field: {field}")

def main():
    try:
        config = SmartFactoryConfig()
        logger.info("Validating configuration...")
        SionnaPLYGenerator.validate_config(config)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        meshes_dir = os.path.join(current_dir, "meshes")
        logger.info("Starting PLY file generation...")
        SionnaPLYGenerator.generate_factory_geometries(config=config, output_dir=meshes_dir)

        logger.info("Verifying generated files...")
        verification_failed = False
        files = os.listdir(meshes_dir)
        for file in files:
            if file.endswith('.ply'):
                file_path = os.path.join(meshes_dir, file)
                if not SionnaPLYGenerator.verify_ply_file(file_path):
                    logger.error(f"Verification failed for {file}")
                    verification_failed = True

        if verification_failed:
            raise ValueError("One or more PLY files failed verification")

        logger.info("PLY files generated and verified successfully!")
        print(f"Meshes directory: {meshes_dir}")
        print("\nGenerated files:")
        for file in files:
            print(f"- {file}")

    except Exception as e:
        logger.error(f"Error in PLY generation process: {str(e)}")
        raise

if __name__ == "__main__":
    import sys  # Added missing import for sys.exit
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)