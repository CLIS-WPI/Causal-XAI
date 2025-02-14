#src/sionna_ply_generator.py
import numpy as np
import struct
import os
import logging

logger = logging.getLogger(__name__)

class SionnaPLYGenerator:
    """Generate PLY files for Sionna ray tracing simulations"""
    
    @staticmethod
    def generate_factory_geometries(config):
        """
        Generate PLY files for factory scenario using configuration
        
        Args:
            config: SmartFactoryConfig object containing all necessary parameters
        """
        try:
            logger.debug("Starting PLY file generation...")
            logger.debug(f"Output directory: {config.ply_config['output_dir']}")
            
            # Create output directory
            os.makedirs(config.ply_config['output_dir'], exist_ok=True)

            # Room boundary generation from config
            room_dims = config.ply_config['room_dims']
            
            # Generate floor and ceiling
            for name in ['floor', 'ceiling']:
                output_file = os.path.join(config.ply_config['output_dir'], f'{name}.ply')
                z_pos = config.ply_config['geometry_mapping'][name]['z']
                SionnaPLYGenerator._generate_horizontal_surface(
                    filename=output_file,
                    width=room_dims[0],
                    depth=room_dims[1],
                    z=z_pos
                )
                logger.debug(f"Generated {name}: {output_file}")

            # Generate walls
            wall_configs = {
                'wall_xp': {'x': room_dims[0], 'orientation': 'yz'},
                'wall_xm': {'x': 0, 'orientation': 'yz'},
                'wall_yp': {'y': room_dims[1], 'orientation': 'xz'},
                'wall_ym': {'y': 0, 'orientation': 'xz'}
            }

            for wall_name, wall_config in wall_configs.items():
                output_file = os.path.join(config.ply_config['output_dir'], f'{wall_name}.ply')
                SionnaPLYGenerator._generate_vertical_wall(
                    filename=output_file,
                    width=room_dims[0] if wall_config['orientation'] == 'xz' else room_dims[1],
                    height=room_dims[2],
                    x=wall_config.get('x', 0),
                    y=wall_config.get('y', 0),
                    orientation=wall_config['orientation']
                )
                logger.debug(f"Generated {wall_name}: {output_file}")

            # Generate shelves
            shelf_dims = config.ply_config['shelf_dims']
            shelf_positions = config.ply_config['shelf_positions']

            logger.debug(f"Generating {len(shelf_positions)} shelves with dimensions: {shelf_dims}")
            for i, pos in enumerate(shelf_positions):
                output_file = os.path.join(config.ply_config['output_dir'], f'shelf_{i}.ply')
                SionnaPLYGenerator._generate_shelf_ply(
                    filename=output_file,
                    dims=shelf_dims,
                    position=pos
                )
                logger.debug(f"Generated shelf PLY {i} at position {pos}")

            logger.info("PLY file generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error generating PLY files: {str(e)}")
            raise RuntimeError(f"PLY file generation failed: {str(e)}") from e

    @staticmethod
    def _generate_horizontal_surface(filename, width, depth, z=0):
        """
        Generate horizontal surface (floor or ceiling) PLY
        """
        try:
            vertices = [
                (0,      0,     z, 0, 0),  # Bottom-left
                (width,  0,     z, 1, 0),  # Bottom-right
                (width,  depth, z, 1, 1),  # Top-right
                (0,      depth, z, 0, 1)   # Top-left
            ]
            faces = [[0, 1, 2], [0, 2, 3]]  # Triangulated faces
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            
        except Exception as e:
            logger.error(f"Error generating horizontal surface PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_vertical_wall(filename, width, height, x=0, y=0, orientation='xz'):
        """
        Generate vertical wall PLY
        
        Args:
            orientation: 'xz' for walls parallel to XZ plane, 'yz' for walls parallel to YZ plane
        """
        try:
            if orientation == 'xz':
                # Wall parallel to XZ plane (constant y)
                vertices = [
                    (0,     y, 0,      0, 0),  # Bottom-left
                    (width, y, 0,      1, 0),  # Bottom-right
                    (width, y, height, 1, 1),  # Top-right
                    (0,     y, height, 0, 1)   # Top-left
                ]
            else:  # orientation == 'yz'
                # Wall parallel to YZ plane (constant x)
                vertices = [
                    (x, 0,     0,      0, 0),  # Bottom-left
                    (x, width, 0,      1, 0),  # Bottom-right
                    (x, width, height, 1, 1),  # Top-right
                    (x, 0,     height, 0, 1)   # Top-left
                ]

            faces = [[0, 1, 2], [0, 2, 3]]  # Triangulated faces
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            
        except Exception as e:
            logger.error(f"Error generating vertical wall PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_shelf_ply(filename, dims, position):
        """
        Generate shelf PLY with all six faces
        """
        try:
            width, depth, height = dims
            x, y, z = position

            # Define all vertices for a complete 3D box
            vertices = [
                # Front face vertices (y constant)
                (x,         y,         z,         0, 0),  # 0 bottom-left
                (x+width,   y,         z,         1, 0),  # 1 bottom-right
                (x+width,   y,         z+height,  1, 1),  # 2 top-right
                (x,         y,         z+height,  0, 1),  # 3 top-left
                
                # Back face vertices (y+depth)
                (x,         y+depth,   z,         0, 0),  # 4 bottom-left
                (x+width,   y+depth,   z,         1, 0),  # 5 bottom-right
                (x+width,   y+depth,   z+height,  1, 1),  # 6 top-right
                (x,         y+depth,   z+height,  0, 1),  # 7 top-left
            ]
            
            # Define faces - each face is made of two triangles
            faces = [
                # Front face
                [0, 1, 2], [0, 2, 3],
                # Back face
                [5, 4, 7], [5, 7, 6],
                # Top face
                [3, 2, 6], [3, 6, 7],
                # Bottom face
                [4, 5, 1], [4, 1, 0],
                # Left face
                [4, 0, 3], [4, 3, 7],
                # Right face
                [1, 5, 6], [1, 6, 2]
            ]
            
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            
        except Exception as e:
            logger.error(f"Error generating shelf PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _save_binary_ply(filename, vertices, faces):
        """
        Save PLY in binary little-endian format
        """
        try:
            with open(filename, 'wb') as f:
                # Write PLY header
                f.write(b'ply\n')
                f.write(b'format binary_little_endian 1.0\n')
                f.write(f'element vertex {len(vertices)}\n'.encode())
                f.write(b'property float x\n')
                f.write(b'property float y\n')
                f.write(b'property float z\n')
                f.write(b'property float u\n')
                f.write(b'property float v\n')
                f.write(f'element face {len(faces)}\n'.encode())
                f.write(b'property list uchar int vertex_indices\n')
                f.write(b'end_header\n')
                
                # Write vertex data
                for vertex in vertices:
                    for coord in vertex:
                        f.write(struct.pack('<f', coord))
                
                # Write face data
                for face in faces:
                    f.write(struct.pack('<B', len(face)))  # Number of vertices in face
                    for idx in face:
                        f.write(struct.pack('<i', idx))    # Vertex indices
                        
        except Exception as e:
            logger.error(f"Error saving PLY file {filename}: {str(e)}")
            raise

    @staticmethod
    def verify_ply_file(filename):
        """
        Verify that a PLY file was generated correctly
        """
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