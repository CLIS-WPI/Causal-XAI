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
            geometries = []
            
            # Generate geometry specifications from config mapping
            for name, pos in config.ply_config['geometry_mapping'].items():
                geo = {
                    'name': name,
                    'width': room_dims[0] if 'y' in pos else room_dims[1],
                    'depth': room_dims[1] if 'x' in pos else room_dims[2],
                    **pos  # Add position information
                }
                geometries.append(geo)
                logger.debug(f"Added geometry: {name} with dimensions: {geo}")

            # Generate room boundary PLYs
            for geo in geometries:
                output_file = os.path.join(config.ply_config['output_dir'], f'{geo["name"]}.ply')
                SionnaPLYGenerator._generate_rectangular_ply(
                    filename=output_file,
                    width=geo['width'], 
                    depth=geo.get('depth', geo['width']),
                    x=geo.get('x', 0),
                    y=geo.get('y', 0),
                    z=geo.get('z', 0)
                )
                logger.debug(f"Generated boundary PLY: {output_file}")

            # Shelf generation from config
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
    def _generate_rectangular_ply(filename, width, depth, x=0, y=0, z=0):
        """
        Generate rectangular PLY following Sionna's binary format
        
        Args:
            filename (str): Output PLY file path
            width (float): Width of the rectangle
            depth (float): Depth of the rectangle
            x (float): X coordinate offset
            y (float): Y coordinate offset
            z (float): Z coordinate offset
        """
        try:
            vertices = [
                (x, y, z, 0, 0),                # Bottom-left
                (x+width, y, z, 1, 0),          # Bottom-right
                (x+width, y+depth, z, 1, 1),    # Top-right
                (x, y+depth, z, 0, 1)           # Top-left
            ]
            faces = [[0, 1, 2], [0, 2, 3]]      # Triangulated faces
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            
        except Exception as e:
            logger.error(f"Error generating rectangular PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_shelf_ply(filename, dims, position):
        """
        Generate complex shelf PLY with multiple surfaces
        
        Args:
            filename (str): Output PLY file path
            dims (list): [width, depth, height] of the shelf
            position (list): [x, y, z] position of the shelf
        """
        try:
            width, depth, height = dims
            x, y, z = position

            vertices = [
                # Bottom face vertices
                (x, y, z, 0, 0),
                (x+width, y, z, 1, 0),
                (x+width, y+depth, z, 1, 1),
                (x, y+depth, z, 0, 1),
                
                # Top face vertices (offset by height)
                (x, y, z+height, 0, 0),
                (x+width, y, z+height, 1, 0),
                (x+width, y+depth, z+height, 1, 1),
                (x, y+depth, z+height, 0, 1)
            ]
            
            faces = [
                # Bottom face triangles
                [0, 1, 2], [0, 2, 3],
                # Top face triangles
                [4, 5, 6], [4, 6, 7],
                # Side faces
                [0, 4, 5], [0, 5, 1],  # Front
                [2, 6, 7], [2, 7, 3],  # Back
                [0, 3, 7], [0, 7, 4],  # Left
                [1, 5, 6], [1, 6, 2]   # Right
            ]
            
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            
        except Exception as e:
            logger.error(f"Error generating shelf PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _save_binary_ply(filename, vertices, faces):
        """
        Save PLY in binary little-endian format
        
        Args:
            filename (str): Output PLY file path
            vertices (list): List of vertex tuples (x, y, z, u, v)
            faces (list): List of face index lists
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
        
        Args:
            filename (str): PLY file to verify
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                # Check header
                header = f.readline().decode().strip()
                if header != 'ply':
                    logger.error(f"Invalid PLY header in {filename}")
                    return False
                    
                # Read format
                format_line = f.readline().decode().strip()
                if not format_line.startswith('format binary_little_endian'):
                    logger.error(f"Invalid format specification in {filename}")
                    return False
                    
                return True
                
        except Exception as e:
            logger.error(f"Error verifying PLY file {filename}: {str(e)}")
            return False