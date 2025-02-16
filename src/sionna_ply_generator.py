#sionna_ply_generator.py
import os
import sys
import numpy as np
import struct
import logging
from config import SmartFactoryConfig
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SionnaPLYGenerator:
    """
    Generate PLY files for Sionna ray tracing simulations
    
    Supported materials:
    - concrete: Building structure (walls, floor, ceiling)
    - metal: Shelves and metallic objects
    Each material has specific electromagnetic properties defined in config.
    """
    
    @staticmethod
    def generate_factory_geometries(config, output_dir):
        """Generate PLY files for factory scenario using configuration"""
        try:
            logger.debug("Starting PLY file generation...")
            
            # Use pathlib for path handling
            from pathlib import Path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate walls using config
            wall_configs = {
                'wall_xp': {'x': config.room_dim[0], 'orientation': 'yz'},
                'wall_xm': {'x': 0, 'orientation': 'yz'},
                'wall_yp': {'y': config.room_dim[1], 'orientation': 'xz'},
                'wall_ym': {'y': 0, 'orientation': 'xz'}
            }
            
            for wall_name, wall_config in wall_configs.items():
                output_file = output_path / f'{wall_name}.ply'
                
                SionnaPLYGenerator._generate_vertical_wall(
                    filename=str(output_file),
                    width=config.room_dim[0] if wall_config['orientation'] == 'xz' else config.room_dim[1],
                    height=config.room_dim[2],
                    x=wall_config.get('x', 0),
                    y=wall_config.get('y', 0),
                    orientation=wall_config['orientation'],
                    material_type=config.static_scene['material']
                )
        
                        
            # Generate shelves using config
            shelf_positions = config.scene_objects['shelf_positions']
            shelf_dims = config.scene_objects['shelf_dimensions']
            
            for i, pos in enumerate(shelf_positions):
                output_file = os.path.join(output_dir, f'shelf_{i}.ply')
                SionnaPLYGenerator._generate_shelf_ply(
                    filename=output_file,
                    dims=shelf_dims,
                    position=pos,
                    material_type=config.scene_objects['shelf_material']
                )
            
            # Generate AGV robots using config
            robot_dims = [0.5, 0.5, config.agv_height]
            for i, pos in enumerate(config.agv_positions):
                output_file = os.path.join(output_dir, f'agv_robot_{i}.ply')
                SionnaPLYGenerator._generate_robot_ply(
                    filename=output_file,
                    dims=robot_dims,
                    position=pos
                )
            
            # Generate base station using config
            bs_dims = [0.2, 0.2, 0.1]
            output_file = os.path.join(output_dir, 'base_station.ply')
            SionnaPLYGenerator._generate_modem_ply(
                filename=output_file,
                dims=bs_dims,
                position=config.bs_position
            )
            
            logger.info("PLY file generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error generating PLY files: {str(e)}")
            raise

    @staticmethod
    def _generate_horizontal_surface(filename, width, depth, z=0, material_type=None):
        """
        Generate horizontal surface (floor or ceiling) PLY
        
        Args:
            filename: Output PLY file path
            width: Surface width
            depth: Surface depth
            z: Z-coordinate (height) of the surface
            material_type: Material type for the surface (optional)
        """
        try:
            vertices = [
                (0,      0,     z, 0, 0),  # Bottom-left
                (width,  0,     z, 1, 0),  # Bottom-right
                (width,  depth, z, 1, 1),  # Top-right
                (0,      depth, z, 0, 1)   # Top-left
            ]
            
            # Add material properties if specified
            if material_type:
                vertices = SionnaPLYGenerator._add_material_properties(vertices, material_type)
                
            faces = [[0, 1, 2], [0, 2, 3]]  # Triangulated faces
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            logger.debug(f"Generated horizontal surface: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating horizontal surface PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_vertical_wall(filename, width, height, x=0, y=0, orientation='xz', material_type=None):
        """
        Generate vertical wall PLY
        
        Args:
            filename: Output PLY file path
            width: Wall width
            height: Wall height
            x: X-coordinate of the wall
            y: Y-coordinate of the wall
            orientation: 'xz' for walls parallel to XZ plane, 'yz' for walls parallel to YZ plane
            material_type: Material type for the wall (optional)
        """
        try:
            if orientation == 'xz':
                # For y-oriented walls, swap width with x dimension
                vertices = [
                    (x,     y, 0,      0, 0),  # Bottom-left
                    (x+width, y, 0,      1, 0),  # Bottom-right
                    (x+width, y, height, 1, 1),  # Top-right
                    (x,     y, height, 0, 1)   # Top-left
                ]
            else:  # orientation == 'yz'
                vertices = [
                    (x, y,     0,      0, 0),  # Bottom-left
                    (x, y+width, 0,      1, 0),  # Bottom-right
                    (x, y+width, height, 1, 1),  # Top-right
                    (x, y,     height, 0, 1)   # Top-left
                ]

            # Add material properties if specified
            if material_type:
                vertices = SionnaPLYGenerator._add_material_properties(vertices, material_type)

            # Ensure proper face orientation
            faces = [[0, 1, 2], [0, 2, 3]]  # Counter-clockwise orientation
            
            # Save the PLY file
            SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)
            logger.debug(f"Generated vertical wall: {filename}")
                
        except Exception as e:
            logger.error(f"Error generating vertical wall PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _generate_shelf_ply(filename, dims, position):
        """
        Generate shelf PLY with all six faces (box geometry)
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
    def _generate_robot_ply(filename, dims, position):
        """
        Generate AGV robot PLY with all six faces (box geometry)
        """
        try:
            width, depth, height = dims
            x, y, z = position

            # Define vertices for a box representing the robot
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
            logger.error(f"Error generating robot PLY {filename}: {str(e)}")
            raise
    @staticmethod
    def _add_material_properties(vertices, material_type):
        """Add material-specific properties to vertices"""
        material_props = {
            'shelf': {'reflectivity': 0.8},
            'wall': {'reflectivity': 0.3},
            'floor': {'reflectivity': 0.2},
            'concrete': {'reflectivity': 0.3},  # Added concrete material
            'metal': {'reflectivity': 0.9},     # Added metal material
            'ceiling': {'reflectivity': 0.4}    # Added ceiling material
        }
        
        # Check if material type exists
        if material_type not in material_props:
            logger.warning(f"Unknown material type '{material_type}', using default properties")
            # Use default properties if material type is unknown
            reflectivity = 0.3  # Default reflectivity
        else:
            reflectivity = material_props[material_type]['reflectivity']
        
        # Add material properties to vertices
        return [vertex + (reflectivity,) for vertex in vertices]
            
    @staticmethod
    def _generate_modem_ply(filename, dims, position):
        """
        Generate modem PLY with all six faces (box geometry)
        """
        try:
            width, depth, height = dims
            x, y, z = position

            # Define vertices for a box representing the modem
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
            logger.error(f"Error generating modem PLY {filename}: {str(e)}")
            raise

    @staticmethod
    def _save_binary_ply(filename, vertices, faces):
        """
        Save PLY in binary little-endian format with proper path handling
        """
        try:
            # Convert to raw string to handle special characters
            filename = str(filename)
            
            # Create absolute path and normalize it
            abs_path = os.path.abspath(filename)
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(abs_path)
            os.makedirs(directory, exist_ok=True)
            
            # Use pathlib for robust path handling
            from pathlib import Path
            file_path = Path(abs_path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file using pathlib Path object
            with file_path.open(mode='wb') as f:
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
                        f.write(struct.pack('<f', float(coord)))
                
                # Write face data
                for face in faces:
                    f.write(struct.pack('<B', len(face)))
                    for idx in face:
                        f.write(struct.pack('<i', idx))
                        
            logger.debug(f"Successfully saved PLY file: {file_path}")
            
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
        
    @staticmethod
    def validate_config(config):
        """Validate configuration parameters"""
        required_fields = ['room_dim', 'static_scene', 'scene_objects']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required configuration field: {field}")
            

def main():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Create config instance
        config = SmartFactoryConfig()
        
        # Validate configuration
        logger.info("Validating configuration...")
        SionnaPLYGenerator.validate_config(config)
        
        # Define the meshes directory path using config
        meshes_dir = os.path.join(current_dir, config.ply_config['output_dir'])
        
        # Ensure meshes directory exists
        os.makedirs(meshes_dir, exist_ok=True)
        
        # Generate PLY files using config
        logger.info("Starting PLY file generation...")
        SionnaPLYGenerator.generate_factory_geometries(
            config=config,
            output_dir=meshes_dir
        )
        
        # Verify all generated files
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
            
        # Print success message and file list
        logger.info("PLY files generated and verified successfully!")
        print(f"Meshes directory: {meshes_dir}")
        print("\nGenerated files:")
        for file in files:
            print(f"- {file}")
            
    except Exception as e:
        logger.error(f"Error in PLY generation process: {str(e)}")
        raise
    
    return 0  # Successful execution

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)