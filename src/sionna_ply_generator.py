#src/sionna_ply_generator.py
import numpy as np
import struct
import os

class SionnaPLYGenerator:
    @staticmethod
    def generate_factory_geometries(
        room_dims=[20, 20, 5], 
        shelf_dims=[2, 1, 4], 
        output_dir='meshes'
    ):
        """Generate PLY files for factory scenario"""
        os.makedirs(output_dir, exist_ok=True)

        # Room boundary generation
        geometries = [
            {
                'name': 'floor',
                'width': room_dims[0],
                'depth': room_dims[1],
                'z': 0
            },
            {
                'name': 'ceiling', 
                'width': room_dims[0],
                'depth': room_dims[1], 
                'z': room_dims[2]
            },
            {
                'name': 'wall_xp',  # Positive X wall
                'width': room_dims[1], 
                'depth': room_dims[2],
                'x': room_dims[0]
            },
            {
                'name': 'wall_xm',  # Negative X wall
                'width': room_dims[1], 
                'depth': room_dims[2],
                'x': 0
            },
            {
                'name': 'wall_yp',  # Positive Y wall
                'width': room_dims[0], 
                'depth': room_dims[2],
                'y': room_dims[1]
            },
            {
                'name': 'wall_ym',  # Negative Y wall
                'width': room_dims[0], 
                'depth': room_dims[2],
                'y': 0
            }
        ]

        # Generate room boundary PLYs
        for geo in geometries:
            SionnaPLYGenerator._generate_rectangular_ply(
                filename=os.path.join(output_dir, f'{geo["name"]}.ply'),
                width=geo['width'], 
                depth=geo.get('depth', geo['width']),
                x=geo.get('x', 0),
                y=geo.get('y', 0),
                z=geo.get('z', 0)
            )

        # Shelf generation
        shelf_positions = [
            [5.0, 5.0, 0.0], 
            [15.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
            [5.0, 15.0, 0.0],
            [15.0, 15.0, 0.0]
        ]

        for i, pos in enumerate(shelf_positions):
            SionnaPLYGenerator._generate_shelf_ply(
                filename=os.path.join(output_dir, f'shelf_{i}.ply'), 
                dims=shelf_dims, 
                position=pos
            )

    @staticmethod
    def _generate_rectangular_ply(filename, width, depth, x=0, y=0, z=0):
        """Generate rectangular PLY following Sionna's binary format"""
        vertices = [
            (x, y, z, 0, 0),           # Bottom-left
            (x+width, y, z, 1, 0),      # Bottom-right
            (x+width, y+depth, z, 1, 1),# Top-right
            (x, y+depth, z, 0, 1)       # Top-left
        ]
        faces = [[0, 1, 2], [0, 2, 3]]  # Triangulated faces
        SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)

    @staticmethod
    def _generate_shelf_ply(filename, dims, position):
        """Generate complex shelf PLY with multiple surfaces"""
        width, depth, height = dims
        x, y, z = position

        vertices = [
            # Bottom face
            (x, y, z, 0, 0),
            (x+width, y, z, 1, 0),
            (x+width, y+depth, z, 1, 1),
            (x, y+depth, z, 0, 1),
            
            # Top face (offset by height)
            (x, y, z+height, 0, 0),
            (x+width, y, z+height, 1, 0),
            (x+width, y+depth, z+height, 1, 1),
            (x, y+depth, z+height, 0, 1)
        ]
        
        faces = [
            # Bottom
            [0, 1, 2], [0, 2, 3],
            # Top
            [4, 5, 6], [4, 6, 7],
            # Side faces (front, back, left, right)
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2]   # Right
        ]
        
        SionnaPLYGenerator._save_binary_ply(filename, vertices, faces)

    @staticmethod
    def _save_binary_ply(filename, vertices, faces):
        """Save PLY in binary little-endian format"""
        with open(filename, 'wb') as f:
            # PLY header
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
            
            # Write vertices
            for vertex in vertices:
                for coord in vertex:
                    f.write(struct.pack('<f', coord))
            
            # Write faces
            for face in faces:
                f.write(struct.pack('<B', len(face)))
                for idx in face:
                    f.write(struct.pack('<i', idx))