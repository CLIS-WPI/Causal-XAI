import tensorflow as tf
from sionna.constants import SPEED_OF_LIGHT
import os

class SmartFactoryConfig:
    """Configuration class for Smart Factory Channel Simulation using Ray Tracing"""
    def __init__(self):
        # Simulation parameters optimized for beam switching
        self.num_time_steps = 200  # Increased for longer observation
        self.sampling_frequency = tf.cast(10000, tf.float32)  # Increased for better granularity
        self.batch_size = 1
        self.dtype = tf.complex64
        self.real_dtype = tf.float32
        self.seed = 42
        self.num_subcarriers = 128
        self.subcarrier_spacing = 15e3
        self.scene_type = "indoor"
        
        # Room dimensions remain the same
        self.room_dim = [20.0, 20.0, 5.0]
        
        # Frequency configuration for mmWave
        self.carrier_frequency = tf.cast(28e9, tf.float32)
        self.wavelength = tf.cast(SPEED_OF_LIGHT/self.carrier_frequency, tf.float32)
        
        # Base station configuration optimized for beam switching
        self.bs_position = [10.0, 10.0, 4.5]
        self.bs_orientation = [0.0, 0.0, -90.0]
        self.bs_array = {
            'num_rows': 16,          
            'num_cols': 4,
            'vertical_spacing': 0.7,    # Vertical spacing between elements
            'horizontal_spacing': 0.5,   # Horizontal spacing between elements
            'pattern': "tr38901",       # Antenna pattern
            'polarization': "VH"        # Type of polarization
        }

        # Enhanced Material properties for better reflection modeling
        self.materials = {
            'concrete': {
                'name': "concrete",
                'relative_permittivity': 5.31,
                'conductivity': 1.0e7,
                'roughness': 0.1e-3,
                'scattering_coefficient': 0.5,
                'xpd_coefficient': 8.0
            },
            'metal': {
                'name': "metal",
                'relative_permittivity': 1.0,
                'conductivity': 1.0e7,
                'roughness': 0.2e-3,
                'scattering_coefficient': 0.3,
                'xpd_coefficient': 15.0
            }
        }

        # AGV Configuration optimized for beam switching demonstration
        self.num_agvs = 2
        self.agv_height = 0.5
        self.agv_speed = 0.5  # Reduced for better observation
        self.agv_dimensions = [1.0, 1.0, 0.5]
        self.agv_orientations = [
            [0.0, 0.0, 0.0],  # Initial orientation for AGV 0
            [0.0, 0.0, 0.0]   # Initial orientation for AGV 1
        ]
        # Simplified AGV array configuration
        self.agv_array = {
            'num_rows': 1,
            'num_cols': 1,
            'vertical_spacing': 0.5 * self.wavelength,
            'horizontal_spacing': 0.5 * self.wavelength,
            'pattern': "dipole",
            'polarization': "cross",
            'pattern': "tr38901",      # Antenna pattern
            'polarization': "VH"       # Type of polarization
        }

        # Optimized AGV trajectories for beam switching demonstration
        self.agv_trajectories = {
            'agv_1': [[2.0, 2.0], [18.0, 2.0], [2.0, 2.0]],  # Linear path
            'agv_2': [[17.0, 18.0], [3.0, 18.0], [3.0, 3.0], [17.0, 3.0]]  # Rectangular path
        }

        # AGV movement settings optimized for beam switching
        self.agv_movement = {
            'path_type': 'predefined',
            'update_interval': 0.1,  # More frequent updates
            'obstacle_avoidance': True,
            'min_distance': 1.0
        }

        # Initial AGV positions
        self.agv_positions = [
            [2.0, 2.0, self.agv_height],
            [17.0, 18.0, self.agv_height]
        ]

        # static scene configuration
        self.static_scene = {
            'material': 'concrete',  # Default material for walls
            'wall_thickness': 0.2,   # Wall thickness in meters
            'floor_material': 'concrete',
            'ceiling_material': 'concrete',
            'wall_height': 5.0,      # Height of walls in meters
            'reflectivity': 0.603815 # Concrete reflectivity at 28GHz
        }
        
        # Add PLY configuration if not already present
        self.ply_config = {
            'output_dir': 'meshes',
            'verify_files': True,
            'material_properties': {
                'concrete': {
                    'reflectivity': 0.603815,
                    'roughness': 0.1
                },
                'metal': {
                    'reflectivity': 0.087140,
                    'roughness': 0.05
                }
            }
        }

        # Optimized obstacle configuration for beam switching
        self.scene_objects = {
            'num_shelves': 5,
            'shelf_dimensions': [
                [4.0, 1.0, 4.0],
                [2.0, 1.0, 4.0],
                [1.0, 2.0, 5.0],
                [1.0, 2.0, 5.0],
                [4.0, 1.0, 5.0]
            ],
            'shelf_material': 'metal',
            'shelf_positions': [
                [4.0, 4.0, 0.0],
                [12.0, 4.0, 0.0],
                [4.0, 10.0, 0.0],
                [14.0, 12.0, 0.0],
                [8.0, 16.0, 0.0]
            ],
            'shelf_orientation': [0.0, 0.0, 0.0]
        }

        # Enhanced ray tracing configuration for better beam switching
        self.ray_tracing = {
            'max_depth': 4,  # Reduced for faster computation
            'method': "fibonacci",
            'num_samples': 2000,
            'diffraction': True,
            'scattering': True,
            'los': True,
            'reflection': True,
            'ris': False,
            'scene_type': self.scene_type,
            'scat_keep_prob': 1.0,
            'edge_diffraction': True
        }

        # Optimized beamforming configuration for switching demonstration
        self.beamforming = {
            'num_beams': 16,  # Reduced for faster switching
            'beam_width': 15,
            'max_steering_angle': 60,
            'adaptation_interval': 0.05,  # Faster adaptation
            'min_snr_threshold': 5.0,  # More sensitive threshold
            'blockage_detection': True,
            'beam_switching': {
                'enabled': True,
                'switching_threshold': 3.0,  # More sensitive
                'hysteresis': 1.0,  # Reduced for faster switching
            },
            'codebook': {
                'type': 'DFT',
                'size': 32,
                'oversampling': 2
            }
        }

        # Simulation parameters
        self.simulation = {
            'time_step': 0.001,
            'snr_range': [-20, 40],
            'channel_estimation_error': 0.1
        }

        # Basic camera configuration for visualization
        self.cameras = {
            'top': {
                'position': [10.0, 10.0, 30.0],
                'look_at': [10.0, 10.0, 0.0],
                'up': [0.0, 1.0, 0.0],
                'fov': 70.0
            }
        }