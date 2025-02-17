import tensorflow as tf
from sionna.constants import SPEED_OF_LIGHT
import os
class SmartFactoryConfig:
    """Configuration class for Smart Factory Channel Simulation using Ray Tracing"""
    def __init__(self):
        self.num_time_steps = 100
        self.sampling_frequency = tf.cast(5000, tf.float32)
        self.batch_size = 1
        self.dtype = tf.complex64
        self.real_dtype = tf.float32  # Add this for real number operations
        self.seed = 42
        self.num_subcarriers = 128 #Number of OFDM subcarriers
        self.subcarrier_spacing = 15e3  # 15 kHz subcarrier spacing (typical for 5G)
        self.scene_type = "indoor"  # Explicit scene type at top level
        #self.ris_pattern = "iso"
        #self.ris_polarization = "V"
        # Room dimensions [m]
        self.room_dim = [20.0, 20.0, 5.0]  # Length x Width x Height
        # Frequency configuration
        self.carrier_frequency = tf.cast(28e9, tf.float32)  # Cast to float32
        self.wavelength = tf.cast(SPEED_OF_LIGHT/self.carrier_frequency, tf.float32)
##############################################################################        
        # Enhanced Base station configuration
        self.bs_position = [10.0, 0.5, 4.5]  # Center of room, near ceiling
        self.bs_orientation = [0.0, 0.0, -90.0]  # Facing downward
        self.bs_array = [16, 4]  # 16x4 UPA
        self.bs_array_spacing = 0.5 * self.wavelength
        self.bs_array_pattern = "tr38901"  # Added antenna pattern type
        self.bs_polarization = "VH"        # Added polarization config
##############################################################################        
        # Enhanced Material properties
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
                'roughness': 0.2e-3,  # Smoother surface
                'scattering_coefficient': 0.3,
                'xpd_coefficient': 15.0
            }
        }
##############################################################################        
        # ==========================
        # AGV Configuration Section
        # ==========================
        #  AGV configuration
        self.num_agvs = 2
        self.agv_height = 0.5
        self.agv_speed = 0.83
        self.agv_array = [1, 1]          # you can Increased to 2x2 array for better reception
        self.agv_array_spacing = 0.5 * self.wavelength
        self.rx_array_spacing = 0.5 * self.wavelength
        self.agv_array_pattern = "tr38901"     # iso bud##Added antenna pattern type
        self.agv_polarization = "VH"           # Added polarization config
        # Define AGV physical dimensions (size)
        self.agv_dimensions = [1.0, 1.0, 0.5]  # AGV size (L × W × H)

        # Define AGV trajectories (predefined movement paths)
        # AGV 1: Moves back and forth in a straight line (simple path, periodic blockages)
        # AGV 2: Moves in a rectangular loop around obstacles (ensures beamforming adjustments)
        # Define AGV trajectories (updated)
        self.agv_trajectories = {
            'agv_1': [
                [2.0, 2.0], [18.0, 2.0], [2.0, 2.0]  # AGV 1: Back & forth path
            ],
            'agv_2': [
                [17.0, 18.0], [3.0, 18.0], [3.0, 3.0], [17.0, 3.0], [17.0, 18.0]  # AGV 2: Curved path
            ]
        }

        # AGV movement settings
        # - AGVs will follow predefined paths ('predefined')
        # - Their positions will be updated at each time step (0.1s interval)
        # - They will avoid obstacles when necessary (obstacle_avoidance=True)
        self.agv_movement = {
            'path_type': 'predefined',    # Use predefined paths instead of random movement
            'update_interval': 0.05,      # How often AGVs update their positions
            'obstacle_avoidance': True,   # Enable obstacle avoidance logic
            'min_distance': 1.0           # Minimum allowable distance from an obstacle
        }

        # Initial positions of AGVs before movement starts
        # AGV 1 starts at (2,2) and follows a straight-line path
        # AGV 2 starts at (5,5) and follows a looped trajectory
        self.agv_positions = [
            [2.0, 2.0, self.agv_height],  # AGV 1 initial position
            [5.0, 5.0, self.agv_height]   # AGV 2 initial position
        ]

############################################################################
        # ==========================
        # static metallic shelves (obstacles) Section
        # ==========================

        # Define static metallic shelves (obstacles) that block LoS periodically
        # These obstacles match the plotting code (size & position)
        self.scene_objects = {
            'num_shelves': 5,
            'shelf_dimensions': [
                [4.0, 1.0, 4.0],  # Obstacle 1: (4m × 1m × 4m)
                [2.0, 1.0, 4.0],  # Obstacle 2: (2m × 1m × 4m)
                [1.0, 2.0, 5.0],  # Obstacle 3: (1m × 2m × 5m) - Rotated
                [1.0, 2.0, 5.0],  # Obstacle 4: (1m × 2m × 5m) - Rotated
                [4.0, 1.0, 5.0]   # Obstacle 5: (4m × 1m × 5m)
            ],
            'shelf_material': 'metal',
            'shelf_positions': [
                [4.0, 4.0, 0.0],   # Obstacle 1
                [12.0, 4.0, 0.0],  # Obstacle 2
                [4.0, 10.0, 0.0],  # Obstacle 3
                [14.0, 12.0, 0.0], # Obstacle 4
                [8.0, 16.0, 0.0]   # Obstacle 5
            ],
            'shelf_orientation': [0.0, 0.0, 0.0]  # Default: No rotation
        }

######################################################################################
        # Update ray tracing parameters for better path detection
        self.ray_tracing = {
            'max_depth': 6,            # Reduced for more focused paths
            'method': "fibonacci",
            'num_samples': 5000,       # Doubled for better coverage
            'diffraction': True,
            'scattering': True,
            'los': True,
            'reflection': True,
            'ris': False,
            'scene_type': self.scene_type,
            'scat_keep_prob': 1.0,      # Increased probability
            'edge_diffraction': True
        }
######################################################################################
        # Static scene configuration
        self.static_scene = {
            'walls': True,
            'floor': True,
            'ceiling': True,
            'wall_thickness': 0.2,
            'material': 'concrete',
            'diffraction_edges': True,
            'scene_type': self.scene_type
        }

        self.path_loss = {
        'los_model': 'InF_LoS',      # Indoor Factory LoS model
        'nlos_model': 'InF_NLoS',    # Indoor Factory NLoS model
        'shadow_fading': True,        # Enable shadow fading
        'ricean_factor': 10,          # K-factor for LoS conditions
        }
        
    
        # Add beamforming configuration
        self.beamforming = {
            'num_beams': 32,  # Number of possible beam directions
            'beam_width': 15, # Beam width in degrees
            'max_steering_angle': 60, # Maximum steering angle
            'adaptation_interval': 0.1, # Beam update interval (seconds)
            'min_snr_threshold': 10.0, # Minimum acceptable SNR (dB)
        }
        
        
        # Add causal analysis parameters
        self.causal = {
            'observation_window': 100,  # Number of samples for causal analysis
            'treatment_variables': ['beam_direction', 'obstacle_presence'],
            'outcome_variables': ['snr', 'throughput'],
            'confounders': ['agv_speed', 'distance_to_bs'],
            'effect_threshold': 0.3  # Minimum effect size to consider significant
        }

        self.simulation = {
            'time_step': 0.001,
            'snr_range': [-20, 40],
            'channel_estimation_error': 0.1
            }

        # PLY generator configuration
        self.ply_config = {
            'room_dims': self.room_dim,  # Use existing room dimensions
            'shelf_dims': self.scene_objects['shelf_dimensions'],  # Use existing shelf dimensions
            'shelf_positions': self.scene_objects['shelf_positions'],  # Use existing shelf positions
            'output_dir': os.path.join(os.path.dirname(__file__), 'meshes'),
            'material': 'concrete',  # default material for static structures
            'geometry_mapping': {
                'floor': {'z': 0},
                'ceiling': {'z': self.room_dim[2]},
                'wall_xp': {'x': self.room_dim[0]},
                'wall_xm': {'x': 0},
                'wall_yp': {'y': self.room_dim[1]},
                'wall_ym': {'y': 0}
            }
        }

        self.shap = {
            'analysis': {
                'num_background_samples': 100,
                'min_samples_required': 10,
                'max_display_features': 20
            },
            'visualization': {
                'plot_type': 'bar',
                'show_feature_importance': True,
                'max_display': 10,
                'figure_size': (10, 6)
            },# In shap
            'features': {
                'channel_response': True,
                'los_condition': True,
                'agv_position': True,
                'ris_state': True
            }
        }
#############################################################################        
        # Add camera configurations
        self.cameras = {
            'top': {
                'position': [10.0, 10.0, 20.0],
                'look_at': [10.0, 10.0, 0.0]
            },
            'side': {
                'position': [30.0, 10.0, 5.0],
                'look_at': [10.0, 10.0, 0.0]
            },
            'corner': {
                'position': [20.0, 20.0, 10.0],
                'look_at': [10.0, 10.0, 0.0]
            }
        }
        # Update camera configurations
        self.cameras = {
            'top': {
                'position': [10.0, 10.0, 30.0],     # Higher up for better top view
                'look_at': [10.0, 10.0, 0.0],
                'up': [0.0, 1.0, 0.0],
                'fov': 70.0,                        # Wider field of view
                'filename': 'top_view.png'
            },
            'side': {
                'position': [40.0, 10.0, 8.0],      # Further back and higher
                'look_at': [10.0, 10.0, 2.5],
                'up': [0.0, 0.0, 1.0],
                'fov': 60.0,
                'filename': 'side_view.png'
            },
            'corner': {
                'position': [35.0, 35.0, 15.0],     # Higher and further for corner view
                'look_at': [10.0, 10.0, 2.5],
                'up': [0.0, 0.0, 1.0],
                'fov': 70.0,
                'filename': 'corner_view.png'
            }
        }

        self.render_config = {
            # Basic display settings
            'width': 1920,                    # Render width in pixels
            'height': 1080,                   # Render height in pixels
            'background_color': [0.8, 0.8, 0.8],  # RGB background color
            
            # Camera settings
            'fov': 120.0,                      # Field of view in degrees
            'near': 0.1,                      # Near clipping plane
            'far': 1000.0,                    # Far clipping plane
            
            # Rendering quality settings
            'samples': 64,                    # Number of samples per pixel
            'max_bounces': 4,                 # Maximum number of light bounces
            'quality': 'high',                # Rendering quality preset
            
            # Lighting settings
            'ambient_light': [0.1, 0.1, 0.1], # Ambient light color
            'exposure': 1.0,                  # Exposure value
            
            # Post-processing
            'gamma': 2.2,                     # Gamma correction
            'tone_mapping': True,             # Enable/disable tone mapping
            
            # Output settings
            'file_format': 'png',             # Output file format
            'transparent': False,             # Transparent background
            'dpi': 300                        # Dots per inch for output
        }
################################################################################