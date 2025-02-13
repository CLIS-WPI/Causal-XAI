import tensorflow as tf
from sionna.constants import SPEED_OF_LIGHT

class SmartFactoryConfig:
    """Configuration class for Smart Factory Channel Simulation using Ray Tracing"""
    def __init__(self):
        # Keep existing basic parameters
        self.num_time_steps = 100
        self.sampling_frequency = tf.cast(1000, tf.float32)
        self.batch_size = 1
        self.dtype = tf.complex64
        self.real_dtype = tf.float32  # Add this for real number operations
        self.seed = 42
        self.num_subcarriers = 1024 #Number of OFDM subcarriers
        self.subcarrier_spacing = 15e3  # 15 kHz subcarrier spacing (typical for 5G)
        self.scene_type = "indoor"  # Explicit scene type at top level
        self.ris_pattern = "iso"
        self.ris_polarization = "V"

        # Room dimensions [m]
        self.room_dim = [20.0, 20.0, 5.0]  # Length x Width x Height

        # Frequency configuration
        self.carrier_frequency = tf.cast(28e9, tf.float32)  # Cast to float32
        self.wavelength = tf.cast(SPEED_OF_LIGHT/self.carrier_frequency, tf.float32)
        
        # Enhanced Base station configuration
        self.bs_position = [10.0, 10.0, 4.5]  # Center of ceiling
        self.bs_orientation = [0.0, 0.0, -90.0]  # Facing down
        self.bs_array = [16, 4]  # 16x4 UPA
        self.bs_array_spacing = 0.5 * self.wavelength
        self.bs_array_pattern = "tr38901"  # Added antenna pattern type
        self.bs_polarization = "VH"        # Added polarization config
        
        # Enhanced RIS configuration
        self.ris_position = [10.0, 19.5, 2.5] # 3D position array [x, y, z]
        self.ris_orientation = [0.0, -90.0, 0.0]  # Facing south # 3D orientation array [rx, ry, rz] 
        self.ris_elements = [8, 8]  # 8x8 elements # 2D array [num_rows, num_cols]
        self.ris_spacing = 0.5 * self.wavelength # RIS element spacing
        self.ris_modes = 1  # Number of modes for RIS # Number of modes for RIS
        
        # Enhanced AGV configuration
        self.num_agvs = 2
        self.agv_height = 0.5
        self.agv_speed = 0.83
        self.agv_array = [1, 1]
        self.agv_array_spacing = 0.5 * self.wavelength
        self.rx_array_spacing = 0.5 * self.wavelength
        self.agv_array_pattern = "iso"     # Added antenna pattern type
        self.agv_polarization = "V"        # Added polarization config
        
        # Enhanced Material properties
        self.materials = {
            'concrete': {
                'name': "concrete",
                'relative_permittivity': 4.5,  # Changed from complex(4.5)
                'conductivity': 0.01,
                'scattering_coefficient': 0.2,
                'xpd_coefficient': 8.0
            },
            'metal': {
                'name': "metal",
                'relative_permittivity': 1.0,  # Changed from complex(1.0, -1e7)
                'conductivity': 1e7,
                'scattering_coefficient': 0.1,
                'xpd_coefficient': 10.0
            }
        }

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
        
        # Add AGV positions (could be initial positions)
        self.agv_positions = [
            [15.0, 15.0, self.agv_height],
            [10.0, 15.0, self.agv_height]
        ]
        
        self.agv_orientations = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]

        # Enhanced Ray tracing parameters
        self.ray_tracing = {
            'max_depth': 4,         
            'method': "fibonacci",
            'num_samples': 1024, 
            'diffraction': True,
            'scattering': False ,
            'los': True,
            'reflection': True,
            'ris': True,
            'scene_type': self.scene_type,
            'scat_keep_prob': 0.001,
            'edge_diffraction': False

        }

        # Material properties for ray tracing
        self.materials = {
            'concrete': {
                'name': 'concrete',
                'relative_permittivity': 5.31,
                'conductivity': 0.0462
            },
            'metal': {
                'name': 'metal',
                'relative_permittivity': 1.0,
                'conductivity': 1e7
            }
        }

        # Scene objects with enhanced configuration
        self.scene_objects = {
            'num_shelves': 5,
            'shelf_dimensions': [2.0, 1.0, 4.0],  
            'shelf_material': 'metal',
            'shelf_positions': [
                [5.0, 5.0, 0.0],    
                [15.0, 5.0, 0.0],
                [10.0, 10.0, 0.0],
                [5.0, 15.0, 0.0],
                [15.0, 15.0, 0.0]
            ]
        }

        # Static scene configuration
        self.static_scene = {
            'walls': True,
            'floor': True,
            'ceiling': True,
            'wall_thickness': 0.2,
            'material': 'concrete',
            'scene_type': self.scene_type
        }

        # Keep existing configurations
        self.beamforming = {
            'num_beams': 64,
            'codebook_type': 'DFT',
            'beam_elevation': [-60, 60],
            'beam_azimuth': [-60, 60]
        }

        self.ris_config = {
            'phase_resolution': 2,
            'optimization_method': 'gradient',
            'update_interval': 10
        }

        self.energy = {
            'beam_scan_power': 0.1,
            'ris_config_power': 0.5,
            'rf_chain_power': 1.0,
            'power_amplifier_efficiency': 0.3
        }
        
        self.causal = {
            # In causal
            'variables': {
                'agv_position': True,
                'los_condition': True,
                'beam_choice': True,
                'ris_state': True,
                'path_gains': True
            },
            'analysis_interval': 5,
            'min_correlation_threshold': 0.3
        }

        self.simulation = {
            'time_step': 0.001,
            'snr_range': [-10, 30],
            'channel_estimation_error': 0.1
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