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
        self.num_subcarriers = 128 #Number of OFDM subcarriers
        self.subcarrier_spacing = 15e3  # 15 kHz subcarrier spacing (typical for 5G)

        # Room dimensions [m]
        self.room_dim = [20.0, 20.0, 5.0]  # Length x Width x Height

        # Frequency configuration
        self.carrier_frequency = tf.cast(28e9, tf.float32)  # Cast to float32
        self.wavelength = tf.cast(SPEED_OF_LIGHT/self.carrier_frequency, tf.float32)
        
        # Enhanced Base station configuration
        self.bs_position = [10.0, 0.5, 4.5]
        self.bs_orientation = [0.0, 0.0, 0.0]
        self.bs_array = [16, 4]  # 16x4 UPA
        self.bs_array_spacing = 0.5 * self.wavelength
        self.bs_array_pattern = "tr38901"  # Added antenna pattern type
        self.bs_polarization = "VH"        # Added polarization config
        
        # Enhanced RIS configuration
        self.ris_position = [10.0, 19.5, 2.5]
        self.ris_orientation = [0.0, 0.0, 0.0]
        self.ris_elements = [8, 8]  # 8x8 elements
        self.ris_spacing = 0.5 * self.wavelength
        self.ris_modes = 1  # Number of modes for RIS
        
        # Enhanced AGV configuration
        self.num_agvs = 2
        self.agv_height = 0.5
        self.agv_speed = 0.83
        self.agv_array = [1, 1]
        self.agv_array_spacing = 0.5 * self.wavelength
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

        # Enhanced Ray tracing parameters
        self.ray_tracing = {
            'max_depth': 5,
            'method': "fibonacci",          # Added ray tracing method
            'num_samples': int(1e6),        # Added number of rays
            'diffraction': True,
            'scattering': True,
            'los': True,                    # Added LOS path option
            'reflection': True,             # Added reflection option
            'ris': True,                    # Added RIS path option
            'scene_type': "indoor"
        }

        # Scene objects with enhanced configuration
        self.scene_objects = {
            'num_shelves': 5,
            'shelf_dimensions': [2.0, 1.0, 4.0],  # Updated dimensions
            'shelf_material': 'metal',
            'shelf_positions': [
                [5.0, 5.0, 1.5],
                [15.0, 5.0, 1.5],
                [10.0, 10.0, 1.5],
                [5.0, 15.0, 1.5],
                [15.0, 15.0, 1.5]
            ]
        }

        # Static scene configuration
        self.static_scene = {
            'walls': True,
            'floor': True,
            'ceiling': True,
            'wall_thickness': 0.2,
            'material': 'concrete',
            'scene_type': 'indoor'
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