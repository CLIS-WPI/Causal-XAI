import tensorflow as tf
from sionna.constants import SPEED_OF_LIGHT

class SmartFactoryConfig:
    """Configuration class for Smart Factory Channel Simulation using Ray Tracing"""
    def __init__(self):
        # Basic simulation parameters
        self.num_time_steps = 100
        self.sampling_frequency = 1000  # in Hz
        self.batch_size = 1
        self.dtype = tf.complex64
        self.seed = 42

        # Room dimensions [m]
        self.room_dim = [20.0, 20.0, 5.0]  # Length x Width x Height
        
        # Frequency configuration
        self.carrier_frequency = 28e9  # 28 GHz
        self.wavelength = SPEED_OF_LIGHT/self.carrier_frequency
        
        # Base station configuration
        self.bs_position = [10.0, 0.5, 4.5]
        self.bs_orientation = [0.0, 0.0, 0.0]
        self.bs_array = [16, 4]  # 16x4 UPA
        self.bs_array_spacing = 0.5 * self.wavelength  # Half-wavelength spacing
        
        # RIS configuration
        self.ris_position = [10.0, 19.5, 2.5]
        self.ris_orientation = [0.0, 0.0, 0.0]
        self.ris_elements = [8, 8]  # 8x8 elements
        self.ris_spacing = 0.5 * self.wavelength
        
        # AGV configuration
        self.num_agvs = 2
        self.agv_height = 0.5
        self.agv_speed = 0.83  # 3 km/h in m/s
        self.agv_array = [1, 1]  # Single antenna for AGVs
        self.agv_array_spacing = 0.5 * self.wavelength
        
        # Ray tracing specific parameters
        self.ray_tracing = {
            'max_depth': 5,  # Maximum number of reflections
            'diffraction': True,  # Enable diffraction
            'scattering': True,  # Enable scattering
            'min_path_gain': -200,  # dB, minimum path gain to consider
            'scene_type': "indoor"
        }
        
        # Material properties for ray tracing
        self.materials = {
            'walls': {
                'permittivity': 4.5,  # Concrete
                'conductivity': 0.01
            },
            'metal_shelves': {
                'permittivity': 1.0,
                'conductivity': 1e7  # High conductivity for metal
            }
        }

        # Beamforming configuration
        self.beamforming = {
            'num_beams': 64,  # Number of beams in codebook
            'codebook_type': 'DFT',  # Type of codebook
            'beam_elevation': [-60, 60],  # Elevation angle range
            'beam_azimuth': [-60, 60]  # Azimuth angle range
        }

        # RIS optimization parameters
        self.ris_config = {
            'phase_resolution': 2,  # bits for phase resolution
            'optimization_method': 'gradient',
            'update_interval': 10  # Update RIS every N time steps
        }

        # Energy efficiency parameters
        self.energy = {
            'beam_scan_power': 0.1,  # Power per beam scan [W]
            'ris_config_power': 0.5,  # Power for RIS configuration [W]
            'rf_chain_power': 1.0,   # Power per RF chain [W]
            'power_amplifier_efficiency': 0.3
        }
        
        # Causal analysis configuration
        self.causal = {
            'variables': {
                'agv_position': True,
                'los_condition': True,
                'beam_choice': True,
                'ris_state': True,
                'path_gains': True
            },
            'analysis_interval': 5,  # Analyze every N time steps
            'min_correlation_threshold': 0.3
        }

        # Scene objects configuration
        self.scene_objects = {
            'num_shelves': 5,
            'shelf_dimensions': [2.0, 0.5, 2.0],  # Length x Width x Height
            'shelf_material': 'metal_shelves'
        }

        # Simulation quality parameters
        self.simulation = {
            'time_step': 0.001,  # Simulation time step [s]
            'snr_range': [-10, 30],  # SNR range to simulate [dB]
            'channel_estimation_error': 0.1  # Channel estimation error variance
        }