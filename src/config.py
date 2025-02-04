import tensorflow as tf

class SmartFactoryConfig:
    """Configuration class for Smart Factory Channel Simulation"""
    def __init__(self):
        self.num_time_steps = 100
        self.sampling_frequency = 1000  # in Hz
        self.num_agvs = 2  # Number of AGVs
        # Room dimensions
        self.room_dim = [20.0, 20.0, 5.0]  # Length x Width x Height
        
        # Base station configuration
        self.bs_position = [10.0, 0.5, 4.5]
        self.bs_orientation = [0.0, 0.0, 0.0]
        self.bs_array = [16, 4]  # 16x4 UPA
        
        # RIS configuration
        self.ris_position = [10.0, 19.5, 2.5]
        self.ris_orientation = [0.0, 0.0, 0.0]
        self.ris_elements = [8, 8]  # 8x8 elements
        
        # AGV configuration
        self.num_agvs = 2
        self.agv_height = 0.5
        self.agv_speed = 0.83  # 3 km/h in m/s
        
        # Channel configuration
        self.carrier_frequency = 28e9  # 28 GHz
        self.sampling_frequency = 1.0  # 1 Hz sampling
        self.num_time_steps = 10
        self.batch_size = 1
        self.model = "InF-SL"
        self.scenario = "sparse"
        self.dtype = tf.complex64
        
        # Random seed
        self.seed = 42

        # Additional parameters needed for scene setup
        self.num_tx = 1  # Number of transmitters
        self.num_rx = 2  # Number of receivers (AGVs)

        # Add beam configuration parameters
        self.num_beams = 64  # Number of beams in codebook
        self.xai_pruning_factor = 0.3  # Default pruning factor (30% reduction)

        # Add parameters for causal analysis
        self.causal_variables = {
            'agv_position': True,
            'los_condition': True,
            'beam_choice': True
        }