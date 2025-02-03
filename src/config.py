import tensorflow as tf

class SmartFactoryConfig:
    """Configuration class for Smart Factory Channel Simulation"""
    def __init__(self):
        
        # Environment Parameters
        self.room_dim = [20.0, 20.0, 5.0]  # Room dimensions [m]
        self.carrier_frequency = 28e9  # 28 GHz
        self.num_time_steps = 100
        self.sampling_frequency = 1000  # Hz
        self.dtype = tf.complex64
        
        # Base Station Parameters
        self.bs_array = [16, 4]  # UPA configuration
        self.bs_position = [0.0, 0.0, 4.5]  # BS mounted near ceiling
        self.bs_orientation = [0.0, 0.0, 0.0]  # Default orientation
        
        # RIS Parameters
        self.ris_elements = [8, 8]  # 8x8 RIS panel
        self.ris_position = [0.0, 20.0, 2.5]  # North wall
        self.ris_orientation = [0.0, 0.0, 0.0]  # Default orientation
        
        # AGV Parameters
        self.num_agvs = 2
        self.agv_height = 0.5  # meters
        self.agv_speed = 3.0/3.6  # 3 km/h to m/s
        self.min_speed = 0.0  # m/s
        self.max_speed = 3.0/3.6  # m/s
        
        # Channel Parameters
        self.model = "A"  # CDL-A model
        self.delay_spread = 100e-9  # 100ns delay spread
        self.num_paths = 23  # Number of paths (from CDL-A)
        self.scenario = "InF-SL"  # Indoor Factory Shopping Line
        
        # Simulation Parameters
        self.batch_size = 32
        self.seed = 42
        
        # Obstacle Parameters
        self.num_shelves = 5
        self.shelf_dims = [2.0, 1.0, 4.0]  # Length, width, height
        self.shelf_positions = [
            [5.0, 5.0, 0.0],
            [5.0, 15.0, 0.0],
            [10.0, 10.0, 0.0],
            [15.0, 5.0, 0.0],
            [15.0, 15.0, 0.0]
        ]

        # Additional parameters needed for scene setup
        self.num_tx = 1  # Number of transmitters
        self.num_rx = 2  # Number of receivers (AGVs)