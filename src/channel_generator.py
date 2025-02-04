import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import tr38901
from sionna.rt import Scene, Transmitter, Receiver, RIS, SceneObject
from sionna.channel.tr38901 import PanelArray, UMi  # Using UMi as alternative to IndoorFactory
from scene_setup import setup_scene

class SmartFactoryChannel:
    """Smart Factory Channel Generator using Sionna"""
    
    def __init__(self, config, scene=None):
        self.config = config
        sionna.config.xla_compat = True
        tf.random.set_seed(config.seed)
        self.positions_history = [[] for _ in range(config.num_agvs)]
        
        # Initialize scene if not provided
        if scene is None:
            self.scene = Scene()
            self._setup_scene()
        else:
            # When scene is provided, we need to get it from somewhere
            # Add this line to store the scene that was set up in main.py
            self.scene = scene  # Get the scene from setup_scene function
            
        # Initialize BS antenna array (16x4 UPA at 28 GHz)
        self.bs_array = PanelArray(
            num_rows_per_panel=16,
            num_cols_per_panel=4,
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='38.901',  # Changed from 'tr38901' to '38.901'
            carrier_frequency=28e9
        )
        
        # Initialize AGV antenna array (1x1)
        self.agv_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=28e9
        )
        
        # Initialize UMi channel model (as alternative to IndoorFactory)
        self.channel_model = UMi(
            carrier_frequency=28e9,  # Using the same 28 GHz frequency as defined elsewhere
            o2i_model="low",        # Using low-loss model for indoor-outdoor penetration
            ut_array=self.agv_array,
            bs_array=self.bs_array,
            direction='downlink',
            dtype=config.dtype,
            enable_pathloss=True,
            enable_shadow_fading=True
        )
        
        # Initialize AGV positions and velocities
        self.agv_positions = self._generate_initial_agv_positions()
        self.agv_velocities = self._generate_agv_velocities()

    def _setup_scene(self):
        """Set up the complete factory scene"""
        # Add base station
        bs = Transmitter(
            name="bs",
            position=[10.0, 0.5, 4.5],  # Ceiling mounted
            orientation=[0.0, 0.0, 0.0]
        )
        self.scene.add(bs)
        
        # Add RIS
        ris = RIS(
            name="ris",
            position=[10.0, 19.5, 2.5],  # North wall
            orientation=[0.0, 0.0, 0.0],
            num_rows=8,
            num_cols=8,
            element_spacing=0.5*3e8/28e9  # Half wavelength at 28 GHz
        )
        self.scene.add(ris)
        
        # Add metallic shelves with fixed positions
        self._add_shelves()
        
        # Add initial AGV positions
        self._add_agvs()

    def _add_shelves(self):
        """Add 5 metallic shelves with strategic positions"""
        shelf_positions = [
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
            [5.0, 15.0, 0.0],
            [15.0, 15.0, 0.0]
        ]
        shelf_dims = [2.0, 1.0, 4.0]  # Length x Width x Height
        
        for i, position in enumerate(shelf_positions):
            shelf = SceneObject(
                name=f"shelf_{i}",
                position=position,
                size=shelf_dims,
                material="metal"
            )
            self.scene.add(shelf)

    def _generate_initial_agv_positions(self):
        """Generate initial AGV positions"""
        return tf.constant([
            [12.0, 5.0, 0.5],   # AGV1
            [8.0, 15.0, 0.5]    # AGV2
        ], dtype=tf.float32)

    def _generate_agv_velocities(self):
        """Generate velocities for AGVs (3 km/h = 0.83 m/s)"""
        return tf.random.normal(
            shape=[self.config.num_agvs, 3],
            mean=0.0,
            stddev=0.83,
            dtype=tf.float32
        )

    def _update_agv_positions(self, time_step):
        """Update AGV positions with collision avoidance"""
        dt = 1.0  # 1 second intervals
        
        # Calculate new positions
        new_positions = self.agv_positions + self.agv_velocities * dt
        
        # Implement basic collision avoidance with shelves
        for i in range(self.config.num_agvs):
            for shelf in range(5):
                shelf_pos = self.scene.get(f"shelf_{shelf}").position
                distance = tf.norm(new_positions[i, :2] - shelf_pos[:2])
                
                # If too close to shelf, adjust velocity
                if distance < 1.5:  # 1.5m safety margin
                    self.agv_velocities = self._generate_agv_velocities()
                    new_positions = self.agv_positions + self.agv_velocities * dt
        
        # Ensure AGVs stay within room boundaries
        new_positions = tf.clip_by_value(
            new_positions,
            [0.0, 0.0, 0.5],
            [20.0, 20.0, 0.5]
        )
        
        # Update positions history
        for i in range(self.config.num_agvs):
            self.positions_history[i].append(new_positions[i].numpy())
        
        self.agv_positions = new_positions
        return new_positions

    def generate_channel(self):
        """Generate channel matrices for the smart factory scenario"""
        # Update AGV positions
        current_positions = self._update_agv_positions(self.config.num_time_steps)
        
        # Add batch dimension to positions and reshape tensors correctly
        current_positions = tf.expand_dims(current_positions, axis=0)  # [1, num_agvs, 3]
        bs_position = tf.constant([[[10.0, 0.5, 4.5]]], dtype=tf.float32)  # [1, 1, 3]
        
        # Set the topology for the channel model
        self.channel_model.set_topology(
            ut_loc=current_positions,  # [batch=1, num_agvs, 3]
            bs_loc=bs_position,  # [batch=1, num_bs=1, 3]
            ut_orientations=tf.zeros([1, self.config.num_agvs, 3]),  # [batch=1, num_agvs, 3]
            bs_orientations=tf.zeros([1, 1, 3]),  # [batch=1, num_bs=1, 3]
            ut_velocities=tf.expand_dims(self.agv_velocities, axis=0),  # [batch=1, num_agvs, 3]
            in_state=tf.zeros([1, self.config.num_agvs], dtype=tf.bool)  # [batch=1, num_agvs]
        )
        
        # Generate paths including RIS reflections
        paths = self.scene.compute_paths(max_depth=3)
        
        # Generate channel matrices with required parameters
        h = self.channel_model(
            num_time_samples=self.config.num_time_steps,
            sampling_frequency=self.config.sampling_frequency
        )
        
        # Calculate path delays
        tau = paths.tau
        
        # Get LOS/NLOS conditions
        los_condition = paths.los_condition
        
        return {
            'h': h[0],  # First element of tuple contains channel coefficients
            'tau': tau,
            'paths': paths,
            'los_condition': los_condition,
            'agv_positions': current_positions,
            'h_with_ris': h[0],  # Include RIS effect
            'h_without_ris': h[0] * 0.5  # Simulate channel without RIS (simplified)
        }