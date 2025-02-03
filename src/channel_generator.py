import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import tr38901
from sionna.rt import Scene, Transmitter, Receiver, RIS, SceneObject
from sionna.channel.tr38901 import IndoorFactory, PanelArray

class SmartFactoryChannel:
    """Smart Factory Channel Generator using Sionna"""
    
    def __init__(self, config, scene_provided=False):
        self.config = config
        sionna.config.xla_compat = True
        tf.random.set_seed(config.seed)
        self.positions_history = [[] for _ in range(config.num_agvs)]
        
        # Initialize scene if not provided
        if not scene_provided:
            self.scene = Scene()
            self._setup_scene()
        
        # Initialize BS antenna array (16x4 UPA at 28 GHz)
        self.bs_array = PanelArray(
            num_rows_per_panel=16,
            num_cols_per_panel=4,
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='38.901',
            carrier_frequency=28e9  # 28 GHz
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
        
        # Initialize Indoor Factory channel model
        self.channel_model = IndoorFactory(
            scenario="InF-SL",  # Sparse clutter
            ut_array=self.agv_array,
            bs_array=self.bs_array,
            direction='downlink',
            dtype=config.dtype
        )
        
        # Initialize AGV positions and velocities
        self.agv_positions = self._generate_agv_positions()
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
        
        # Add metallic shelves
        self._add_shelves()
        
        # Add initial AGV positions
        self._add_agvs()

    def _add_shelves(self):
        """Add 5 metallic shelves with random positions"""
        shelf_dims = [2.0, 1.0, 4.0]  # Length x Width x Height
        for i in range(5):
            position = tf.random.uniform(
                shape=[3],
                minval=[5.0, 5.0, 0.0],
                maxval=[15.0, 15.0, 0.0]
            )
            position = tf.concat([position[:2], tf.constant([2.0])], axis=0)  # Fixed height
            
            shelf = SceneObject(
                name=f"shelf_{i}",
                position=position,
                size=shelf_dims,
                material="metal"
            )
            self.scene.add(shelf)

    def _add_agvs(self):
        """Add AGVs as receivers"""
        initial_positions = [
            [12.0, 5.0, 0.5],  # AGV1
            [8.0, 15.0, 0.5]   # AGV2
        ]
        
        for i, pos in enumerate(initial_positions):
            self.scene.add(Receiver(
                name=f"agv_{i}",
                position=pos,
                orientation=[0.0, 0.0, 0.0]
            ))

    def _generate_agv_velocities(self):
        """Generate velocities for AGVs (3 km/h = 0.83 m/s)"""
        return tf.random.normal(
            shape=[self.config.num_agvs, 3],
            mean=0.0,
            stddev=0.83,
            dtype=tf.float32
        )

    def _update_agv_positions(self, time_step):
        """Update AGV positions based on velocities"""
        dt = 1.0  # 1 second intervals
        new_positions = self.agv_positions + self.agv_velocities * dt
        
        # Ensure AGVs stay within room boundaries
        new_positions = tf.clip_by_value(
            new_positions,
            [0.0, 0.0, 0.5],  # Min bounds (fixed height)
            [20.0, 20.0, 0.5]  # Max bounds (fixed height)
        )
        
        # Update receiver positions in scene
        for i in range(self.config.num_agvs):
            self.scene.get(f"agv_{i}").position = new_positions[i]
            self.positions_history[i].append(new_positions[i].numpy())
        
        return new_positions

    def generate_channel(self):
        """Generate channel matrices for the smart factory scenario"""
        # Update AGV positions
        current_positions = self._update_agv_positions(self.config.num_time_steps)
        
        # Generate paths including RIS reflections
        paths = self.scene.compute_paths(max_depth=3)
        
        # Generate channel matrices
        h, tau = self.channel_model(
            batch_size=self.config.batch_size,
            num_time_steps=self.config.num_time_steps,
            sampling_frequency=self.config.sampling_frequency
        )
        
        # Get LOS/NLOS conditions
        los_condition = paths.los_condition
        
        return {
            'h': h,
            'tau': tau,
            'paths': paths,
            'los_condition': los_condition,
            'agv_positions': current_positions
        }