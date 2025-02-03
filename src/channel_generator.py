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

        if not scene_provided:
            self.scene = Scene()
            self._setup_scene()
        
        # Initialize antenna arrays
        self.bs_array = PanelArray(
            num_rows_per_panel=config.bs_array[0],
            num_cols_per_panel=config.bs_array[1],
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='38.901',
            carrier_frequency=config.carrier_frequency
        )
    
        self.agv_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=config.carrier_frequency
        )
    
        # Initialize Indoor Factory channel model instead of CDL
        self.channel_model = IndoorFactory(
            scenario=config.scenario,  # "InF-SL"
            ut_array=self.agv_array,
            bs_array=self.bs_array,
            direction='downlink',
            dtype=config.dtype
        )
        
        # Initialize AGV positions and velocities
        self.agv_positions = self._generate_agv_positions()
        self.agv_velocities = self._generate_agv_velocities()

    def _setup_scene(self):
        """Set up the complete scene with all components"""
        # Add transmitter (BS)
        tx_position = tf.constant([self.config.bs_position], dtype=tf.float32)
        tx_orientation = tf.constant([self.config.bs_orientation], dtype=tf.float32)
        self.scene.add(Transmitter(name="tx",
                                position=tx_position,
                                orientation=tx_orientation))
        
        # Add RIS with proper physics-based modeling
        self._add_ris()
        
        # Add metallic shelves as obstacles
        self._add_shelves()
        
        # Add initial AGV positions
        self._add_agvs()

    def _add_ris(self):
        """Add RIS to the scene with proper configuration"""
        self.scene.add(RIS(
            name="ris",
            position=self.config.ris_position,
            orientation=self.config.ris_orientation,
            num_rows=self.config.ris_elements[0],
            num_cols=self.config.ris_elements[1],
            element_spacing=0.5*self.config.carrier_frequency/3e8, # Half wavelength spacing
            phase_offset=0.0
        ))

    def _add_shelves(self):
        """Add metallic shelves as Cuboid objects"""
        for i, pos in enumerate(self.config.shelf_positions):
            self.scene.add(Cuboid(
                name=f"shelf_{i}",
                position=pos,
                size=self.config.shelf_dims,
                material="metal"  # Using metallic material properties
            ))

    def _add_agvs(self):
        """Add AGVs as receivers to the scene"""
        for i in range(self.config.num_agvs):
            self.scene.add(Receiver(
                name=f"agv_{i}",
                position=self.agv_positions[i],
                orientation=[0.0, 0.0, 0.0]
            ))

    def _generate_agv_velocities(self):
        """Generate random velocities for AGVs"""
        return tf.random.uniform(
            shape=[self.config.num_agvs, 3],
            minval=-self.config.agv_speed,
            maxval=self.config.agv_speed,
            dtype=tf.float32
        )

    def _generate_agv_positions(self):
        """Generate initial AGV positions"""
        return tf.random.uniform(
            shape=[self.config.num_agvs, 3],
            minval=[0.0, 0.0, self.config.agv_height],
            maxval=[self.config.room_dim[0], 
                    self.config.room_dim[1], 
                    self.config.agv_height],
            dtype=tf.float32
        )

    def _update_agv_positions(self, time_step):
        """Update AGV positions based on velocities"""
        dt = 1.0 / self.config.sampling_frequency
        new_positions = self.agv_positions + self.agv_velocities * dt * time_step
        
        # Ensure AGVs stay within room boundaries
        new_positions = tf.clip_by_value(
            new_positions,
            [0.0, 0.0, self.config.agv_height],
            [self.config.room_dim[0], self.config.room_dim[1], self.config.agv_height]
        )
        
        # Update receiver positions in scene
        for i in range(self.config.num_agvs):
            self.scene.get(f"agv_{i}").position = new_positions[i]
        
        return new_positions

    def _check_los_condition(self):
        """Check Line-of-Sight conditions using ray tracing"""
        paths = self.scene.compute_paths()
        los_condition = tf.zeros([self.config.batch_size, self.config.num_agvs], dtype=tf.bool)
        
        # Check if direct paths exist between transmitter and receivers
        for path in paths:
            if len(path.interactions) == 0:  # Direct path with no interactions
                los_condition = tf.logical_or(los_condition, True)
        
        return los_condition

    def generate_channel(self):
        """Generate channel matrices for the smart factory scenario"""
        # Update AGV positions for this timestep
        current_positions = self._update_agv_positions(self.config.num_time_steps)
        
        # Store positions for trajectory analysis
        for i in range(self.config.num_agvs):
            self.positions_history[i].append(current_positions[i].numpy())
        
        # Generate channel with RIS
        h_with_ris, tau = self.channel_model(
            batch_size=self.config.batch_size,
            num_time_steps=self.config.num_time_steps,
            sampling_frequency=self.config.sampling_frequency
        )
        
        # Temporarily disable RIS and generate channel without it
        ris = self.scene.get("ris")
        self.scene.remove("ris")
        h_without_ris, _ = self.channel_model(
            batch_size=self.config.batch_size,
            num_time_steps=self.config.num_time_steps,
            sampling_frequency=self.config.sampling_frequency
        )
        # Restore RIS
        self.scene.add(ris)
        
        # Get ray-tracing paths and LOS conditions
        paths = self.scene.compute_paths()
        los_condition = self._check_los_condition()
        
        return {
            'h_with_ris': h_with_ris,
            'h_without_ris': h_without_ris,
            'tau': tau,
            'paths': paths,
            'los_condition': los_condition,
            'agv_positions': current_positions
        }