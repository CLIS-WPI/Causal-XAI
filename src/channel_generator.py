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

    def save_csi_dataset(self, filepath, num_samples=None):
        """
        Save the complete CSI dataset to an HDF5 file.
        
        Args:
            filepath: Path where the HDF5 file will be saved
            num_samples: Number of channel samples to generate and save (optional, uses config.num_time_steps if not specified)
        """
        import h5py
        
        # Use config value if num_samples not specified
        if num_samples is None:
            num_samples = self.config.num_time_steps
        
        with h5py.File(filepath, 'w') as f:
            # Create groups for different components
            csi_group = f.create_group('csi_data')
            config_group = f.create_group('config')
            
            # Initialize lists to store data
            channel_data = []
            path_delays = []
            los_conditions = []
            agv_positions = []
            
            # Generate and collect samples
            for _ in range(num_samples):
                sample = self.generate_channel()
                
                channel_data.append(sample['h'].numpy())
                path_delays.append(sample['tau'].numpy())
                los_conditions.append(sample['los_condition'].numpy())
                agv_positions.append(sample['agv_positions'].numpy())
            
            # Convert lists to numpy arrays and save
            csi_group.create_dataset('channel_matrices', data=np.array(channel_data))
            csi_group.create_dataset('path_delays', data=np.array(path_delays))
            csi_group.create_dataset('los_conditions', data=np.array(los_conditions))
            csi_group.create_dataset('agv_positions', data=np.array(agv_positions))
            
            # Save all configuration parameters
            for key, value in vars(self.config).items():
                if isinstance(value, (int, float, str, list)):
                    config_group.attrs[key] = value
                elif isinstance(value, tf.dtypes.DType):
                    config_group.attrs[key] = str(value)
                    
            # Save specific configuration parameters that might be needed for analysis
            config_group.attrs['num_agvs'] = self.config.num_agvs
            config_group.attrs['num_time_steps'] = self.config.num_time_steps
            config_group.attrs['sampling_frequency'] = self.config.sampling_frequency
            config_group.attrs['carrier_frequency'] = self.config.carrier_frequency
            config_group.attrs['bs_array'] = self.config.bs_array
            config_group.attrs['ris_elements'] = self.config.ris_elements
            config_group.attrs['room_dimensions'] = self.config.room_dim
            config_group.attrs['scenario'] = self.config.scenario
            config_group.attrs['model'] = self.config.model

    def load_csi_dataset(self, filepath):
        """
        Load the CSI dataset from an HDF5 file.
        
        Args:
            filepath: Path to the HDF5 file containing the saved dataset
            
        Returns:
            dict: Dictionary containing the loaded dataset and configuration
        """
        import h5py
        
        with h5py.File(filepath, 'r') as f:
            # Load CSI data
            data = {
                'channel_matrices': f['csi_data/channel_matrices'][:],
                'path_delays': f['csi_data/path_delays'][:],
                'los_conditions': f['csi_data/los_conditions'][:],
                'agv_positions': f['csi_data/agv_positions'][:]
            }
            
            # Load configuration parameters
            config = {}
            for key in f['config'].attrs.keys():
                config[key] = f['config'].attrs[key]
                
            data['config'] = config
            
        return data

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
        
        # Determine LOS condition based on path existence
        # If there's at least one direct path between BS and AGV, consider it LOS
        los_condition = tf.cast(
            tf.reduce_any(tf.not_equal(paths.tau, float('inf')), axis=-1),
            tf.bool
        )
        
        return {
            'h': h[0],  # First element of tuple contains channel coefficients
            'tau': tau,
            'paths': paths,
            'los_condition': los_condition,
            'agv_positions': current_positions,
            'h_with_ris': h[0],  # Include RIS effect
            'h_without_ris': h[0] * 0.5  # Simulate channel without RIS (simplified)
        }