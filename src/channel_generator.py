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

    def _generate_agv_waypoints(self):
        """Generate predefined waypoints for each AGV"""
        return [
            # AGV1: Loop around shelves
            [[12.0, 5.0], [12.0, 15.0], [8.0, 15.0], [8.0, 5.0]],
            # AGV2: Cross-diagonal
            [[8.0, 15.0], [12.0, 5.0], [15.0, 8.0], [5.0, 12.0]]
        ]

    def _update_agv_positions(self, time_step):
        """Update AGV positions using waypoint-based movement"""
        if not hasattr(self, 'waypoints'):
            self.waypoints = self._generate_agv_waypoints()
            self.current_waypoint_indices = [0] * self.config.num_agvs
            
        dt = 1.0  # 1 second intervals
        speed = 0.83  # 3 km/h in m/s
        
        new_positions = []
        for i in range(self.config.num_agvs):
            current_pos = self.agv_positions[i]
            target_waypoint = self.waypoints[i][self.current_waypoint_indices[i]]
            
            # Calculate direction vector to next waypoint
            direction = np.array(target_waypoint) - current_pos[:2]
            distance = np.linalg.norm(direction)
            
            if distance < speed * dt:  # Reached waypoint
                self.current_waypoint_indices[i] = (self.current_waypoint_indices[i] + 1) % len(self.waypoints[i])
                new_pos = np.array([*target_waypoint, 0.5])  # 0.5m height
            else:
                # Move towards waypoint
                direction = direction / distance
                new_pos = current_pos + np.array([*direction * speed * dt, 0.0])
            
            new_positions.append(new_pos)
        
        self.agv_positions = tf.constant(new_positions, dtype=tf.float32)
        
        # Update positions history
        for i in range(self.config.num_agvs):
            self.positions_history[i].append(self.agv_positions[i].numpy())
        
        return self.agv_positions

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
        """Generate channel matrices with proper RIS modeling"""
        # Update AGV positions
        current_positions = self._update_agv_positions(self.config.num_time_steps)
        
        # Add batch dimension to positions
        current_positions = tf.expand_dims(current_positions, axis=0)
        bs_position = tf.constant([[[10.0, 0.5, 4.5]]], dtype=tf.float32)
        
        # Set topology
        self.channel_model.set_topology(
            ut_loc=current_positions,
            bs_loc=bs_position,
            ut_orientations=tf.zeros([1, self.config.num_agvs, 3]),
            bs_orientations=tf.zeros([1, 1, 3]),
            ut_velocities=tf.zeros([1, self.config.num_agvs, 3]),  # Zero velocity for waypoint movement
            in_state=tf.zeros([1, self.config.num_agvs], dtype=tf.bool)
        )
        
        # Generate paths with RIS
        paths_with_ris = self.scene.compute_paths(max_depth=3)
        
        # Generate paths without RIS by temporarily removing it
        ris = self.scene.get("ris")
        self.scene.remove("ris")
        paths_without_ris = self.scene.compute_paths(max_depth=1)
        self.scene.add(ris)
        
        # Generate channel matrices
        h_with_ris = self.channel_model(
            num_time_samples=self.config.num_time_steps,
            sampling_frequency=self.config.sampling_frequency,
            paths=paths_with_ris
        )
        
        h_without_ris = self.channel_model(
            num_time_samples=self.config.num_time_steps,
            sampling_frequency=self.config.sampling_frequency,
            paths=paths_without_ris
        )
        
        return {
            'h': h_with_ris[0],
            'tau': paths_with_ris.tau,
            'paths': paths_with_ris,
            'los_condition': tf.cast(
                tf.reduce_any(tf.not_equal(paths_with_ris.tau, float('inf')), axis=-1),
                tf.bool
            ),
            'agv_positions': current_positions,
            'h_with_ris': h_with_ris[0],
            'h_without_ris': h_without_ris[0]  # Using proper ray-traced channel
        }