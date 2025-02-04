import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import tr38901
from sionna.rt import Scene, Transmitter, Receiver, RIS, SceneObject
from sionna.channel.tr38901 import PanelArray, UMi  # Using UMi as alternative to IndoorFactory
from scene_setup import setup_scene
from sionna.channel.tr38901 import CDL
import shap
from dowhy import CausalModel
import networkx as nx
import pandas as pd

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
        
        # In __init__ method, replace the UMi channel model with:
        self.channel_model = CDL(
            model="C",                    # CDL-C model for dense multipath
            delay_spread=100e-9,          # 100ns delay spread for factory environment
            carrier_frequency=28e9,       # 28 GHz carrier frequency
            ut_array=self.agv_array,      # AGV antenna array
            bs_array=self.bs_array,       # Base station antenna array
            direction='downlink',         # Downlink transmission
            dtype=config.dtype,           # Data type from config
            #enable_pathloss=True,         # Enable path loss modeling
            #enable_shadow_fading=True,    # Enable shadow fading
            min_speed=0.0,               # Minimum speed for Doppler (static/slow AGVs)
            max_speed=0.83               # Maximum speed 3 km/h = 0.83 m/s
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

    def get_explanation_metadata(self):
        """
        Returns metadata for explainability analysis using SHAP/LIME and causal relationships.
        
        This method provides structured information about:
        1. Feature importance and relationships for SHAP/LIME analysis
        2. Causal graph structure showing relationships between components
        3. Key factors affecting channel characteristics
        
        Returns:
            dict: Dictionary containing explainability metadata
        """
        return {
            # SHAP/LIME Feature Relationships
            'feature_relationships': {
                'channel_features': {
                    'h_with_ris': {
                        'description': 'Channel matrix with RIS effects',
                        'dependencies': ['agv_positions', 'ris_config', 'los_condition'],
                        'importance_factors': [
                            'RIS reflection coefficients',
                            'AGV-RIS distance',
                            'RIS-BS distance'
                        ]
                    },
                    'h_without_ris': {
                        'description': 'Channel matrix without RIS',
                        'dependencies': ['agv_positions', 'los_condition'],
                        'importance_factors': [
                            'Direct path loss',
                            'Multipath components',
                            'AGV-BS distance'
                        ]
                    }
                },
                'position_features': {
                    'agv_positions': {
                        'description': 'AGV positions affecting channel',
                        'dependencies': ['waypoints', 'velocity'],
                        'importance_factors': [
                            'Distance to obstacles',
                            'Movement patterns',
                            'Height from ground'
                        ]
                    }
                }
            },
            
            # Causal Graph Structure
            'causal_relationships': {
                'nodes': [
                    'AGV Position',
                    'RIS Configuration',
                    'Channel State',
                    'Path Loss',
                    'LOS Condition'
                ],
                'edges': [
                    {
                        'from': 'AGV Position',
                        'to': 'Channel State',
                        'type': 'direct',
                        'description': 'AGV position directly affects channel characteristics'
                    },
                    {
                        'from': 'RIS Configuration',
                        'to': 'Channel State',
                        'type': 'direct',
                        'description': 'RIS configuration modifies channel properties'
                    },
                    {
                        'from': 'AGV Position',
                        'to': 'LOS Condition',
                        'type': 'direct',
                        'description': 'AGV position determines LOS availability'
                    }
                ]
            },
            
            # Key Performance Indicators
            'kpis': {
                'channel_capacity': {
                    'description': 'Channel capacity affected by RIS',
                    'contributing_factors': [
                        'SNR improvement from RIS',
                        'Multipath diversity',
                        'Interference reduction'
                    ]
                },
                'path_loss': {
                    'description': 'Path loss variations',
                    'contributing_factors': [
                        'Distance-dependent losses',
                        'Shadowing effects',
                        'RIS-aided path optimization'
                    ]
                }
            }
        }


    def compute_channel_shap_values(self, channel_response):
        """
        Compute SHAP values for channel response analysis using DeepExplainer
        """
        # Prepare background data (use historical channel responses)
        background_data = self._get_background_data()  # You need to implement this
        
        # Create explainer
        explainer = shap.DeepExplainer(
            model=self._get_channel_model(),  # You need to implement this
            data=background_data
        )
        
        # Compute SHAP values
        channel_data = tf.stack([
            channel_response['h_with_ris'],
            channel_response['h_without_ris'],
            tf.cast(channel_response['los_condition'], tf.float32)
        ], axis=-1)
        
        shap_values = explainer.shap_values(channel_data)
        
        return {
            'position_impact': {
                'description': 'Impact of AGV positions on channel',
                'values': shap_values[0],  # SHAP values for position impact
                'interpretation': 'Shows how AGV positions affect channel quality'
            },
            'ris_impact': {
                'description': 'Impact of RIS on channel',
                'values': shap_values[1],  # SHAP values for RIS impact
                'interpretation': 'Quantifies RIS contribution to channel improvement'
            }
        }

    def _get_background_data(self):
        """Helper method to get background data for SHAP analysis
        
        Returns:
            tf.Tensor: Background data for SHAP analysis from historical channel responses
        """
        # Use position history to create background data
        num_samples = min(100, len(self.positions_history[0]))  # Use up to 100 historical samples
        
        background_channels = []
        for i in range(num_samples):
            # Get historical positions for all AGVs
            historical_positions = tf.constant(
                [[self.positions_history[j][-(i+1)] for j in range(self.config.num_agvs)]],
                dtype=tf.float32
            )
            
            # Generate channel response for historical positions
            bs_position = tf.constant([[[10.0, 0.5, 4.5]]], dtype=tf.float32)
            
            # Set topology for historical position
            self.channel_model.set_topology(
                ut_loc=historical_positions,
                bs_loc=bs_position,
                ut_orientations=tf.zeros([1, self.config.num_agvs, 3]),
                bs_orientations=tf.zeros([1, 1, 3]),
                ut_velocities=tf.zeros([1, self.config.num_agvs, 3]),
                in_state=tf.zeros([1, self.config.num_agvs], dtype=tf.bool)
            )
            
            # Compute paths and channel response
            paths = self.scene.compute_paths(max_depth=3)
            h = self.channel_model(
                num_time_samples=1,
                sampling_frequency=self.config.sampling_frequency,
                paths=paths
            )
            
            # Stack channel features
            channel_features = tf.stack([
                h[0],
                tf.cast(tf.reduce_any(tf.not_equal(paths.tau, float('inf')), axis=-1), tf.float32),
                tf.reshape(historical_positions, [-1])
            ], axis=-1)
            
            background_channels.append(channel_features)
        
        return tf.stack(background_channels, axis=0)

    def _get_channel_model(self):
        """Helper method to get the channel model for SHAP analysis
        
        Returns:
            callable: Simplified channel model function for SHAP analysis
        """
        def simplified_channel_model(inputs):
            """Simplified channel model for SHAP analysis
            
            Args:
                inputs: Channel features [channel_response, los_condition, positions]
                
            Returns:
                tf.Tensor: Channel quality metric
            """
            channel_response = inputs[..., 0]
            los_condition = inputs[..., 1]
            positions = tf.reshape(inputs[..., 2:], [-1, self.config.num_agvs, 3])
            
            # Compute simplified channel quality metric
            channel_quality = tf.abs(channel_response) * tf.cast(los_condition, tf.float32)
            
            # Add position-dependent effects
            distance_effect = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(positions), axis=-1)),
                axis=-1
            )
            
            return channel_quality * tf.exp(-0.1 * distance_effect)
        
        return simplified_channel_model

    def generate_causal_analysis(self, channel_data):
        """
        Generate causal analysis of channel characteristics.
        
        Args:
            channel_data: Dictionary containing channel measurement data
            
        Returns:
            dict: Causal analysis results
        """
        return {
            'direct_effects': {
                'ris_to_channel': {
                    'effect_size': None,  # Add actual effect size calculation
                    'confidence': None,  # Add confidence measure
                    'description': 'Direct causal effect of RIS on channel quality'
                },
                'position_to_channel': {
                    'effect_size': None,  # Add actual effect size calculation
                    'confidence': None,  # Add confidence measure
                    'description': 'Direct causal effect of AGV position on channel'
                }
            },
            'indirect_effects': {
                'position_via_los': {
                    'effect_size': None,  # Add actual effect size calculation
                    'path': ['AGV Position', 'LOS Condition', 'Channel Quality'],
                    'description': 'Indirect effect through LOS condition'
                }
            }
        }

    def setup_causal_graph(self):
        """Create causal graph for AGV→LOS→BeamChoice relationships"""
        graph = nx.DiGraph([
            ('AGV_Position', 'LOS_Condition'),
            ('LOS_Condition', 'Channel_Quality'),
            ('RIS_Config', 'Channel_Quality'),
            ('AGV_Position', 'Channel_Quality')
        ])
        return graph

    def perform_causal_analysis(self, channel_response):
        """Perform causal analysis using DoWhy"""
        # Prepare data for causal analysis
        data = pd.DataFrame({
            'agv_x': channel_response['agv_positions'][0, :, 0].numpy(),
            'agv_y': channel_response['agv_positions'][0, :, 1].numpy(),
            'los_condition': channel_response['los_condition'].numpy(),
            'channel_quality_with_ris': channel_response['channel_quality']['with_ris'].numpy(),
            'channel_quality_without_ris': channel_response['channel_quality']['without_ris'].numpy()
        })
        
        # Create causal model
        model = CausalModel(
            data=data,
            graph=self.setup_causal_graph(),
            treatment=['agv_x', 'agv_y'],
            outcome='channel_quality_with_ris'
        )
        
        # Identify causal effect
        identified_estimand = model.identify_effect()
        
        # Estimate effect
        estimate = model.estimate_effect(identified_estimand,
                                    method_name="backdoor.linear_regression")
        
        return {
            'causal_effect': estimate.value,
            'confidence_intervals': estimate.get_confidence_intervals(),
            'treatment_variables': ['agv_x', 'agv_y'],
            'outcome_variable': 'channel_quality_with_ris'
        }
    
        #Causal analysis using DoWhy to understand relationships between AGV positions, LOS conditions, and channel quality
        #Energy efficiency metrics including beam training overhead and RIS configuration energy
        #XAI-guided beam pruning for energy optimization
        #Comprehensive metrics in the channel response

    def compute_energy_metrics(self, channel_response):
        """Compute energy efficiency metrics"""
        # Calculate baseline beam training overhead
        baseline_scans = self.config.num_beams
        
        # Calculate optimized beam training using XAI-guided pruning
        pruning_factor = self._compute_pruning_factor(channel_response)
        optimized_scans = int(baseline_scans * (1 - pruning_factor))
        
        # Calculate energy consumption for different components
        beam_training_energy = {
            'baseline': baseline_scans * self.config.energy_per_beam_scan,
            'optimized': optimized_scans * self.config.energy_per_beam_scan,
            'savings': (baseline_scans - optimized_scans) * self.config.energy_per_beam_scan
        }
        
        # Calculate RIS-related energy metrics
        ris_energy = {
            'configuration_overhead': self.config.ris_config_energy,
            'improvement_factor': channel_response['channel_quality']['improvement']
        }
        
        return {
            'beam_training': beam_training_energy,
            'ris_overhead': ris_energy,
            'total_energy_savings': beam_training_energy['savings'] - ris_energy['configuration_overhead'],
            'energy_efficiency': channel_response['channel_quality']['improvement'] / 
                            (beam_training_energy['optimized'] + ris_energy['configuration_overhead'])
        }

    def _compute_pruning_factor(self, channel_response):
        """Compute beam pruning factor based on XAI analysis"""
        # Use SHAP values to determine which beams are most important
        shap_analysis = channel_response['shap_analysis']
        position_impact = tf.reduce_mean(tf.abs(shap_analysis['position_impact']['values']))
        ris_impact = tf.reduce_mean(tf.abs(shap_analysis['ris_impact']['values']))
        
        # Calculate pruning factor based on feature importance
        total_impact = position_impact + ris_impact
        pruning_factor = tf.minimum(
            0.5,  # Maximum 50% pruning
            0.3 * (1 - position_impact / total_impact)  # More pruning when position impact is low
        )
        
        return pruning_factor.numpy()

    def generate_channel(self):
        """Generate channel matrices with proper RIS modeling and explainability data and ausal analysis and energy metrics"""
        # Update AGV positions
        current_positions = self._update_agv_positions(self.config.num_time_steps)
        
        # Add batch dimension to positions
        current_positions = tf.expand_dims(current_positions, axis=0)
        bs_position = tf.constant([[[10.0, 0.5, 4.5]]], dtype=tf.float32)
        
        # Calculate AGV velocities based on position changes
        agv_velocities = tf.zeros([1, self.config.num_agvs, 3], dtype=tf.float32)
        if len(self.positions_history[0]) > 1:
            prev_positions = tf.constant(
                [[self.positions_history[j][-2] for j in range(self.config.num_agvs)]],
                dtype=tf.float32
            )
            agv_velocities = (current_positions - prev_positions) / self.config.time_step
        
        # Set topology with updated velocities
        self.channel_model.set_topology(
            ut_loc=current_positions,
            bs_loc=bs_position,
            ut_orientations=tf.zeros([1, self.config.num_agvs, 3]),
            bs_orientations=tf.zeros([1, 1, 3]),
            ut_velocities=agv_velocities,
            in_state=tf.zeros([1, self.config.num_agvs], dtype=tf.bool)
        )
        
        # Update RIS configuration
        ris = self.scene.get("ris")
        if ris is not None:
            # Calculate optimal RIS phases for each AGV
            phase_shifts = []
            for i in range(self.config.num_agvs):
                agv_pos = current_positions[0, i]
                bs_pos = bs_position[0, 0]
                ris_pos = ris.position
                
                # Calculate angles for optimal reflection
                bs_to_ris = ris_pos - bs_pos
                ris_to_agv = agv_pos - ris_pos
                
                # Normalize vectors
                bs_to_ris = bs_to_ris / tf.norm(bs_to_ris)
                ris_to_agv = ris_to_agv / tf.norm(ris_to_agv)
                
                # Calculate phase shift for this AGV
                phase = tf.math.angle(
                    tf.complex(bs_to_ris[0] * ris_to_agv[0], bs_to_ris[1] * ris_to_agv[1])
                )
                phase_shifts.append(phase)
            
            # Set RIS phase configuration as average of individual optimal phases
            ris.phase_shifts = tf.reduce_mean(phase_shifts)
        
        # Generate paths with RIS
        paths_with_ris = self.scene.compute_paths(
            max_depth=3,
            diffraction=True,
            scattering=True
        )
        
        # Generate paths without RIS
        if ris is not None:
            self.scene.remove("ris")
        paths_without_ris = self.scene.compute_paths(
            max_depth=1,
            diffraction=True,
            scattering=True
        )
        if ris is not None:
            self.scene.add(ris)
        
        # Generate channel matrices with proper sampling
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
        
        # Calculate channel quality metrics
        channel_quality_with_ris = tf.reduce_mean(tf.abs(h_with_ris[0]))
        channel_quality_without_ris = tf.reduce_mean(tf.abs(h_without_ris[0]))
        
        # Create comprehensive channel response dictionary
        channel_response = {
            'h': h_with_ris[0],
            'tau': paths_with_ris.tau,
            'paths': paths_with_ris,
            'los_condition': tf.cast(
                tf.reduce_any(tf.not_equal(paths_with_ris.tau, float('inf')), axis=-1),
                tf.bool
            ),
            'agv_positions': current_positions,
            'agv_velocities': agv_velocities,
            'h_with_ris': h_with_ris[0],
            'h_without_ris': h_without_ris[0],
            'channel_quality': {
                'with_ris': channel_quality_with_ris,
                'without_ris': channel_quality_without_ris,
                'improvement': channel_quality_with_ris - channel_quality_without_ris
            }
        }
        
        # Add causal analysis
        causal_analysis = self.perform_causal_analysis(channel_response)
        
        # Add energy metrics
        energy_metrics = self.compute_energy_metrics(channel_response)
        
        # Update channel response with new metrics
        channel_response.update({
            'causal_analysis': causal_analysis,
            'energy_metrics': energy_metrics,
            'explanation_metadata': self.get_explanation_metadata(),
            'shap_analysis': self.compute_channel_shap_values(channel_response),
            'performance_metrics': {
                'channel_capacity': {
                    'with_ris': channel_quality_with_ris,
                    'without_ris': channel_quality_without_ris,
                    'improvement': channel_quality_with_ris - channel_quality_without_ris
                },
                'path_loss': {
                    'with_ris': tf.reduce_mean(paths_with_ris.tau),
                    'without_ris': tf.reduce_mean(paths_without_ris.tau)
                }
            },
            'feature_importance': {
                'position_impact': tf.reduce_mean(tf.abs(current_positions)),
                'velocity_impact': tf.reduce_mean(tf.abs(agv_velocities)),
                'ris_impact': tf.reduce_mean(tf.abs(h_with_ris[0] - h_without_ris[0])),
                'los_impact': tf.reduce_mean(tf.cast(channel_response['los_condition'], tf.float32))
            }
        })
        
        return channel_response