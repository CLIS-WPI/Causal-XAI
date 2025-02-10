import tensorflow as tf
import numpy as np
import sionna
from scene_setup import setup_scene
import shap
from dowhy import CausalModel
import networkx as nx
import pandas as pd
from sionna.constants import SPEED_OF_LIGHT
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.rt import Scene, Transmitter, Receiver, RIS, SceneObject, PlanarArray, RadioMaterial
from sionna.rt import DiscretePhaseProfile, CellGrid
from sionna.rt import CellGrid, DiscretePhaseProfile
from sionna.channel.utils import cir_to_ofdm_channel
from sionna.constants import SPEED_OF_LIGHT
from shap.shap_analyzer import ShapAnalyzer
from shap.shap_utils import preprocess_channel_data


class SmartFactoryChannel:
    """Smart Factory Channel Generator using Sionna
    
    This class handles the generation and analysis of wireless channels
    in a smart factory environment with two mobile AGVs and RIS elements.
    
    Args:
        config: Configuration object containing simulation parameters
        scene: Optional pre-configured Scene object. If None, creates new scene
        
    Attributes:
        config: Simulation configuration
        scene: Sionna ray tracing scene
        positions_history: Historical AGV positions
        agv_positions: Current AGV positions
        bs_array: Base station antenna array
        agv_array: AGV antenna array
        channel_model: Ray tracing channel model
    """
    
    def __init__(self, config, scene=None):
        """
        Initialize the environment with configuration and scene setup.
        
        Args:
            config: Configuration object containing environment parameters
            scene: Optional pre-configured scene object
            
        Raises:
            ValueError: If required config parameters are missing or invalid
            RuntimeError: If initialization of any component fails
        """
        try:
            # Validate required config attributes
            self._validate_config(config)
            
            # Initialize basic attributes
            self.config = config
            sionna.config.xla_compat = True
            tf.random.set_seed(config.seed)

            # Initialize SHAP analyzer
            self.shap_analyzer = ShapAnalyzer(config)

            # Initialize position tracking
            self.positions_history = [[] for _ in range(config.num_agvs)]
            self.agv_positions = self._generate_initial_agv_positions()

            # Initialize scene
            self.scene = scene if scene is not None else setup_scene(config)
            
            # Configure and set antenna arrays
            self._setup_antenna_arrays()
            
            # Perform final validations
            self._validate_scene()

        except ValueError as e:
            raise ValueError(f"Configuration error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Environment initialization failed: {str(e)}") from e

    def _validate_config(self, config):
        """Validate required configuration parameters."""
        required_attrs = ['num_agvs', 'seed', 'bs_array', 'bs_array_spacing', 
                        'bs_array_pattern', 'bs_polarization', 'dtype',
                        'agv_array_spacing']
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ValueError(f"Config must specify {attr}")

    def _setup_antenna_arrays(self):
        """Configure and set up antenna arrays."""
        try:
            # Base station array
            self.bs_array = PlanarArray(
                num_rows=self.config.bs_array[0],
                num_cols=self.config.bs_array[1],
                vertical_spacing=self.config.bs_array_spacing,
                horizontal_spacing=self.config.bs_array_spacing,
                pattern=self.config.bs_array_pattern,
                polarization=self.config.bs_polarization,
                dtype=self.config.dtype
            )
            
            # AGV array
            self.agv_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=self.config.agv_array_spacing,
                horizontal_spacing=self.config.agv_array_spacing,
                pattern="iso",
                polarization="V"
            )
            
            # Get current max object ID
            max_id = -1
            for obj in self.scene.objects.values():
                if hasattr(obj, 'object_id'):
                    max_id = max(max_id, obj.object_id)
            
            # Set arrays in scene with proper object IDs
            self.scene.tx_array = self.bs_array
            
            # Add base station with next available ID
            bs = self.scene.get('bs')
            if bs is not None:
                bs.object_id = max_id + 1
                
            # Add RIS with next available ID after base station
            ris = self.scene.get('ris')
            if ris is not None:
                ris.object_id = max_id + 2
                
            self.scene.rx_array = self.agv_array

        except Exception as e:
            raise RuntimeError(f"Error setting up antenna arrays: {str(e)}")

    def _validate_scene(self):
        """Validate scene configuration and object IDs."""
        if not hasattr(self.scene, 'objects'):
            raise ValueError("Scene is not properly configured - missing objects attribute")
            
        if not self.scene.objects:
            raise ValueError("Scene contains no objects")
            
        # Get total number of objects including BS and RIS
        total_objects = len(self.scene.objects)
        
        # Find maximum object ID
        max_object_id = -1
        object_ids = []
        for obj in self.scene.objects.values():
            if hasattr(obj, 'object_id'):
                max_object_id = max(max_object_id, obj.object_id)
                object_ids.append(obj.object_id)
        
        # Check for duplicate IDs
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Duplicate object IDs detected")
        
        # Verify object IDs are zero-based and continuous
        if max_object_id >= total_objects:
            # Fix object IDs by reassigning them sequentially
            print("[DEBUG] Reassigning object IDs sequentially")
            for i, obj in enumerate(self.scene.objects.values()):
                if hasattr(obj, 'object_id'):
                    obj.object_id = i
                    print(f"[DEBUG] Assigned ID {i} to object {obj.name}")
        
        # Verify required scene components
        required_objects = ['bs', 'ris']  # Add other required object names if needed
        for obj_name in required_objects:
            if not any(obj.name == obj_name for obj in self.scene.objects.values()):
                raise ValueError(f"Scene is missing required object: {obj_name}")
        
        # Validate antenna arrays
        if not hasattr(self.scene, 'tx_array') or not hasattr(self.scene, 'rx_array'):
            raise ValueError("Scene is missing antenna array configuration")
            
    def _verify_object_ids(self):
        """Verify all object IDs are unique and valid"""
        object_ids = []
        for obj in self.scene.objects.values():
            if hasattr(obj, 'object_id'):
                if obj.object_id in object_ids:
                    raise ValueError(f"Duplicate object ID {obj.object_id} found")
                object_ids.append(obj.object_id)
                
            if obj.object_id >= len(self.scene.objects):
                raise ValueError(f"Object ID {obj.object_id} exceeds scene size")
        
    def add(self, item):
        """Add an item to the scene with proper object ID management"""
        try:
            # Get current max object ID
            max_id = -1
            for obj in self.scene.objects.values():
                if hasattr(obj, 'object_id'):
                    max_id = max(max_id, obj.object_id)
            
            # Handle RIS objects specially
            if isinstance(item, RIS):
                # Assign next available ID
                item.object_id = max_id + 1
                
                # Set radio material
                item.radio_material = RadioMaterial("itu_metal", dtype=self.config.dtype)
                
                # Add to scene's RIS dictionary if it exists
                if hasattr(self.scene, '_ris'):
                    self.scene._ris[item.name] = item
                
                # Add to scene objects
                self.scene._scene_objects[item.name] = item
                
                # Set scene reference
                item.scene = self.scene
                
                # Verify object IDs after adding RIS
                self._verify_object_ids()
                
            else:
                # For non-RIS objects, use scene's default add method
                self.scene.add(item)
                
                # Ensure object ID is set properly
                if hasattr(item, 'object_id'):
                    item.object_id = max_id + 1
                    
                # Verify object IDs after adding any object
                self._verify_object_ids()
                        
            return item
                
        except Exception as e:
            raise RuntimeError(f"Error adding item to scene: {str(e)}")
    
    def _validate_configuration(self):
        """Validate critical configuration parameters"""
        required_attrs = [
            'bs_array', 'bs_array_spacing', 
            'agv_array_spacing', 'ray_tracing',
            'sampling_frequency', 'carrier_frequency',
            'num_subcarriers', 'subcarrier_spacing'
        ]
        
        missing_attrs = [attr for attr in required_attrs 
                        if not hasattr(self.config, attr)]
        
        if missing_attrs:
            raise ValueError(
                f"Missing required configuration attributes: {missing_attrs}"
            )
            
        # Validate array configurations
        if len(self.config.bs_array) != 2:
            raise ValueError("bs_array must specify [num_rows, num_cols]")
            
        # Validate frequency parameters
        if self.config.carrier_frequency <= 0:
            raise ValueError("carrier_frequency must be positive")
        if self.config.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")

    def _generate_initial_agv_positions(self):
        """Generate initial AGV positions"""
        return tf.constant([
            [12.0, 5.0, 0.5],   # AGV1
            [8.0, 15.0, 0.5]    # AGV2
        ], dtype=tf.float32)

    def _generate_agv_waypoints(self):
        """Generate predefined waypoints for each AGV"""
        return [
            # AGV1: Loop around shelves
            [[12.0, 5.0], [12.0, 15.0], [8.0, 15.0], [8.0, 5.0]],
            # AGV2: Cross-diagonal
            [[8.0, 15.0], [12.0, 5.0], [15.0, 8.0], [5.0, 12.0]]
        ]

    def _update_agv_positions(self, time_step):
        """Update AGV positions using waypoint-based movement
        
        Args:
            time_step (float): Time step for position update in seconds
            
        Returns:
            tf.Tensor: Updated AGV positions with shape [num_agvs, 3]
        """
        if not hasattr(self, 'waypoints'):
            self.waypoints = self._generate_agv_waypoints()
            self.current_waypoint_indices = [0] * self.config.num_agvs
        
        speed = 0.83  # 3 km/h in m/s
        new_positions = []
        
        # Convert current positions to numpy if it's a tensor
        current_positions = self.agv_positions.numpy() if tf.is_tensor(self.agv_positions) else self.agv_positions
        
        for i in range(self.config.num_agvs):
            current_pos = current_positions[i]
            target_waypoint = np.array(self.waypoints[i][self.current_waypoint_indices[i]] + [0.5])  # Add z-coordinate
            
            # Calculate direction vector to next waypoint
            direction = target_waypoint - current_pos
            distance = np.linalg.norm(direction)
            
            if distance < speed * time_step:  # Reached waypoint
                self.current_waypoint_indices[i] = (self.current_waypoint_indices[i] + 1) % len(self.waypoints[i])
                new_pos = target_waypoint
            else:
                # Move towards waypoint
                normalized_direction = direction / distance
                new_pos = current_pos + normalized_direction * speed * time_step
                
            new_positions.append(new_pos)
        
        # Convert list of positions to numpy array first
        new_positions_array = np.array(new_positions, dtype=np.float32)
        
        # Update positions history with numpy arrays
        for i in range(self.config.num_agvs):
            self.positions_history[i].append(new_positions_array[i].copy())
        
        # Convert to tensorflow tensor with explicit shape
        return tf.constant(new_positions_array, dtype=tf.float32)

    def save_csi_dataset(self, filepath, num_samples=None):
        """
        Save the complete CSI dataset to an HDF5 file.
        
        Args:
            filepath: Path where the HDF5 file will be saved
            num_samples: Number of channel samples to generate and save (optional, uses config.num_time_steps if not specified)
        """
        import h5py
        
        if num_samples is None:
            num_samples = self.config.num_time_steps
        
        with h5py.File(filepath, 'w') as f:
            csi_group = f.create_group('csi_data')
            config_group = f.create_group('config')
            
            channel_data = []
            path_delays = []
            los_conditions = []
            agv_positions = []
            
            for _ in range(num_samples):
                sample = self.generate_channel()
                
                channel_data.append(sample['h'].numpy())
                path_delays.append(sample['tau'].numpy())
                # Convert integer LOS condition to numpy array
                los_conditions.append(np.array(sample['los_condition'], dtype=np.int32))
                agv_positions.append(sample['agv_positions'].numpy())
            
            # Save datasets
            csi_group.create_dataset('channel_matrices', data=np.array(channel_data))
            csi_group.create_dataset('path_delays', data=np.array(path_delays))
            csi_group.create_dataset('los_conditions', data=np.array(los_conditions))
            csi_group.create_dataset('agv_positions', data=np.array(agv_positions))
            
        # Rest of the method remains the same
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
        """Compute SHAP values for channel response analysis"""
        try:
            if not hasattr(self, 'shap_analyzer'):
                self.shap_analyzer = ShapAnalyzer(self.config)
                
            # Process data and get SHAP analysis
            processed_data = preprocess_channel_data(channel_response, self.config)
            return self.shap_analyzer.analyze_channel_response(processed_data)
        except ValueError as e:
            # Return default values for insufficient samples
            return {
                'position_impact': {
                    'description': 'Impact of AGV positions on channel',
                    'values': tf.zeros_like(channel_response['h_with_ris']),
                    'interpretation': 'Insufficient samples for SHAP analysis'
                },
                'ris_impact': {
                    'description': 'Impact of RIS on channel',
                    'values': tf.zeros_like(channel_response['h_with_ris']),
                    'interpretation': 'Insufficient samples for SHAP analysis'
                }
            }

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
            ('agv_x', 'los_condition'),
            ('agv_y', 'los_condition'),
            ('los_condition', 'channel_quality_with_ris'),
            ('agv_x', 'channel_quality_with_ris'),
            ('agv_y', 'channel_quality_with_ris')
        ])
        return graph

    def perform_causal_analysis(self, channel_response):
        """Perform causal analysis using DoWhy"""
        # Prepare data for causal analysis
        data = pd.DataFrame({
            'agv_x': channel_response['agv_positions'][0, :, 0].numpy(),
            'agv_y': channel_response['agv_positions'][0, :, 1].numpy(),
            'los_condition': tf.cast(channel_response['los_condition'], tf.float32).numpy(),
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
        baseline_scans = self.config.beamforming['num_beams']
        
        # Calculate optimized beam training using XAI-guided pruning
        pruning_factor = self._compute_pruning_factor(channel_response)
        optimized_scans = int(baseline_scans * (1 - pruning_factor))
        
        # Calculate energy consumption for different components
        beam_training_energy = {
            'baseline': baseline_scans * self.config.energy['beam_scan_power'],
            'optimized': optimized_scans * self.config.energy['beam_scan_power'],
            'savings': (baseline_scans - optimized_scans) * self.config.energy['beam_scan_power']
        }
        
        # Calculate RIS-related energy metrics
        ris_energy = {
            'configuration_overhead': self.config.energy['ris_config_power'],
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
        try:
            # Use SHAP values to determine which beams are most important
            shap_analysis = channel_response['shap_analysis']
            position_impact = tf.reduce_mean(tf.abs(shap_analysis['position_impact']['values']))
            ris_impact = tf.reduce_mean(tf.abs(shap_analysis['ris_impact']['values']))
            
            # Calculate pruning factor based on feature importance
            total_impact = position_impact + ris_impact
            
            # Handle division by zero and NaN
            if tf.math.is_nan(total_impact) or tf.math.equal(total_impact, 0.0):
                return 0.0  # Default to no pruning if values are invalid
                
            pruning_factor = tf.minimum(
                0.5,  # Maximum 50% pruning
                0.3 * (1 - position_impact / total_impact)  # More pruning when position impact is low
            )
            
            # Ensure result is a valid number
            return 0.0 if tf.math.is_nan(pruning_factor) else float(pruning_factor)
            
        except Exception as e:
            print(f"Warning: Error computing pruning factor: {str(e)}")
            return 0.0  # Default to no pruning on error

    def _cleanup_temporary_ris(self):
        """Remove temporary RIS objects created during channel generation"""
        try:
            ris_names = [name for name in self.scene.objects.keys() 
                        if name.startswith('ris_temp')]
            for name in ris_names:
                self.scene.remove(name)
        except Exception as e:
            print(f"Warning: Error cleaning up temporary RIS: {str(e)}")

    def generate_channel(self):
        """Generate channel matrices with proper RIS modeling, explainability data, causal analysis and energy metrics
        
        Returns:
            dict: Comprehensive channel response including:
                - Channel matrices (with/without RIS)
                - Path information
                - Channel quality metrics
                - Causal analysis
                - Energy metrics
                - Feature importance
                - Performance metrics
        """
        # Update AGV positions and calculate velocities
        current_positions = self._update_agv_positions(self.config.simulation['time_step'])
        
        # Calculate AGV velocities from position history
        agv_velocities = tf.zeros_like(current_positions)  # Initialize with zeros
        if len(self.positions_history[0]) > 1:
            for i in range(self.config.num_agvs):
                prev_pos = tf.constant(self.positions_history[i][-2], dtype=tf.float32)
                curr_pos = tf.constant(self.positions_history[i][-1], dtype=tf.float32)
                agv_velocities = tf.tensor_scatter_nd_update(
                    agv_velocities,
                    [[i]],
                    [(curr_pos - prev_pos) / self.config.simulation['time_step']]  # Access time_step from simulation dict
                )
        
        # Add batch dimension to positions
        current_positions = tf.expand_dims(current_positions, axis=0)
        bs_position = tf.constant([self.config.bs_position], dtype=tf.float32)
        
        # Update AGV positions in the scene
        for i in range(self.config.num_agvs):
            agv = self.scene.get(f"agv_{i}")
            if agv is not None:
                agv.position = current_positions[0, i].numpy()
        
        # Update RIS configuration
        ris = self.scene.get("ris")
        if ris is not None:
            # Create cell grid for RIS
            cell_grid = CellGrid(
                num_rows=8,  # Match RIS dimensions
                num_cols=8,
                dtype=tf.complex64
            )
            
            # Calculate optimal RIS phases for each AGV using the scene configuration
            phase_shifts = []
            for i in range(self.config.num_agvs):
                agv_pos = current_positions[0, i]
                bs_pos = bs_position[0]
                ris_pos = tf.constant(self.config.ris_position, dtype=tf.float32)
                
                # Calculate angles for optimal reflection using scene geometry
                bs_to_ris = ris_pos - bs_pos
                ris_to_agv = agv_pos - ris_pos
                
                # Normalize vectors
                bs_to_ris = bs_to_ris / tf.norm(bs_to_ris)
                ris_to_agv = ris_to_agv / tf.norm(ris_to_agv)
                
                # Calculate optimal phase shift based on path difference
                wavelength = SPEED_OF_LIGHT / self.config.carrier_frequency
                path_difference = tf.norm(bs_to_ris) + tf.norm(ris_to_agv)
                phase = 2 * np.pi * (path_difference / wavelength)
                phase_shifts.append(phase)
            
            # Average phase shifts from all AGVs
            optimal_phase = tf.reduce_mean(phase_shifts)
            
            # Create phase profile with correct shape (1, 8, 8)
            phase_values = optimal_phase * tf.ones([1, 8, 8], dtype=tf.float32)
            
            # Create and set the phase profile
            phase_profile = DiscretePhaseProfile(
                cell_grid=cell_grid,
                values=phase_values,
                dtype=tf.complex64
            )
            
            # Set the phase profile for the RIS
            ris.phase_profile = phase_profile

            try:
                
                # Ensure tensor sizes account for all objects
                array_size = len(self.scene.objects)
                
                # Initialize tensors with correct size
                relative_permittivity = tf.zeros([array_size], dtype=self.config.dtype)
                scattering_coefficient = tf.zeros([array_size], dtype=tf.float32)
                
                # When updating tensors, verify indices
                for obj_id in range(array_size):
                    if obj_id >= array_size:
                        raise ValueError(f"Object ID {obj_id} exceeds array size {array_size}")
                    
                # Update material properties for each object
                for obj in self.scene.objects.values():
                    if hasattr(obj, 'object_id'):
                        if obj.object_id >= array_size:
                            raise ValueError(
                                f"Object ID {obj.object_id} exceeds array size {array_size}"
                            )
                        # Set material properties based on object type
                        if isinstance(obj, RIS):
                            relative_permittivity = tf.tensor_scatter_nd_update(
                                relative_permittivity,
                                [[obj.object_id]],
                                [tf.cast(1.0, self.config.dtype)]
                            )
                            scattering_coefficient = tf.tensor_scatter_nd_update(
                                scattering_coefficient,
                                [[obj.object_id]],
                                [0.1]
                            )
                        # Add other object types as needed
                        
                # Generate paths with RIS using Sionna's ray tracing
                paths_with_ris = self.scene.compute_paths(
                    max_depth=self.config.ray_tracing['max_depth'],
                    method=self.config.ray_tracing['method'],
                    num_samples=self.config.ray_tracing['num_samples'],
                    los=self.config.ray_tracing['los'],
                    reflection=self.config.ray_tracing['reflection'],
                    diffraction=self.config.ray_tracing['diffraction'],
                    scattering=self.config.ray_tracing['scattering']
                )
                
                # Apply Doppler effect and get channel impulse response with RIS
                paths_with_ris.apply_doppler(
                    sampling_frequency=self.config.sampling_frequency,
                    num_time_steps=1
                )
                
                # Get channel impulse response
                a_with_ris, tau_with_ris = paths_with_ris.cir()
                
                # Calculate OFDM subcarrier frequencies
                frequencies = tf.range(
                    start=0,
                    limit=self.config.num_subcarriers,
                    dtype=tf.float32
                ) * self.config.subcarrier_spacing
                
                # Generate OFDM channel with RIS
                h_with_ris = cir_to_ofdm_channel(
                    frequencies=frequencies,
                    a=a_with_ris,
                    tau=tau_with_ris,
                )
                
                # Temporarily remove RIS to compute channel without it
                if ris is not None:
                    # Store RIS configuration before removing
                    stored_ris = self.scene.get("ris")  # Get the RIS object
                    if stored_ris is not None:
                        # Verify it's a valid RIS object before proceeding
                        if not isinstance(stored_ris, RIS):
                            raise ValueError("Invalid RIS object type")
                        
                        # Generate unique temporary name for restored RIS
                        temp_ris_name = f"ris_temp_{id(stored_ris)}"
                        
                        # Store RIS properties before removing
                        stored_ris_config = {
                            'name': temp_ris_name,  # Use temporary name
                            'position': stored_ris.position,
                            'orientation': stored_ris.orientation,
                            'size': stored_ris.size,
                            'phase_profile': stored_ris.phase_profile,
                            'object_id': len(self.scene.objects) - 1  # Assign proper ID
                        }
                        
                        # Remove RIS from scene
                        self.scene.remove("ris")
                        
                        try:
                            # Generate paths without RIS
                            paths_without_ris = self.scene.compute_paths(
                                max_depth=self.config.ray_tracing['max_depth'],
                                method=self.config.ray_tracing['method'],
                                num_samples=self.config.ray_tracing['num_samples'],
                                los=self.config.ray_tracing['los'],
                                reflection=self.config.ray_tracing['reflection'],
                                diffraction=self.config.ray_tracing['diffraction'],
                                scattering=self.config.ray_tracing['scattering']
                            )
                            
                            # Get channel impulse response without RIS
                            a_without_ris, tau_without_ris = paths_without_ris.cir()

                            # Generate OFDM channel without RIS
                            h_without_ris = cir_to_ofdm_channel(
                                frequencies=frequencies,
                                a=a_without_ris,
                                tau=tau_without_ris,
                            )

                            # Create new RIS object with stored configuration and temporary name
                            new_ris = RIS(
                                name=temp_ris_name,  # Use temporary name
                                position=stored_ris_config['position'],
                                orientation=stored_ris_config['orientation'],
                                num_rows=self.config.ris_elements[0],
                                num_cols=self.config.ris_elements[1],
                                dtype=self.config.dtype
                            )
                            new_ris.phase_profile = stored_ris_config['phase_profile']
                            
                            try:
                                # Add the new RIS back to scene
                                self.add(new_ris)
                                self._verify_object_ids()
                            except ValueError as add_error:
                                print(f"Warning: Could not add temporary RIS: {add_error}")
                                # If adding fails, ensure we have fallback values
                                paths_without_ris = paths_with_ris
                                h_without_ris = h_with_ris
                            
                        except Exception as e:
                            # Make sure to restore RIS even if an error occurs
                            if stored_ris is not None:
                                try:
                                    new_ris = RIS(
                                        name=temp_ris_name,  # Use temporary name
                                        position=stored_ris_config['position'],
                                        orientation=stored_ris_config['orientation'],
                                        num_rows=self.config.ris_elements[0],
                                        num_cols=self.config.ris_elements[1],
                                        dtype=self.config.dtype
                                    )
                                    new_ris.phase_profile = stored_ris_config['phase_profile']
                                    self.add(new_ris)
                                    self._verify_object_ids()
                                except Exception as restore_error:
                                    print(f"Error restoring RIS: {restore_error}")
                                    # Ensure we have fallback values
                                    paths_without_ris = paths_with_ris
                                    h_without_ris = h_with_ris
                            raise e
                        finally:
                            # Cleanup any temporary RIS objects
                            try:
                                temp_ris = self.scene.get(temp_ris_name)
                                if temp_ris is not None:
                                    self.scene.remove(temp_ris_name)
                            except Exception as cleanup_error:
                                print(f"Warning: Error during temporary RIS cleanup: {cleanup_error}")
                    else:
                        paths_without_ris = paths_with_ris
                        h_without_ris = h_with_ris
                
                # Calculate channel quality metrics
                channel_quality_with_ris = tf.reduce_mean(tf.abs(h_with_ris))
                channel_quality_without_ris = tf.reduce_mean(tf.abs(h_without_ris))
                
                # In generate_channel(), change the order of updates:
                channel_response = {
                    'h': h_with_ris,
                    'tau': paths_with_ris.tau,
                    'paths': paths_with_ris,
                    'los_condition': paths_with_ris.LOS,
                    'agv_positions': current_positions,
                    'agv_velocities': agv_velocities,
                    'h_with_ris': h_with_ris,
                    'h_without_ris': h_without_ris,
                    'ris_state': phase_values,
                    'channel_quality': {
                        'with_ris': channel_quality_with_ris,
                        'without_ris': channel_quality_without_ris,
                        'improvement': channel_quality_with_ris - channel_quality_without_ris
                    }
                }

                # First add SHAP analysis
                shap_results = self.compute_channel_shap_values(channel_response)
                channel_response['shap_analysis'] = shap_results

                # Then add all other metrics that depend on SHAP analysis
                channel_response.update({
                    'ray_tracing_metrics': {
                        'num_paths': tf.shape(paths_with_ris.tau)[-1],
                        'path_amplitudes': tf.abs(paths_with_ris.a), 
                        'angles_of_arrival': paths_with_ris.theta_t,
                        'angles_of_departure': paths_with_ris.theta_r
                    },
                    'explanation_metadata': self.get_explanation_metadata(),
                    'energy_metrics': self.compute_energy_metrics(channel_response),
                    'causal_analysis': self.perform_causal_analysis(channel_response),
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
                        'ris_impact': tf.reduce_mean(tf.abs(h_with_ris - h_without_ris)),
                        'los_impact': tf.reduce_mean(tf.cast(paths_with_ris.LOS, tf.float32))
                    }
                })

                return channel_response
                
            except Exception as e:
                raise RuntimeError(f"Error generating channel response: {str(e)}") from e