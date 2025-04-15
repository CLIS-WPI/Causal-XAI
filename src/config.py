import tensorflow as tf  # Import TensorFlow for numerical computations and machine learning operations
from sionna.phy.constants import SPEED_OF_LIGHT
  # Import SPEED_OF_LIGHT constant from Sionna for wavelength calculations
import os  # Import os module for file and directory operations

class SmartFactoryConfig:  # Define configuration class for smart factory simulation
    """Configuration class for Smart Factory Channel Simulation using Ray Tracing"""  # Docstring describing the class purpose

    #class variables at the top
    tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation to accelerate TensorFlow computations
    tf.config.optimizer.set_experimental_options({  # Configure experimental optimization options for performance
        'layout_optimizer': True,  # Enable layout optimization to improve computation graph structure
        'constant_folding': True,  # Fold constant expressions to reduce computation overhead
        'shape_optimization': True,  # Optimize tensor shapes for better efficiency
        'remapping': True,  # Remap subgraphs to equivalent, more efficient ones
        'arithmetic_optimization': True,  # Simplify arithmetic operations for faster execution
        'dependency_optimization': True,  # Remove redundant dependencies in the computation graph
        'loop_optimization': True,  # Optimize loop structures for better performance
        'function_optimization': True,  # Optimize function calls within the graph
        'debug_stripper': True,  # Strip debug operations to reduce graph size and improve speed
    })

    def __init__(self):  # Initialize the configuration object with all parameters
        # Simulation parameters optimized for beam switching
        self.num_time_steps = 60000  # Set total time steps to 60,000 for 60 seconds of simulation at 1ms intervals
        self.sampling_frequency = tf.cast(1000, tf.float32)  # Define sampling frequency as 1000 Hz (1 sample per ms)
        self.batch_size = 1  # Set batch size to 1 for single-scenario simulation
        self.dtype = tf.complex64  # Use complex64 data type for signal processing (real + imaginary components)
        self.real_dtype = tf.float32  # Use float32 for real-valued computations
        self.seed = 42  # Set random seed to 42 for reproducible results
        self.bandwidth = 400e6  # Increase from 75MHz to a more realistic 6G bandwidth
        self.num_subcarriers = 256   # Increase from 64 to support wider bandwidth
        self.subcarrier_spacing = 120e3  # Increase from 7.5kHz for 6G systems
        self.scene_type = "indoor"  # Specify the simulation environment as indoor
        self.tx_power = 30  # Set transmit power to 60 dBm for the base station
        # Room dimensions remain the same
        self.room_dim = [20.0, 20.0, 5.0]  # Define room dimensions as 20m x 20m x 5m (length, width, height)
        #self.path_loss_db = 80  # Typical value for indoor factory at 28GHz ## Removed, let Ray Tracing calculate dynamically
        # Frequency configuration for mmWave
        self.carrier_frequency = tf.cast(140e9, tf.float32)  # Set carrier frequency to 140 GHz (THz band for 6G)
        self.wavelength = tf.cast(SPEED_OF_LIGHT/self.carrier_frequency, tf.float32)  # Calculate wavelength based on frequency
        
        # Base station configuration optimized for beam switching
        self.bs_position = [10.0, 10.0, 4.8]  # Position BS at center of ceiling (x=10, y=10, z=4.5)
        self.bs_orientation = [0.0, 0.0, -90.0]  # Orient BS downward (-90° in z-axis, radians)
        self.bs_array = {  # Configure BS antenna array as a dictionary
            'num_rows': 32,  # Set 32 rows for a 32x32 Massive MIMO array
            'num_cols': 32,  # Set 32 columns for a total of 1024 elements
            'vertical_spacing': 0.7,  # Define vertical spacing between elements as 0.7 (in wavelengths)
            'horizontal_spacing': 0.5,  # Define horizontal spacing as 0.5 (in wavelengths)
            'pattern': "tr38901",  # Use TR 38.901 antenna pattern (standard for 5G/6G)
            'polarization': "VH",  # Set polarization to Vertical-Horizontal for dual polarization
            'antenna_gain_db': 25  # BS antenna gain in dB
        }

        self.inf_params = {
                'los_k_factor': 17.0,
                'nlos_sigma': 3.5,
                'path_loss_exp': 1.6,
                'shadow_std': 1.0,
                'penetration_loss': 8.0,
                'reflection_coeff': 0.85
            }
        # Enhanced Material properties for better reflection modeling
        self.materials = {  # Define material properties for ray tracing
            'concrete': {  # Properties for concrete material (walls, floor, ceiling)
                'name': "concrete",  # Name identifier for the material
                'relative_permittivity': 6.8,  # Updated for 140 GHz 
                'conductivity': 2.8,  # Higher for 6G frequencies
                'roughness': 0.1e-3,  # Set surface roughness to 0.1 mm
                'scattering_coefficient': 0.7,  # Higher scattering at 140 GHz
                'xpd_coefficient': 8.0  # Set cross-polarization discrimination coefficient
            },
            'metal': {  # Properties for metal material (shelves)
                'relative_permittivity': 1.0,  # Set relative permittivity for metal
                'conductivity': 1.0e7,  # Define high conductivity for metal (S/m)
                'roughness': 0.2e-3,  # Set surface roughness to 0.2 mm
                'scattering_coefficient': 0.3,  # Define scattering coefficient for metal
                'xpd_coefficient': 15.0  # Set cross-polarization discrimination coefficient
            }
        }

        # AGV Configuration optimized for beam switching demonstration
        self.num_agvs = 2  # Define 2 Autonomous Guided Vehicles (AGVs) in the simulation
        self.agv_height = 0.5  # Set AGV height to 0.5m (used as reference, though explicit in positions)
        self.agv_speed = 1.0  # Set AGV speed to 1 m/s for slower movement and detailed beam switching
        self.agv_dimensions = [1.0, 1.0, 0.5]  # Define AGV size as 1m x 1m x 0.5m
        self.agv_orientations = [  # Set initial orientations for AGVs
            [0.0, 0.0, 0.0],  # AGV 1 orientation (no rotation, facing forward)
            [0.0, 0.0, 0.0]   # AGV 2 orientation (no rotation, facing forward)
        ]
        # Simplified AGV array configuration
        self.agv_array = {  # Configure AGV antenna array
            'num_rows': 4,  # Single row for simple AGV antenna
            'num_cols': 4,  # Single column (1-element antenna)
            'vertical_spacing': 0.5 * self.wavelength,  # Vertical spacing based on wavelength
            'horizontal_spacing': 0.5 * self.wavelength,  # Horizontal spacing based on wavelength
            'pattern': "tr38901",  # Use TR 38.901 pattern for AGV antenna
            'polarization': "VH",  # Set polarization to Vertical-Horizontal
            'antenna_gain_db': 15  # AGV antenna gain in dB
        }

        # Optimized AGV trajectories for beam switching demonstration
        self.agv_trajectories = {
            'agv_1': [[2.0, 3.0, 0.5], [18.0, 3.0, 0.5], [2.0, 3.0, 0.5]],  # Behind Shelf 1 and 2
            'agv_2': [[17.0, 18.0, 0.5], [3.0, 18.0, 0.5], [3.0, 4.0, 0.5], [17.0, 4.0, 0.5], [17.0, 18.0, 0.5]]
        }

        # AGV movement settings optimized for beam switching
        self.agv_movement = {  # Configure AGV movement behavior
            'path_type': 'predefined',  # Use fixed, predefined trajectories
            'update_interval': 0.001,  # Update AGV position every 0.5 seconds
            'obstacle_avoidance': False,  # Enable basic obstacle avoidance (though paths are fixed)
            'min_distance': 1.0,  # Set minimum distance to obstacles as 1 meter
            'safety_margin': 0.5  # Safety margin in meters
           
        }

        # Initial AGV positions
        self.agv_positions = [  # Set starting positions for AGVs explicitly
            [2.0, 3.0, 0.5],  # AGV 1 starts at bottom-left corner, 0.5m height
            [17.0, 18.0, 0.5]  # AGV 2 starts at top-right corner, 0.5m height
        ]

        # static scene configuration
        self.static_scene = {  # Define static environment properties
            'material': 'concrete',  # Set default wall material to concrete
            'wall_thickness': 0.2,  # Define wall thickness as 0.2 meters
            'floor_material': 'concrete',  # Set floor material to concrete
            'ceiling_material': 'concrete',  # Set ceiling material to concrete
            'wall_height': 5.0,  # Set wall height to 5 meters (matches room_dim)
            'reflectivity': 0.65  # Adjusted for 140 GHz
        }
        
        # Add PLY configuration if not already present
        self.ply_config = {  # Configure PLY file generation settings
            'output_dir': 'meshes',  # Set output directory for PLY files
            'verify_files': True,  # Enable verification of generated PLY files
            'material_properties': {  # Define material properties for PLY export
                'concrete': {  # Properties for concrete in PLY
                    'reflectivity': 0.603815,  # Reflectivity value for concrete
                    'roughness': 0.1  # Roughness value for concrete
                },
                'metal': {  # Properties for metal in PLY
                    'reflectivity': 0.087140,  # Reflectivity value for metal
                    'roughness': 0.05  # Roughness value for metal
                }
            }
        }

        # Optimized obstacle configuration for beam switching
        self.scene_objects = {
            'num_shelves': 5,
            'shelf_dimensions': [
                [4.0, 1.0, 4.0],  # O1
                [2.0, 1.0, 4.0],  # O2
                [1.0, 2.0, 5.0],  # O3
                [1.0, 2.0, 5.0],  # O4
                [4.0, 1.0, 5.0]   # O5
            ],
            'shelf_material': 'metal',
            'shelf_positions': [
                [6.0, 6.0, 0.0],  # O1 closer to BS
                [12.0, 6.0, 0.0],  # O2 - closer to BS
                [4.0, 10.0, 0.0],  # O3
                [14.0, 12.0, 0.0],  # O4
                [8.0, 16.0, 0.0]   # O5
            ],
            'shelf_orientation': [0.0, 0.0, 0.0]
        }

        # Enhanced ray tracing configuration for better beam switching
        self.ray_tracing = {  # Configure ray tracing parameters
            'max_depth': 5,  # Set maximum reflection depth to 5 for detailed paths
            'method': "fibonacci",  # Use Fibonacci sampling for ray distribution
            'num_samples': 1000,  # Set 1000 rays for high tracing accuracy
            'diffraction': True,  # Enable diffraction effects
            'scattering': True,  # Enable scattering effects
            'los': True,  # Enable line-of-sight paths
            'reflection': True,  # Enable reflection paths
            'ris': False,  # Disable Reconfigurable Intelligent Surfaces (not used)
            'scene_type': self.scene_type,  # Use scene type (indoor) from earlier
            'scat_keep_prob': 0.7,  # Set scattering keep probability to 0.7
            'edge_diffraction': True  # Enable edge diffraction for realistic modeling
        }

        # Optimized beamforming configuration for switching demonstration
        self.beamforming = {  # Configure beamforming settings
            'num_beams': 1024,  # Increase from 512 for finer resolution at higher frequency
            'beam_width': 1.0,  # Set beam width to 2 degrees for precision
            'max_steering_angle': 60,  # Allow ±60 degrees steering range
            'adaptation_interval': 0.001,  # Update beams every 1ms (matches sampling)
            'min_snr_threshold': 15.0,  # Set minimum SNR threshold to 15 dB
            'blockage_detection': True,  # Enable blockage detection for beam switching
            'beam_switching': {  # Configure beam switching behavior
                'enabled': True,  # Enable beam switching feature
                'switching_threshold': 1.0,  # Increased threshold (from 1.0) for more stable indoor switching
                'hysteresis': 1.0,  # Increased hysteresis (from 0.5) for indoor environments
            },
            'codebook': {  # Configure beamforming codebook
                'type': 'DFT',  # Use Discrete Fourier Transform codebook
                'size': 1024,  #  1024 to match num_beams
                'oversampling': 4,  # Use 2x oversampling for better resolution
                'grid_size_az': 16,  # Number of azimuth beams
                'grid_size_el': 8    # Number of elevation beams
            },
            'obstacle_radius': 1.5,  # Safety margin for obstacle collision in meters
            'min_angular_separation': 10.0,  # Minimum angular separation for reflection paths in degrees
            'good_snr_threshold': 25.0,  # SNR threshold for good channel quality in dB
            'refinement_factor_good': 0.1,  # Refinement factor for good SNR
            'refinement_factor_poor': 0.3,   # Refinement factor for poor SNR
            'steering_penalty': 0.1         # Penalty for large steering angles
        }

        # Simulation parameters
        self.simulation = {  # Define additional simulation settings
            'time_step': 0.001,  # Set time step to 1ms (matches sampling frequency)
            'snr_range': [10, 40],  # Define SNR range from -20 to 40 dB
            'channel_estimation_error': 0.15,  # Set channel estimation error to 10%
            'noise_power': 1.380649e-23 * 293.15 * self.bandwidth, # Update noise power with new bandwidth
            'default_path_loss_db': 80, # for fallback scenario in case of ray tracing issue as a defult source
            'noise_power_db' : -90, #this is just primary value as a source
            'min_expected_snr_db': 20 # the minmum snr to be expeceted in LOS
        }

        # Basic camera configuration for visualization
        self.cameras = {  # Define camera configurations for scene visualization
            'scene-cam-0': {  # Top view camera from 25m height
                'position': [10.0, 10.0, 25.0],  # Set camera position at center, 25m up
                'look_at': [10.0, 10.0, 0.0],  # Point camera at center of floor
                'up': [0.0, 1.0, 0.0],  # Define up direction as y-axis
                'fov': 70.0  # Set field of view to 70 degrees
            },
            'scene-cam-1': {  # Front view camera at 2.5m height
                'position': [-5.0, 10.0, 2.5],  # Set position outside left wall
                'look_at': [10.0, 10.0, 2.5],  # Point at center at same height
                'up': [0.0, 0.0, 1.0],  # Define up direction as z-axis
                'fov': 70.0  # Set field of view to 70 degrees
            },
            'scene-cam-2': {  # Corner view camera from 15m height
                'position': [-5.0, -5.0, 15.0],  # Set position at southwest corner, 15m up
                'look_at': [10.0, 10.0, 0.0],  # Point at center of floor
                'up': [0.0, 0.0, 1.0],  # Define up direction as z-axis
                'fov': 70.0  # Set field of view to 70 degrees
            },
            'scene-cam-3': {  # Side view camera at 2.5m height
                'position': [10.0, -5.0, 2.5],  # Set position outside south wall
                'look_at': [10.0, 10.0, 2.5],  # Point at center at same height
                'up': [0.0, 0.0, 1.0],  # Define up direction as z-axis
                'fov': 70.0  # Set field of view to 70 degrees
            },
            'scene-cam-4': {  # Top-down view camera from 10m height
                'position': [10.0, 10.0, 10.0],  # Set position above center at 10m
                'look_at': [10.0, 10.0, 0.0],  # Point at center of floor
                'up': [0.0, 1.0, 0.0],  # Define up direction as y-axis
                'fov': 70.0  # Set field of view to 70 degrees
            }
        }
    # In config.py, add this method to SmartFactoryConfig class
    def get_obstacle_list(self):  # Define method to convert shelves to obstacle list
        """Convert scene_objects dictionary to list format for collision detection"""  # Docstring for method purpose
        obstacles = []  # Initialize empty list for obstacles
        
        # Convert shelves to obstacle format
        for i in range(self.scene_objects['num_shelves']):  # Loop through all shelves
            obstacle = {  # Create dictionary for each shelf
                'position': self.scene_objects['shelf_positions'][i],  # Set shelf position
                'dimensions': self.scene_objects['shelf_dimensions'][i],  # Set shelf dimensions
                'type': 'shelf',  # Identify object type as shelf
                'material': self.scene_objects['shelf_material']  # Set shelf material
            }
            obstacles.append(obstacle)  # Add shelf to obstacles list
        
        return obstacles  # Return list of obstacles