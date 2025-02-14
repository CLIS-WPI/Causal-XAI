import tensorflow as tf
from sionna.rt import Scene, SceneObject , Transmitter, Receiver, RIS, RadioMaterial, PlanarArray
import logging
from config import SmartFactoryConfig
from sionna.rt import CellGrid, DiscretePhaseProfile 
from sionna.rt import Scene, RadioMaterial, PlanarArray
logger = logging.getLogger(__name__)

class SceneManager:
    def __init__(self, scene: Scene, config: SmartFactoryConfig):
        self._scene = scene
        self.config = config
        
        # Initialize scene frequency
        self._scene.frequency = tf.cast(config.carrier_frequency, scene.dtype.real_dtype)
        
        # Add room boundaries and materials
        self._add_materials()
        self._add_room_boundaries()
        self._add_metal_shelves()

        # Check visibility after setup
        self.visibility_results = self._check_visibility()

        # Log overall visibility status
        num_visible = sum(1 for result in self.visibility_results.values() if result['los_available'])
        logger.info(f"LOS paths available to {num_visible}/{len(self.visibility_results)} receivers")
        
    def _add_materials(self):
        """Add materials to the scene with complete properties"""
        # Add concrete material
        if 'concrete' not in self._scene.radio_materials:
            concrete = RadioMaterial(
                name="concrete",
                relative_permittivity=self.config.materials['concrete']['relative_permittivity'],
                conductivity=self.config.materials['concrete']['conductivity'],
                scattering_coefficient=self.config.materials['concrete']['scattering_coefficient'],
                xpd_coefficient=self.config.materials['concrete']['xpd_coefficient']
            )
            concrete.roughness = self.config.materials['concrete']['roughness']
            self._scene.add(concrete)
        
        # Add metal material
        if 'metal' not in self._scene.radio_materials:
            metal = RadioMaterial(
                name="metal",
                relative_permittivity=self.config.materials['metal']['relative_permittivity'],
                conductivity=self.config.materials['metal']['conductivity'],
                scattering_coefficient=self.config.materials['metal']['scattering_coefficient'],
                xpd_coefficient=self.config.materials['metal']['xpd_coefficient']
            )
            metal.roughness = self.config.materials['metal']['roughness']
            self._scene.add(metal)
    
    # In SceneManager._add_room_boundaries:
    def _add_room_boundaries(self):
        """Add walls, floor and ceiling"""
        length, width, height = self.config.room_dim
        
        # Define boundaries with explicit sizes and orientations
        boundaries = [
            # name, position, orientation, size, normal
            ("floor", [length/2, width/2, 0], [0, 0, 0], [length, width], [0, 0, 1]),
            ("ceiling", [length/2, width/2, height], [180, 0, 0], [length, width], [0, 0, -1]),
            ("wall_front", [length/2, 0, height/2], [90, 0, 0], [length, height], [0, 1, 0]),
            ("wall_back", [length/2, width, height/2], [-90, 0, 0], [length, height], [0, -1, 0]),
            ("wall_left", [0, width/2, height/2], [0, 90, 0], [width, height], [1, 0, 0]),
            ("wall_right", [length, width/2, height/2], [0, -90, 0], [width, height], [-1, 0, 0])
        ]

        for name, position, orientation, size, normal in boundaries:
            if name not in self._scene.transmitters:
                boundary = Transmitter(
                    name=name,
                    position=tf.constant(position, dtype=tf.float32),
                    orientation=tf.constant(orientation, dtype=tf.float32)
                )
                boundary.scene = self._scene
                boundary.size = size
                boundary.normal = tf.constant(normal, dtype=tf.float32)
                boundary.radio_material = self._scene.radio_materials["concrete"]
                boundary.scattering_coefficient = self.config.materials['concrete']['scattering_coefficient']
                self._scene.add(boundary)

    def _add_metal_shelves(self):
        """Add metal shelves as physical objects with proper material properties"""
        shelf_positions = self.config.scene_objects['shelf_positions']
        dimensions = self.config.scene_objects['shelf_dimensions']
        num_shelves = self.config.scene_objects['num_shelves']

        # Create metal material if not already present
        if "metal" not in self._scene.radio_materials:
            metal = RadioMaterial(
                relative_permittivity=1.0,
                conductivity=1e7,
                name="metal"
            )
            self._scene.add(metal)

        for i in range(num_shelves):
            width, depth, height = dimensions
            x, y, z = shelf_positions[i]
            
            # Create six transmitters for each face of the shelf
            # Bottom face
            bottom = Transmitter(name=f"shelf_{i}_bottom",
                            position=tf.constant([x + width/2, y + depth/2, z]),
                            orientation=tf.constant([0., 0., -1.]))
            bottom.size = [width, depth]
            
            # Top face
            top = Transmitter(name=f"shelf_{i}_top",
                            position=tf.constant([x + width/2, y + depth/2, z + height]),
                            orientation=tf.constant([0., 0., 1.]))
            top.size = [width, depth]
            
            # Front face
            front = Transmitter(name=f"shelf_{i}_front",
                            position=tf.constant([x + width/2, y, z + height/2]),
                            orientation=tf.constant([0., -1., 0.]))
            front.size = [width, height]
            
            # Back face
            back = Transmitter(name=f"shelf_{i}_back",
                            position=tf.constant([x + width/2, y + depth, z + height/2]),
                            orientation=tf.constant([0., 1., 0.]))
            back.size = [width, height]
            
            # Left face
            left = Transmitter(name=f"shelf_{i}_left",
                            position=tf.constant([x, y + depth/2, z + height/2]),
                            orientation=tf.constant([-1., 0., 0.]))
            left.size = [depth, height]
            
            # Right face
            right = Transmitter(name=f"shelf_{i}_right",
                            position=tf.constant([x + width, y + depth/2, z + height/2]),
                            orientation=tf.constant([1., 0., 0.]))
            right.size = [depth, height]
            
            # Add all faces to the scene
            faces = [bottom, top, front, back, left, right]
            for face in faces:
                face.radio_material = self._scene.radio_materials["metal"]
                self._scene.add(face)
            
            logger.debug(f"Added shelf_{i} at position {shelf_positions[i]} with dimensions: {dimensions}")
    
    def add_transmitter(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Transmitter:
        """Add base station"""
        tx = Transmitter(name=name, position=position, orientation=orientation)
        tx.scene = self._scene
        
        tx_array = PlanarArray(
            num_rows=self.config.bs_array[0],
            num_cols=self.config.bs_array[1],
            vertical_spacing=self.config.bs_array_spacing,
            horizontal_spacing=self.config.bs_array_spacing,
            pattern=self.config.bs_array_pattern,
            polarization=self.config.bs_polarization
        )
        tx.array = tx_array
        
        self._scene.add(tx)
        return tx

    def add_ris(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> RIS:
        """Add RIS"""
        ris = RIS(
            name=name,
            position=position,
            orientation=orientation,
            num_rows=self.config.ris_elements[0],
            num_cols=self.config.ris_elements[1],
            dtype=self._scene.dtype
        )
        
        # Set up RIS phase profile
        cell_grid = CellGrid(
            num_rows=self.config.ris_elements[0],
            num_cols=self.config.ris_elements[1],
            dtype=self._scene.dtype
        )
        
        # Initialize with zeros for phase values
        phase_values = tf.zeros([1, self.config.ris_elements[0], 
                            self.config.ris_elements[1]], 
                            dtype=tf.float32)
        
        # Create phase profile
        phase_profile = DiscretePhaseProfile(
            cell_grid=cell_grid,
            values=phase_values,
            dtype=self._scene.dtype
        )
        
        # Set phase profile
        ris.phase_profile = phase_profile
        
        # Set scene and add to scene
        ris.scene = self._scene
        self._scene.add(ris)
        return ris
    
    def add_receiver(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Receiver:
        """Add AGV receiver"""
        rx = Receiver(name=name, position=position, orientation=orientation)
        rx.scene = self._scene
        
        rx_array = PlanarArray(
            num_rows=self.config.agv_array[0],
            num_cols=self.config.agv_array[1],
            vertical_spacing=self.config.agv_array_spacing,
            horizontal_spacing=self.config.agv_array_spacing,
            pattern=self.config.agv_array_pattern,
            polarization=self.config.agv_polarization
        )
        rx.array = rx_array
        
        self._scene.add(rx)
        return rx
    
    def _check_visibility(self):
        """Check line-of-sight visibility between BS and AGVs"""
        logger = logging.getLogger(__name__)
        
        # Get BS position as reference
        bs_pos = tf.constant(self.config.bs_position, dtype=tf.float32)
        
        visibility_results = {}
        for rx_name, rx in self._scene.receivers.values():
            # Calculate direct path vector and distance
            rx_pos = rx.position
            direct_path_vector = rx_pos - bs_pos
            direct_path_distance = tf.norm(direct_path_vector)
            direction = direct_path_vector / direct_path_distance
            
            # Initialize visibility check
            los_blocked = False
            blocking_objects = []
            
            # Check each shelf for intersection
            for obj in self._scene.transmitters.values():
                if 'shelf' in obj.name:
                    # Get shelf dimensions and position
                    shelf_pos = obj.position
                    shelf_dims = tf.constant([
                        obj.size[0],  # width
                        obj.size[1],  # depth
                        self.config.scene_objects['shelf_dimensions'][2]  # height
                    ], dtype=tf.float32)
                    
                    # Calculate intersection with shelf
                    intersects = self._check_ray_box_intersection(
                        bs_pos, direction, direct_path_distance,
                        shelf_pos, shelf_dims
                    )
                    
                    if intersects:
                        los_blocked = True
                        blocking_objects.append(obj.name)
            
            # Store results
            visibility_results[rx_name] = {
                'distance': direct_path_distance.numpy(),
                'los_available': not los_blocked,
                'blocking_objects': blocking_objects,
                'elevation_angle': tf.math.asin(-direction[2]).numpy() * 180/np.pi
            }
            
            # Log visibility status
            status = "BLOCKED" if los_blocked else "CLEAR"
            logger.info(f"LOS path to {rx_name} is {status} at distance {direct_path_distance:.2f}m")
            if blocking_objects:
                logger.info(f"Blocking objects for {rx_name}: {', '.join(blocking_objects)}")
                
            # Calculate expected path loss
            free_space_pl = 20 * tf.math.log(direct_path_distance * self.config.carrier_frequency / SPEED_OF_LIGHT) / tf.math.log(10.0)
            logger.debug(f"Expected free space path loss to {rx_name}: {free_space_pl:.1f} dB")
        
        return visibility_results

    def _check_ray_box_intersection(self, origin, direction, max_distance, box_pos, box_dims):
        """
        Check if a ray intersects with a box
        
        Args:
            origin: Ray origin point [x, y, z]
            direction: Normalized ray direction vector
            max_distance: Maximum distance to check
            box_pos: Box center position [x, y, z]
            box_dims: Box dimensions [width, depth, height]
        
        Returns:
            bool: True if ray intersects box before max_distance
        """
        # Calculate box bounds
        box_min = box_pos - box_dims/2
        box_max = box_pos + box_dims/2
        
        # Calculate intersection with each axis
        t1 = (box_min - origin) / (direction + 1e-10)  # Add small epsilon to avoid division by zero
        t2 = (box_max - origin) / (direction + 1e-10)
        
        # Find entrance and exit points
        t_min = tf.reduce_max(tf.minimum(t1, t2))
        t_max = tf.reduce_min(tf.maximum(t1, t2))
        
        # Check if intersection occurs within ray length
        return tf.logical_and(t_max > 0, t_min < max_distance)
