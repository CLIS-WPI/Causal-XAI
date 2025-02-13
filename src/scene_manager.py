import tensorflow as tf
from sionna.rt import Scene, Transmitter, Receiver, RIS, RadioMaterial, PlanarArray
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

    def _add_materials(self):
        """Add materials to the scene"""
        # Add concrete material if it doesn't exist
        if 'concrete' not in self._scene.radio_materials:
            concrete = RadioMaterial(
                name="concrete",
                relative_permittivity=self.config.materials['concrete']['relative_permittivity'],
                conductivity=self.config.materials['concrete']['conductivity']
            )
            self._scene.add(concrete)
        
        # Add metal material if it doesn't exist
        if 'metal' not in self._scene.radio_materials:
            metal = RadioMaterial(
                name="metal",
                relative_permittivity=self.config.materials['metal']['relative_permittivity'],
                conductivity=self.config.materials['metal']['conductivity']
            )
            self._scene.add(metal)
    
    def _add_room_boundaries(self):
        """Add walls, floor and ceiling"""
        length, width, height = self.config.room_dim
        boundaries = [
            ("floor", [length/2, width/2, 0]),
            ("ceiling", [length/2, width/2, height]),
            ("wall_front", [length/2, 0, height/2]),
            ("wall_back", [length/2, width, height/2]),
            ("wall_left", [0, width/2, height/2]),
            ("wall_right", [length, width/2, height/2])
        ]

        for name, position in boundaries:
            # Check if boundary already exists
            if name not in self._scene.transmitters:
                boundary = Transmitter(
                    name=name,
                    position=tf.constant(position, dtype=tf.float32),
                    orientation=tf.constant([0, 0, 0], dtype=tf.float32)
                )
                boundary.scene = self._scene
                boundary.radio_material = self._scene.radio_materials["concrete"]
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
            # Create shelf vertices (8 corners of a box)
            width, depth, height = dimensions
            x, y, z = shelf_positions[i]
            
            vertices = [
                [x, y, z],               # bottom front left
                [x + width, y, z],       # bottom front right 
                [x + width, y + depth, z], # bottom back right
                [x, y + depth, z],       # bottom back left
                [x, y, z + height],      # top front left
                [x + width, y, z + height], # top front right
                [x + width, y + depth, z + height], # top back right
                [x, y + depth, z + height]  # top back left
            ]

            # Define faces using vertex indices
            faces = [
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # front
                [2, 3, 7, 6],  # back
                [0, 3, 7, 4],  # left
                [1, 2, 6, 5]   # right
            ]

            # Create shelf mesh
            shelf_mesh = {
                'vertices': vertices,
                'faces': faces
            }

            # Add shelf to scene using scene.add() method
            self._scene.add(
                shelf_mesh,
                radio_material=self._scene.radio_materials["metal"],
                name=f"shelf_{i}"
            )
            
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