import tensorflow as tf
from sionna.rt import Scene, Transmitter, Receiver, RIS, RadioMaterial, PlanarArray
import logging
from typing import Any 
from sionna.rt import CellGrid, DiscretePhaseProfile 
logger = logging.getLogger(__name__)

class SceneManager:
    """Simplified scene manager that relies on Sionna's built-in management"""
    
    def __init__(self, scene: Scene, config: Any):
        self.scene = scene
        self.config = config
        
        # Initialize scene frequency
        self.scene.frequency = tf.cast(config.carrier_frequency, scene.dtype.real_dtype)
        
        # Add room boundaries
        self._add_room_boundaries()
        
    def _add_room_boundaries(self):
        """Add walls, floor and ceiling using Sionna's built-in management"""
        # Create concrete material if not exists
        if "concrete" not in self.scene.radio_materials:
            concrete = RadioMaterial(
                name="concrete",
                relative_permittivity=4.5,
                conductivity=0.01,
                dtype=self.scene.dtype
            )
            self.scene.add(concrete)
        
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
            boundary = Transmitter(
                name=name,
                position=tf.constant(position, dtype=tf.float32),
                orientation=tf.constant([0, 0, 0], dtype=tf.float32)
            )
            boundary.scene = self.scene
            boundary.radio_material = self.scene.radio_materials["concrete"]
            self.scene.add(boundary)

    def add_transmitter(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Transmitter:
        """Add transmitter using Sionna's built-in management"""
        tx = Transmitter(name=name, position=position, orientation=orientation)
        tx.scene = self.scene
        
        # Configure antenna array
        tx_array = PlanarArray(
            num_rows=self.config.bs_array[0],
            num_cols=self.config.bs_array[1],
            vertical_spacing=self.config.bs_array_spacing,
            horizontal_spacing=self.config.bs_array_spacing,
            pattern="tr38901",
            polarization="V"
        )
        tx.array = tx_array
        
        self.scene.add(tx)
        return tx

    def add_ris(self, name: str, position: tf.Tensor, orientation: tf.Tensor,
                num_rows: int, num_cols: int, dtype=tf.complex64) -> RIS:
        """Add a RIS to the scene using Sionna's built-in management"""
        print(f"[DEBUG PRINT] Adding RIS {name}")
        
        try:
            # Step 1: Create RIS object first
            ris = RIS(
                name=name,
                position=position,
                orientation=orientation,
                num_rows=num_rows,
                num_cols=num_cols,
                dtype=dtype
            )
            print(f"[DEBUG PRINT] RIS object created")

            # Step 2: Set scene reference
            ris.scene = self._scene
            print(f"[DEBUG PRINT] Scene reference set")

            # Step 3: Create and register material before assigning to RIS
            metal = RadioMaterial(
                name="itu_metal",
                relative_permittivity=1.0,
                conductivity=1e7,
                dtype=dtype
            )
            self._scene.add(metal)
            print(f"[DEBUG PRINT] Metal material created and added to scene")

            # Step 4: Generate and set object ID
            if len(self._scene.objects) > 0:
                max_id = max(obj.object_id for obj in self._scene.objects.values())
            else:
                max_id = 0
            object_id = max_id + 1
            ris.object_id = object_id
            print(f"[DEBUG PRINT] Object ID {object_id} assigned")

            # Step 5: Register RIS in material's tracking set
            metal.add_object_using(object_id)
            print(f"[DEBUG PRINT] RIS registered with material")

            # Step 6: Set material reference on RIS
            ris._radio_material = metal
            print(f"[DEBUG PRINT] Material reference set on RIS")

            # Step 7: Add RIS to scene
            self._scene.add(ris)
            print(f"[DEBUG PRINT] RIS added to scene")

            # Step 8: Configure phase profile
            cell_grid = CellGrid(num_rows=num_rows, num_cols=num_cols, dtype=dtype)
            phase_profile = DiscretePhaseProfile(cell_grid=cell_grid, dtype=dtype)
            ris.phase_profile = phase_profile
            print(f"[DEBUG PRINT] Phase profile configured")

            return ris

        except Exception as e:
            print(f"[DEBUG PRINT] Error in add_ris: {str(e)}")
            # Clean up if needed
            if 'object_id' in locals():
                try:
                    metal.discard_object_using(object_id)
                except:
                    pass
            raise RuntimeError(f"Failed to add RIS: {str(e)}") from e

    def add_receiver(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Receiver:
        """Add receiver using Sionna's built-in management"""
        rx = Receiver(name=name, position=position, orientation=orientation)
        rx.scene = self.scene
        
        # Configure antenna array
        rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=self.config.rx_array_spacing,
            horizontal_spacing=self.config.rx_array_spacing,
            pattern="iso",
            polarization="V"
        )
        rx.array = rx_array
        
        self.scene.add(rx)
        return rx