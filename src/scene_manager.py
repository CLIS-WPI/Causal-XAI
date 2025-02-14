#src/scene_manager.py
import tensorflow as tf
from sionna.rt import (
    Scene,
    Transmitter,
    Receiver,
    RIS,
    RadioMaterial,
    PlanarArray
)
import logging
from config import SmartFactoryConfig
from sionna.rt import CellGrid, DiscretePhaseProfile

logger = logging.getLogger(__name__)

class SceneManager:
    """
    A stripped-down SceneManager that does NOT create walls/floor/ceiling or shelves.
    We assume your factory_scene.xml now handles passive geometry. 
    This class only provides methods for adding transmitters, receivers, RIS, or materials if needed.
    """
    def __init__(self, scene: Scene, config: SmartFactoryConfig):
        self._scene = scene
        self.config = config

        # Initialize scene frequency
        self._scene.frequency = tf.cast(config.carrier_frequency, scene.dtype.real_dtype)

        # Optionally add or override materials if you like:
        self._add_materials()

        # COMMENTED OUT: No longer add boundaries or shelves in Python
        # self._add_room_boundaries()
        # self._add_metal_shelves()

        # No custom geometry creation, so no "visibility" logic needed
        # But if you still want to keep it for debugging, see below
        self.visibility_results = {}

        logger.info("SceneManager initialized. Passive objects are loaded from XML.")
        logger.info(f"Scene frequency set to {self._scene.frequency:.2f} Hz")

    def _add_materials(self):
        """
        Example: Add or override materials if needed.
        If your XML already defines them, you can remove or adapt this method.
        """
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

    # COMMENTED OUT: No in-Python geometry creation
    # def _add_room_boundaries(self):
    #     pass

    # def _add_metal_shelves(self):
    #     pass

    def add_transmitter(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Transmitter:
        """
        Add a new transmitter (e.g., base station).
        """
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

    def add_receiver(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Receiver:
        """
        Add a new receiver (e.g., AGV).
        """
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

    def add_ris(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> RIS:
        """
        Example method to add a Reconfigurable Intelligent Surface (RIS), if needed.
        """
        ris = RIS(
            name=name,
            position=position,
            orientation=orientation,
            num_rows=self.config.ris_elements[0],
            num_cols=self.config.ris_elements[1],
            dtype=self._scene.dtype
        )
        # Example of setting up a discrete phase profile
        cell_grid = CellGrid(
            num_rows=self.config.ris_elements[0],
            num_cols=self.config.ris_elements[1],
            dtype=self._scene.dtype
        )
        phase_values = tf.zeros([1, self.config.ris_elements[0],
                                self.config.ris_elements[1]],
                                dtype=tf.float32)
        phase_profile = DiscretePhaseProfile(
            cell_grid=cell_grid,
            values=phase_values,
            dtype=self._scene.dtype
        )
        ris.phase_profile = phase_profile

        ris.scene = self._scene
        self._scene.add(ris)
        return ris

    # If you still want a custom LOS check, you can keep or remove the method below.
    # Just note that shelves/walls are now from XML, so python-based bounding box checks might be moot.
    # def _check_visibility(self):
    #     return {}

