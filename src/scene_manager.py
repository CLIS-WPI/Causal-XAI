#src/scene_manager.py
# Core libraries
import mitsuba
import tensorflow as tf
import logging

# Sionna imports
from sionna.rt.scenes import Scene
from sionna.rt.components import Transmitter, Receiver
from sionna.rt.antenna import PlanarArray
from sionna.rt.materials import RadioMaterial

# Local imports
from config import SmartFactoryConfig
logger = logging.getLogger(__name__)

class SceneManager:
    """
    Manages the Sionna scene by adding transmitters, receivers, and materials.
    Assumes geometry (walls, floor, ceiling, shelves) is already loaded from PLY files via scene_setup.py.
    """
    def __init__(self, scene: Scene, config: SmartFactoryConfig):
        self._scene = scene
        self.config = config

        # Set scene frequency
        self._scene.frequency = tf.cast(config.carrier_frequency, tf.float32)
        logger.info(f"Scene frequency set to {self._scene.frequency:.2f} Hz")

        # Add or override materials from config
        self._add_materials()

        # No geometry creation here; assume PLY files are loaded in scene_setup.py
        logger.info("SceneManager initialized. Passive geometry is loaded from PLY files.")

        # Define radio materials from config if not already present
        if 'factory_concrete' not in self._scene.radio_materials:
            concrete = RadioMaterial(
                name="factory_concrete",
                relative_permittivity=self.config.materials['concrete']['relative_permittivity'],
                conductivity=self.config.materials['concrete']['conductivity'],
                scattering_coefficient=self.config.materials['concrete']['scattering_coefficient'],
                xpd_coefficient=self.config.materials['concrete']['xpd_coefficient']
            )
            self._scene.add(concrete)
            logger.info(f"Added radio material: factory_concrete "
                        f"(permittivity={self.config.materials['concrete']['relative_permittivity']}, "
                        f"conductivity={self.config.materials['concrete']['conductivity']})")

        if 'factory_metal' not in self._scene.radio_materials:
            metal = RadioMaterial(
                name="factory_metal",
                relative_permittivity=self.config.materials['metal']['relative_permittivity'],
                conductivity=self.config.materials['metal']['conductivity'],
                scattering_coefficient=self.config.materials['metal']['scattering_coefficient'],
                xpd_coefficient=self.config.materials['metal']['xpd_coefficient']
            )
            self._scene.add(metal)
            logger.info(f"Added radio material: factory_metal "
                        f"(permittivity={self.config.materials['metal']['relative_permittivity']}, "
                        f"conductivity={self.config.materials['metal']['conductivity']})")

        logger.debug("Registered radio materials: " + ", ".join(self._scene.radio_materials.keys()))

    def _add_materials(self):
        """Add materials to the scene if not already present. Relies on config definitions."""
        # This method is now redundant since materials are added directly in __init__
        logger.debug("Materials assumed to be defined by scene_setup.py or added in SceneManager init. Skipping additional material checks.")

    def add_transmitter(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Transmitter:
        """Add a new transmitter (e.g., base station)."""
        tx = Transmitter(name=name, position=position, orientation=orientation)
        tx_array = PlanarArray(
            num_rows=self.config.bs_array['num_rows'],
            num_cols=self.config.bs_array['num_cols'],
            vertical_spacing=self.config.bs_array.get('vertical_spacing', 0.7),
            horizontal_spacing=self.config.bs_array.get('horizontal_spacing', 0.5),
            pattern=self.config.bs_array.get('pattern', "tr38901"),
            polarization=self.config.bs_array.get('polarization', "VH")
        )
        tx.array = tx_array
        self._scene.add(tx)
        logger.debug(f"Added transmitter {name} at position {position.numpy()}")
        return tx

    def add_receiver(self, name: str, position: tf.Tensor, orientation: tf.Tensor) -> Receiver:
        """Add a new receiver (e.g., AGV)."""
        rx = Receiver(name=name, position=position, orientation=orientation)
        rx_array = PlanarArray(
            num_rows=self.config.agv_array['num_rows'],
            num_cols=self.config.agv_array['num_cols'],
            vertical_spacing=self.config.agv_array.get('vertical_spacing', 0.5),
            horizontal_spacing=self.config.agv_array.get('horizontal_spacing', 0.5),
            pattern=self.config.agv_array.get('pattern', "tr38901"),
            polarization=self.config.agv_array.get('polarization', "VH")
        )
        rx.array = rx_array
        self._scene.add(rx)
        logger.debug(f"Added receiver {name} at position {position.numpy()}")
        return rx

    def update_scene_with_agv_positions(self, agv_positions, agv_orientations=None):
        """Update receiver positions and optionally orientations based on AGV movements."""
        for i, position in enumerate(agv_positions):
            receiver_name = f"rx_agv_{i}"
            if receiver_name in self._scene.receivers:
                self._scene.receivers[receiver_name].position = tf.constant(position, dtype=tf.float32)
                if agv_orientations is not None and i < len(agv_orientations):
                    self._scene.receivers[receiver_name].orientation = tf.constant(agv_orientations[i], dtype=tf.float32)
                    logger.debug(f"Updated {receiver_name} position to {position} "
                                 f"and orientation to {agv_orientations[i]}")
                else:
                    logger.debug(f"Updated {receiver_name} position to {position} (no orientation update)")
            else:
                logger.warning(f"Receiver {receiver_name} not found in scene for position update")

    @property
    def scene(self) -> Scene:
        """Return the managed scene."""
        return self._scene