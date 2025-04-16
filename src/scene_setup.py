# src/scene_setup.py
# src/scene_setup.py
import os
import logging
import numpy as np
import tensorflow as tf
import mitsuba as mi
from typing import Optional, Dict

from sionna.rt import (
    Scene,
    Transmitter,
    Receiver,
    SceneObject,
    PlanarArray,
    RadioMaterial
)
from config import SmartFactoryConfig

logger = logging.getLogger(__name__)
mi.set_variant("cuda_ad_rgb")

class RTScene:
    def __init__(self, dtype: tf.DType = tf.complex64) -> None:
        self._dtype: tf.DType = dtype
        self.transmitters: Dict[str, Transmitter] = {}
        self.receivers: Dict[str, Receiver] = {}
        self.objects: Dict[str, SceneObject] = {}
        self.radio_materials: Dict[str, RadioMaterial] = {}
        self.synthetic_array: bool = False
        self.frequency: Optional[tf.Tensor] = None
        self.tx_array: Optional[PlanarArray] = None
        self.rx_array: Optional[PlanarArray] = None
        self.scene: Optional[Scene] = None

    def add(self, obj: object) -> None:
        if isinstance(obj, RadioMaterial):
            self.radio_materials[obj.name] = obj
        elif hasattr(obj, "position") and hasattr(obj, "orientation"):
            if obj.name == "bs":
                self.transmitters[obj.name] = obj
            elif obj.name.startswith("rx_"):
                self.receivers[obj.name] = obj
            else:
                self.objects[obj.name] = obj
        else:
            self.objects[obj.name] = obj

    def validate(self):
        if not self.transmitters:
            raise ValueError("Scene must have at least one transmitter")
        if not self.receivers:
            raise ValueError("Scene must have at least one receiver")


def setup_scene(config: SmartFactoryConfig) -> RTScene:
    scene = RTScene()
    scene.scene = Scene()

    # Create radio materials
    concrete = RadioMaterial("factory_concrete",
        relative_permittivity=config.materials['concrete']['relative_permittivity'],
        conductivity=config.materials['concrete']['conductivity'])
    metal = RadioMaterial("factory_metal",
        relative_permittivity=config.materials['metal']['relative_permittivity'],
        conductivity=config.materials['metal']['conductivity'])

    # Add materials to both scene objects
    scene.add(concrete)
    scene.add(metal)
    scene.scene.add(concrete)
    scene.scene.add(metal)

    # Create a basic holder material
    holder = mi.load_dict({
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}
    })

    meshes_dir = os.path.join(os.path.dirname(__file__), "meshes")
    for ply_file in os.listdir(meshes_dir):
        if ply_file.endswith(".ply"):
            name = ply_file[:-4]
            filepath = os.path.join(meshes_dir, ply_file)
            radio_mat = concrete if any(x in name for x in ["wall", "floor", "ceiling"]) else metal

            # Create shape with the holder material
            mi_shape = mi.load_dict({
                "type": "ply",
                "filename": filepath,
                "face_normals": True,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [0.5, 0.5, 0.5]
                    }
                }
            })

            
            obj = SceneObject(name=name, mi_shape=mi_shape, radio_material=radio_mat)
            scene.scene.add(obj)
            scene.add(obj)
            logger.debug(f"Added {name} with material {radio_mat.name}")


    # Add transmitter
    bs = Transmitter("bs",
        position=tf.constant(config.bs_position, dtype=tf.float32),
        orientation=tf.constant(config.bs_orientation, dtype=tf.float32))
    bs_array = PlanarArray(**config.bs_array)
    bs.array = bs_array
    scene.add(bs)
    scene.scene.add(bs)
    scene.tx_array = bs_array
    scene.synthetic_array = True
    scene.frequency = tf.constant(config.carrier_frequency, tf.float32)

    # Add receivers
    for i in range(config.num_agvs):
        pos = config.agv_trajectories[f"agv_{i+1}"][0]
        ori = config.agv_orientations[i]
        rx = Receiver(f"rx_agv_{i}",
            position=tf.constant(pos, dtype=tf.float32),
            orientation=tf.constant(ori, dtype=tf.float32))
        rx_array = PlanarArray(**config.agv_array)
        rx.array = rx_array
        scene.add(rx)
        scene.scene.add(rx)
        if i == 0:
            scene.rx_array = rx_array

    # Configure tracing
    scene.scene.los = True
    scene.scene.specular = True
    scene.scene.diffraction = True
    scene.scene.scattering = True
    scene.scene.max_depth = config.ray_tracing['max_depth']
    scene.scene.num_samples = config.ray_tracing['num_samples']

    scene.validate()
    return scene
