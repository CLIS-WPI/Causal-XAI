"""
Scene Manager for Sionna RT simulations.
Handles thread-safe scene object management and validation.
"""
#scene_manager.py
import tensorflow as tf
import numpy as np
from sionna.rt import Scene, Transmitter, Receiver, RIS, RadioMaterial, PlanarArray
from typing import Dict, List, Optional, Set, Tuple, Any
import threading
from dataclasses import dataclass
from enum import Enum, auto
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ObjectType(Enum):
    TRANSMITTER = auto()
    RECEIVER = auto()
    RIS = auto()
    MATERIAL = auto()
    SCENE_OBJECT = auto()

@dataclass
class SceneObject:
    """Represents a tracked object in the scene"""
    id: int
    name: str
    obj_type: ObjectType
    radio_material: Optional[str] = None
    reference: Any = None  # Store reference to actual object
    creation_time: datetime = datetime.now()

@dataclass
class AGVState:
    """Tracks AGV state information"""
    position: tf.Tensor
    velocity: tf.Tensor
    orientation: tf.Tensor
    receiver: Receiver
    waypoints: List[tf.Tensor]
    current_waypoint_idx: int
    last_update_time: datetime = datetime.now()

class SceneManager:
    """Thread-safe manager for Sionna scene objects and radio materials"""
    
    def __init__(self, scene: Scene, config: Any):
        """
        Initialize scene manager
        
        Args:
            scene: Sionna Scene object
            config: Configuration object containing scene parameters
        """
        self._scene = scene
        self._config = config
        self._lock = threading.Lock()
        self._object_registry: Dict[int, SceneObject] = {}
        self._material_registry: Dict[str, Set[int]] = {}
        self._agv_states: Dict[str, AGVState] = {}
        self._next_id = 0
        
        # Initialize registries
        self._initialize_registries()
        
        # Room boundaries from config
        self._room_boundaries = {
            'x': (0, config.room_dim[0]),
            'y': (0, config.room_dim[1]),
            'z': (0, config.room_dim[2])
        }
        
        # Add room boundaries
        self._add_room_boundaries()
        
        logger.info("Scene manager initialized successfully")

    def _add_room_boundaries(self):
        """Add walls, floor and ceiling to the scene"""
        try:
            # Create concrete material
            concrete = RadioMaterial(
                name="concrete",
                relative_permittivity=4.5,
                conductivity=0.01,
                dtype=self._scene.dtype
            )
            
            # Add material to scene
            self._scene.add(concrete)
            
            # Room dimensions
            length, width, height = self._config.room_dim
            
            # Create rectangular surfaces for room boundaries using Sionna's built-in components
            # Add floor
            floor = RadioMaterial(
                name="floor",
                relative_permittivity=4.5,
                conductivity=0.01,
                dtype=self._scene.dtype
            )
            self._scene.add(floor)
            
            # Add ceiling
            ceiling = RadioMaterial(
                name="ceiling",
                relative_permittivity=4.5,
                conductivity=0.01,
                dtype=self._scene.dtype
            )
            self._scene.add(ceiling)
            
            # Add walls
            walls = [
                "wall_front",
                "wall_back", 
                "wall_left",
                "wall_right"
            ]
            
            for wall_name in walls:
                wall = RadioMaterial(
                    name=wall_name,
                    relative_permittivity=4.5,
                    conductivity=0.01,
                    dtype=self._scene.dtype
                )
                self._scene.add(wall)
                
        except Exception as e:
            logger.error(f"Failed to add room boundaries: {str(e)}")
            raise

    def _initialize_registries(self):
        """Initialize object and material registries from existing scene state"""
        with self._lock:
            # Scan existing objects and collect IDs
            existing_ids = []
            
            # Check scene objects
            for name, obj in self._scene.objects.items():
                if hasattr(obj, 'object_id'):
                    existing_ids.append(obj.object_id)
                    self._register_existing_object(obj, ObjectType.SCENE_OBJECT)
            
            # Check transmitters
            for name, tx in self._scene.transmitters.items():
                if hasattr(tx, 'object_id'):
                    existing_ids.append(tx.object_id)
                    self._register_existing_object(tx, ObjectType.TRANSMITTER)
            
            # Check RIS
            for name, ris in self._scene.ris.items():
                if hasattr(ris, 'object_id'):
                    existing_ids.append(ris.object_id)
                    self._register_existing_object(ris, ObjectType.RIS)
            
            # Initialize next_id
            self._next_id = max(existing_ids) + 1 if existing_ids else 0
            
            logger.debug(f"Initialized registries with {len(self._object_registry)} objects")

    def _register_existing_object(self, obj: Any, obj_type: ObjectType):
        """Register an existing scene object in our tracking system"""
        scene_obj = SceneObject(
            id=obj.object_id,
            name=obj.name if hasattr(obj, 'name') else f"object_{obj.object_id}",
            obj_type=obj_type,
            radio_material=obj.radio_material.name if hasattr(obj, 'radio_material') else None,
            reference=obj
        )
        self._object_registry[obj.object_id] = scene_obj
        
        # Track material association
        if scene_obj.radio_material:
            if scene_obj.radio_material not in self._material_registry:
                self._material_registry[scene_obj.radio_material] = set()
            self._material_registry[scene_obj.radio_material].add(obj.object_id)

    def _generate_object_id(self) -> int:
        """Thread-safe generation of next available object ID"""
        with self._lock:
            object_id = self._next_id
            self._next_id += 1
            return object_id

    def _register_object(self, obj: Any, obj_type: ObjectType,
                        radio_material: Optional[str] = None) -> int:
        """Register a new object in the scene, with extra debug prints."""
        print("[DEBUG PRINT] Entering _register_object()")
        print(f"[DEBUG PRINT] _register_object() called with obj_type={obj_type}, radio_material={radio_material}")

        with self._lock:
            print("[DEBUG PRINT] _register_object() lock acquired")
            try:
                logger.debug("=== START REGISTRATION PROCESS ===")
                logger.debug(f"Object type: {obj_type}")
                logger.debug(f"Object details: {type(obj)}")
                print("[DEBUG PRINT] Generating a new object ID...")

                # Generate unique ID
                try:
                    object_id = self._generate_object_id()
                    logger.debug(f"Generated ID: {object_id}")
                    print(f"[DEBUG PRINT] _register_object() generated object_id={object_id}")
                except Exception as e:
                    logger.error(f"ID generation failed: {str(e)}")
                    print(f"[DEBUG PRINT] Exception generating object ID: {e}")
                    raise

                # Get object name
                print("[DEBUG PRINT] Extracting object name...")
                try:
                    name = obj.name if hasattr(obj, 'name') else f"object_{object_id}"
                    logger.debug(f"Object name: {name}")
                    print(f"[DEBUG PRINT] _register_object() got name='{name}'")
                except Exception as e:
                    logger.error(f"Name extraction failed: {str(e)}")
                    print(f"[DEBUG PRINT] Exception extracting name: {e}")
                    raise

                # Check object state
                print("[DEBUG PRINT] Checking object state (scene, array, position)...")
                try:
                    logger.debug(f"Object state check:")
                    logger.debug(f"- Has scene: {hasattr(obj, 'scene')}")
                    logger.debug(f"- Has array: {hasattr(obj, 'array')}")
                    logger.debug(f"- Has position: {hasattr(obj, 'position')}")
                    print("[DEBUG PRINT] Object state check complete")
                except Exception as e:
                    logger.error(f"State check failed: {str(e)}")
                    print(f"[DEBUG PRINT] Exception in object state check: {e}")
                    raise

                # Create registration
                print("[DEBUG PRINT] Creating SceneObject...")
                try:
                    scene_object = SceneObject(
                        id=object_id,
                        name=name,
                        obj_type=obj_type,
                        radio_material=radio_material,
                        reference=obj
                    )
                    logger.debug("SceneObject created successfully")
                    print("[DEBUG PRINT] SceneObject created successfully")
                except Exception as e:
                    logger.error(f"SceneObject creation failed: {str(e)}")
                    logger.error(f"Creation parameters: id={object_id}, name={name}, type={obj_type}")
                    print(f"[DEBUG PRINT] Exception creating SceneObject: {e}")
                    raise

                # Registry update
                print("[DEBUG PRINT] Adding to object registry...")
                try:
                    self._object_registry[object_id] = scene_object
                    logger.debug("Added to object registry")
                    print(f"[DEBUG PRINT] Successfully added object_id={object_id} to registry")
                except Exception as e:
                    logger.error(f"Registry update failed: {str(e)}")
                    print(f"[DEBUG PRINT] Exception updating registry: {e}")
                    raise

                logger.debug("=== REGISTRATION COMPLETE ===")
                print(f"[DEBUG PRINT] Registration complete for object_id={object_id}")
                print("[DEBUG PRINT] Exiting _register_object() normally...")
                return object_id

            except Exception as e:
                logger.error("!!! REGISTRATION FAILED !!!")
                logger.error(f"Error: {str(e)}")
                logger.error("Traceback:", exc_info=True)
                print(f"[DEBUG PRINT] EXCEPTION in _register_object(): {e}")
                raise
            
    def _get_or_create_material(self, material_name: str, 
                            dtype=tf.complex64) -> RadioMaterial:
        """Get existing material or create new one"""
        with self._lock:
            material = self._scene.get(material_name)
            if material is None:
                material = RadioMaterial(material_name, dtype=dtype)
                material.scene = self._scene
                self._scene.add(material)
                self._material_registry[material_name] = set()
            return material

    def add_transmitter(self, name: str, position: tf.Tensor,
                        orientation: tf.Tensor, dtype=tf.complex64) -> Transmitter:
        """Add a transmitter to the scene"""
        print(f"[DEBUG PRINT] Entering add_transmitter() for '{name}'")
        print(f"[DEBUG PRINT] About to acquire lock in add_transmitter() for '{name}'")

        with self._lock:
            print(f"[DEBUG PRINT] Lock acquired for add_transmitter() - {name}")
            try:
                # High-level logging
                logger.debug(f"Starting to create transmitter {name}...")
                logger.debug(f"Position: {position.numpy()}, Orientation: {orientation.numpy()}, dtype: {dtype}")

                print(f"[DEBUG PRINT] Creating Transmitter object: {name}")
                tx = Transmitter(
                    name=name,
                    position=position,
                    orientation=orientation,
                    dtype=dtype
                )
                logger.debug(f"Transmitter {name} object created successfully")
                print(f"[DEBUG PRINT] Finished creating Transmitter object: {name}")

                logger.debug(f"Setting scene reference for {name}...")
                tx.scene = self._scene
                logger.debug(f"Scene reference set successfully for {name}")
                print(f"[DEBUG PRINT] Transmitter '{name}' scene reference assigned")

                # Convert spacing to proper tensor
                logger.debug(f"Converting array spacing. Original value: {self._config.bs_array_spacing}")
                array_spacing = tf.cast(self._config.bs_array_spacing, dtype=tf.float32)
                logger.debug(f"Array spacing converted to tensor: {array_spacing}")
                print(f"[DEBUG PRINT] BS array spacing (float32): {array_spacing.numpy()}")

                # Configure antenna array
                try:
                    logger.debug(f"Creating antenna array with params:")
                    logger.debug(f"- Rows: {self._config.bs_array[0]}")
                    logger.debug(f"- Cols: {self._config.bs_array[1]}")
                    logger.debug(f"- Spacing: {array_spacing}")

                    print(f"[DEBUG PRINT] About to create PlanarArray for '{name}'")
                    tx_array = PlanarArray(
                        num_rows=self._config.bs_array[0],
                        num_cols=self._config.bs_array[1],
                        vertical_spacing=array_spacing,
                        horizontal_spacing=array_spacing,
                        pattern="tr38901",  # Hardcode known working pattern
                        polarization="V",   # Simplify polarization
                        dtype=dtype
                    )
                    logger.debug("Antenna array object created successfully")
                    print(f"[DEBUG PRINT] PlanarArray created successfully for '{name}'")

                    logger.debug(f"Assigning antenna array to transmitter {name}...")
                    tx.array = tx_array
                    logger.debug("Antenna array assigned successfully")
                    print(f"[DEBUG PRINT] Antenna array assigned to transmitter '{name}'")

                except Exception as array_error:
                    logger.error(f"Failed to create/assign antenna array: {str(array_error)}")
                    logger.error("Array error traceback:", exc_info=True)
                    print(f"[DEBUG PRINT] Exception while creating PlanarArray: {array_error}")
                    raise

                logger.debug(f"Registering {name} in object registry...")
                print(f"[DEBUG PRINT] About to _register_object() for '{name}'")
                object_id = self._register_object(tx, ObjectType.TRANSMITTER)
                tx.object_id = object_id
                logger.debug(f"Object registered and ID {object_id} assigned to {name}")
                print(f"[DEBUG PRINT] Done registering '{name}' with object ID = {object_id}")

                logger.debug(f"Adding {name} to scene...")
                print(f"[DEBUG PRINT] About to call self._scene.add(tx) for '{name}'")
                self._scene.add(tx)
                logger.debug(f"Successfully added {name} to scene")
                print(f"[DEBUG PRINT] Done calling self._scene.add(tx) for '{name}'")

                logger.info(f"Successfully completed transmitter {name} setup with ID {object_id}")
                print(f"[DEBUG PRINT] Exiting add_transmitter() for '{name}' with success")
                return tx

            except Exception as e:
                logger.error(f"Failed to add transmitter {name}: {str(e)}")
                logger.error("Full error traceback:", exc_info=True)
                print(f"[DEBUG PRINT] ERROR in add_transmitter() for '{name}': {e}")
                if 'object_id' in locals():
                    logger.debug(f"Cleaning up - unregistering object ID {object_id}")
                    print(f"[DEBUG PRINT] Unregistering object ID {object_id}")
                    self._unregister_object(object_id)
                raise


    def add_ris(self, name: str, position: tf.Tensor, orientation: tf.Tensor,
                num_rows: int, num_cols: int, dtype=tf.complex64) -> RIS:
        """Add a RIS to the scene"""
        with self._lock:
            try:
                # First get/create material
                metal_material = self._get_or_create_material("itu_metal", dtype)
                
                # Create RIS
                ris = RIS(
                    name=name,
                    position=position,
                    orientation=orientation,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    dtype=dtype
                )
                
                # Set scene reference before material
                ris.scene = self._scene
                
                # Register object
                object_id = self._register_object(ris, ObjectType.RIS, "itu_metal")
                ris.object_id = object_id
                
                # Set material after ID assigned
                ris.radio_material = metal_material
                
                # Add to scene
                self._scene.add(ris)
                logger.info(f"Added RIS {name} with ID {object_id}")
                return ris
                
            except Exception as e:
                logger.error(f"Failed to add RIS {name}: {str(e)}")
                self._unregister_object(object_id)
                raise

    def add_receiver(self, name: str, position: tf.Tensor,
                    orientation: tf.Tensor, dtype=tf.complex64) -> Receiver:
        """Add a receiver to the scene"""
        with self._lock:
            try:
                # Create receiver
                rx = Receiver(
                    name=name,
                    position=position,
                    orientation=orientation,
                    dtype=dtype
                )
                
                # Set scene reference
                rx.scene = self._scene
                
                # Register object
                object_id = self._register_object(rx, ObjectType.RECEIVER)
                rx.object_id = object_id
                
                # Configure antenna array
                rx_array = PlanarArray(
                    num_rows=1,
                    num_cols=1,
                    vertical_spacing=self._config.rx_array_spacing,
                    horizontal_spacing=self._config.rx_array_spacing,
                    pattern="iso",
                    polarization="V",
                    dtype=dtype
                )
                rx.array = rx_array
                
                # Add to scene
                self._scene.add(rx)
                logger.info(f"Added receiver {name} with ID {object_id}")
                return rx
                
            except Exception as e:
                logger.error(f"Failed to add receiver {name}: {str(e)}")
                self._unregister_object(object_id)
                raise

    def validate_scene(self):
        """Validate entire scene configuration"""
        with self._lock:
            try:
                # Calculate total objects
                total_objects = (len(self._scene.objects) + 
                            len(self._scene.transmitters) + 
                            len(self._scene.ris))
                
                # Collect all object IDs
                all_ids = []
                for obj in self._scene.objects.values():
                    if hasattr(obj, 'object_id'):
                        all_ids.append(obj.object_id)
                for tx in self._scene.transmitters.values():
                    if hasattr(tx, 'object_id'):
                        all_ids.append(tx.object_id)
                for ris in self._scene.ris.values():
                    if hasattr(ris, 'object_id'):
                        all_ids.append(ris.object_id)
                
                # Validate IDs
                if len(all_ids) != total_objects:
                    raise ValueError(f"Missing object IDs: found {len(all_ids)} for {total_objects} objects")
                if len(all_ids) != len(set(all_ids)):
                    raise ValueError("Duplicate object IDs detected")
                if max(all_ids) >= self._next_id:
                    raise ValueError(f"Invalid object ID range")
                
                # Validate material registry
                for material_name, object_ids in self._material_registry.items():
                    material = self._scene.get(material_name)
                    if material is None:
                        raise ValueError(f"Material {material_name} not found")
                    
                    # Check all objects using this material exist
                    for obj_id in object_ids:
                        if obj_id not in self._object_registry:
                            raise ValueError(f"Object {obj_id} using material {material_name} not found")
                
                logger.info("Scene validation completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Scene validation failed: {str(e)}")
                return False

    def _unregister_object(self, object_id: int):
        """Remove object registration and cleanup materials"""
        with self._lock:
            if object_id in self._object_registry:
                obj = self._object_registry[object_id]
                
                # Remove material association
                if obj.radio_material:
                    self._material_registry[obj.radio_material].discard(object_id)
                    
                # Remove from registry
                del self._object_registry[object_id]
                logger.debug(f"Unregistered object with ID {object_id}")

    def _atomic_operation(self, operation):
        """Execute operation atomically with rollback"""
        with self._lock:
            checkpoint = self._create_checkpoint()
            try:
                result = operation()
                return result
            except Exception as e:
                self._restore_checkpoint(checkpoint)
                raise e
            
    def _validate_position(self, position: tf.Tensor, obj_name: str):
        """Validate object position within scene bounds"""
        pos = position.numpy()
        for axis, (min_val, max_val) in zip(['x','y','z'], self._room_boundaries.values()):
            if not min_val <= pos[axis] <= max_val:
                raise ValueError(f"{obj_name} position {pos[axis]} outside {axis} bounds")

    def _create_checkpoint(self):
        """Create checkpoint of current scene state"""
        return {
            'object_registry': self._object_registry.copy(),
            'material_registry': {k: v.copy() for k, v in self._material_registry.items()},
            'agv_states': self._agv_states.copy(),
            'next_id': self._next_id
        }

    def _restore_checkpoint(self, checkpoint):
        """Restore scene state from checkpoint"""
        self._object_registry = checkpoint['object_registry']
        self._material_registry = checkpoint['material_registry']
        self._agv_states = checkpoint['agv_states']
        self._next_id = checkpoint['next_id']
                    
    def cleanup(self):
        """Cleanup scene manager state"""
        with self._lock:
            self._object_registry.clear()
            self._material_registry.clear()
            self._agv_states.clear()
            logger.info("Scene manager cleaned up")