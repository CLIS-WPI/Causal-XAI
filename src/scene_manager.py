"""
Scene Manager for Sionna RT simulations.
Handles thread-safe scene object management and validation.
"""
#scene_manager.py
import tensorflow as tf
import numpy as np
from sionna.rt import (
    Scene, 
    Transmitter, 
    Receiver, 
    RIS, 
    RadioMaterial, 
    PlanarArray,
    CellGrid,  # Add this import
    DiscretePhaseProfile
)
from typing import Dict, List, Optional, Set, Tuple, Any
import threading
from dataclasses import dataclass
from enum import Enum, auto
import logging
from datetime import datetime
import time
from sionna.rt import DiscretePhaseProfile

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
        """Add room boundaries with proper registration"""
        try:
            # Create concrete material
            concrete = self._get_or_create_material("concrete", self._scene.dtype)
            
            length, width, height = self._config.room_dim
            boundaries = [
                ("floor", [length/2, width/2, 0]),
                ("ceiling", [length/2, width/2, height]),
                ("wall_front", [length/2, 0, height/2]),
                ("wall_back", [length/2, width, height/2]),
                ("wall_left", [0, width/2, height/2]),
                ("wall_right", [length, width/2, height/2])
            ]

            for name, position in boundaries:
                # Create boundary
                boundary = Transmitter(
                    name=name,
                    position=tf.constant(position, dtype=tf.float32),
                    orientation=tf.constant([0, 0, 0], dtype=tf.float32)
                )
                
                # Set scene and material
                boundary.scene = self._scene
                boundary.radio_material = concrete
                
                # Register in manager
                object_id = self._register_object(boundary, ObjectType.SCENE_OBJECT, "concrete")
                boundary.object_id = object_id
                
                # Add to scene
                self._scene.add(boundary)
                
            logger.info("Room boundaries added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add room boundaries: {str(e)}")
            raise

    def _get_or_create_material(self, material_name: str, dtype=tf.complex64) -> RadioMaterial:
        """Get or create material with enhanced error handling"""
        try:
            # Check existing material
            if material_name in self._scene.radio_materials:
                return self._scene.radio_materials[material_name]

            # Create new material
            properties = {
                "itu_metal": {"permittivity": 1.0, "conductivity": 1e7},
                "concrete": {"permittivity": 4.5, "conductivity": 0.01}
            }.get(material_name, {"permittivity": 1.0, "conductivity": 0.0})

            material = RadioMaterial(
                name=material_name,
                relative_permittivity=properties["permittivity"],
                conductivity=properties["conductivity"],
                dtype=dtype
            )

            # Add to scene
            self._scene.add(material)

            # Initialize registry entry
            if material_name not in self._material_registry:
                self._material_registry[material_name] = set()

            return material

        except Exception as e:
            logger.error(f"Failed to get/create material {material_name}: {str(e)}")
            raise 
    
    def validate_scene(self) -> bool:
        """Comprehensive scene validation"""
        with self._lock:
            try:
                # Check object counts
                scene_objects = len(self._scene.objects)
                registered_objects = len(self._object_registry)
                
                if scene_objects != registered_objects:
                    logger.error(f"Object count mismatch: Scene={scene_objects}, Registry={registered_objects}")
                    return False

                # Validate object IDs
                scene_ids = {obj.object_id for obj in self._scene.objects.values() 
                            if hasattr(obj, 'object_id')}
                registry_ids = set(self._object_registry.keys())
                
                if scene_ids != registry_ids:
                    logger.error("Object ID mismatch between scene and registry")
                    return False

                # Validate materials
                for obj_id, obj in self._object_registry.items():
                    if obj.radio_material:
                        if obj.radio_material not in self._material_registry:
                            logger.error(f"Missing material registration for object {obj_id}")
                            return False
                        if obj_id not in self._material_registry[obj.radio_material]:
                            logger.error(f"Inconsistent material registration for object {obj_id}")
                            return False

                logger.info("Scene validation passed")
                return True

            except Exception as e:
                logger.error(f"Scene validation failed: {str(e)}")
                return False


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

    def _register_object_unlocked(self, obj: Any, obj_type: ObjectType, radio_material: Optional[str] = None) -> int:
        """Register object without acquiring lock (for use when lock is already held)."""
        try:
            print(f"[DEBUG PRINT] Registering object without lock: {obj_type}")
            
            # Generate ID (without lock since we already have it)
            object_id = self._next_id
            self._next_id += 1
            
            # Create scene object
            name = obj.name if hasattr(obj, 'name') else f"object_{object_id}"
            scene_object = SceneObject(
                id=object_id,
                name=name,
                obj_type=obj_type,
                radio_material=radio_material,
                reference=obj
            )
            
            # Add to registry
            self._object_registry[object_id] = scene_object
            
            # Track material association
            if radio_material:
                if radio_material not in self._material_registry:
                    self._material_registry[radio_material] = set()
                self._material_registry[radio_material].add(object_id)
                
            print(f"[DEBUG PRINT] Object registered successfully with ID: {object_id}")
            return object_id
            
        except Exception as e:
            print(f"[DEBUG PRINT] Error in _register_object_unlocked: {str(e)}")
            raise RuntimeError(f"Failed to register object without lock: {str(e)}") from e
    
    def _generate_object_id(self) -> int:
        """Thread-safe generation of the next available object ID with detailed logging and safety checks."""
        print("[DEBUG PRINT] Entering _generate_object_id()")
        start_time = time.time()
        
        with self._lock:
            try:
                # Log the current state of the object registry and next_id
                print(f"[DEBUG PRINT] Current next_id: {self._next_id}")
                print(f"[DEBUG PRINT] Current registry size: {len(self._object_registry)}")
                
                # Generate a unique object ID
                object_id = self._next_id
                self._next_id += 1
                
                # Check for potential ID overflow or large jumps
                if object_id >= 1e7:  # Arbitrary large value for safety
                    raise ValueError(f"[DEBUG PRINT] Object ID {object_id} exceeds safe limit!")

                # Log time taken to generate ID
                duration = time.time() - start_time
                print(f"[DEBUG PRINT] Generated new ID: {object_id} in {duration:.4f} seconds")
                
                return object_id

            except ValueError as e:
                print(f"[DEBUG PRINT] ValueError in _generate_object_id: {e}")
                raise

            except Exception as e:
                print(f"[DEBUG PRINT] Failed to generate object ID: {e}")
                raise RuntimeError("Failed to generate object ID.") from e

    def _register_object(self, obj: Any, obj_type: ObjectType, radio_material: Optional[str] = None) -> int:
        """Thread-safe object registration with enhanced validation"""
        with self._lock:
            try:
                # Generate new ID
                object_id = self._next_id
                self._next_id += 1

                # Create SceneObject
                name = obj.name if hasattr(obj, 'name') else f"object_{object_id}"
                scene_object = SceneObject(
                    id=object_id,
                    name=name,
                    obj_type=obj_type,
                    radio_material=radio_material,
                    reference=obj
                )

                # Add to registry
                self._object_registry[object_id] = scene_object

                # Handle material registration
                if radio_material:
                    if radio_material not in self._material_registry:
                        self._material_registry[radio_material] = set()
                    self._material_registry[radio_material].add(object_id)

                return object_id

            except Exception as e:
                logger.error(f"Failed to register object: {str(e)}")
                raise RuntimeError(f"Object registration failed: {str(e)}") from e



    def add_transmitter(self, name: str, position: tf.Tensor, orientation: tf.Tensor, dtype=tf.complex64) -> Transmitter:
        """Add a transmitter to the scene with detailed debugging and deadlock prevention."""
        print(f"[DEBUG PRINT] Entering add_transmitter() for '{name}'")
        print(f"[DEBUG PRINT] Thread ID: {id(threading.current_thread())}")
        print(f"[DEBUG PRINT] Initial lock state: {self._lock.locked()}")
        start_time = time.time()

        with self._lock:
            print(f"[DEBUG PRINT] Lock acquired in add_transmitter() - {name}")
            print(f"[DEBUG PRINT] Lock acquisition time: {time.time() - start_time:.4f} seconds")
            
            try:
                # Step 1: Create Transmitter object
                print(f"[DEBUG PRINT] Step 1: Creating Transmitter object for {name}")
                tx_start = time.time()
                try:
                    tx = Transmitter(name=name, position=position, orientation=orientation, dtype=dtype)
                    print(f"[DEBUG PRINT] Transmitter object created at {hex(id(tx))}")
                    logger.debug(f"Transmitter '{name}' created successfully")
                except Exception as tx_error:
                    print(f"[DEBUG PRINT] Error creating Transmitter: {str(tx_error)}")
                    logger.error(f"Failed to create Transmitter: {str(tx_error)}")
                    raise
                print(f"[DEBUG PRINT] Transmitter creation took {time.time() - tx_start:.4f} seconds")

                # Step 2: Set scene reference
                print(f"[DEBUG PRINT] Step 2: Setting scene reference")
                scene_start = time.time()
                try:
                    tx.scene = self._scene
                    print(f"[DEBUG PRINT] Scene reference set successfully")
                except Exception as scene_error:
                    print(f"[DEBUG PRINT] Error setting scene reference: {str(scene_error)}")
                    raise
                print(f"[DEBUG PRINT] Scene reference assignment took {time.time() - scene_start:.4f} seconds")

                # Step 3: Configure antenna array
                print(f"[DEBUG PRINT] Step 3: Configuring antenna array")
                array_start = time.time()
                try:
                    # Convert spacing
                    print(f"[DEBUG PRINT] Converting array spacing from {self._config.bs_array_spacing}")
                    array_spacing = tf.cast(self._config.bs_array_spacing, dtype=tf.float32)
                    print(f"[DEBUG PRINT] Array spacing converted to {array_spacing.numpy()}")

                    # Create array
                    print(f"[DEBUG PRINT] Creating PlanarArray with configuration:")
                    print(f"[DEBUG PRINT] - Rows: {self._config.bs_array[0]}")
                    print(f"[DEBUG PRINT] - Columns: {self._config.bs_array[1]}")
                    print(f"[DEBUG PRINT] - Spacing: {array_spacing.numpy()}")
                    print(f"[DEBUG PRINT] - Pattern: tr38901")
                    
                    tx_array = PlanarArray(
                        num_rows=self._config.bs_array[0],
                        num_cols=self._config.bs_array[1],
                        vertical_spacing=array_spacing,
                        horizontal_spacing=array_spacing,
                        pattern="tr38901",
                        polarization="V",
                        dtype=dtype
                    )
                    print(f"[DEBUG PRINT] PlanarArray created at {hex(id(tx_array))}")

                    # Assign array
                    tx.array = tx_array
                    print(f"[DEBUG PRINT] Array assigned to transmitter")
                    
                except Exception as array_error:
                    print(f"[DEBUG PRINT] Error in antenna array setup: {str(array_error)}")
                    logger.error(f"Antenna array configuration failed: {str(array_error)}")
                    raise
                print(f"[DEBUG PRINT] Antenna array setup took {time.time() - array_start:.4f} seconds")

                # Step 4: Register object
                print(f"[DEBUG PRINT] Step 4: Registering object in registry")
                reg_start = time.time()
                try:
                    print(f"[DEBUG PRINT] Calling _register_object() for {name}")
                    object_id = self._register_object(tx, ObjectType.TRANSMITTER)
                    tx.object_id = object_id
                    print(f"[DEBUG PRINT] Registration successful with ID {object_id}")
                except Exception as reg_error:
                    print(f"[DEBUG PRINT] Error in object registration: {str(reg_error)}")
                    logger.error(f"Object registration failed: {str(reg_error)}")
                    raise
                print(f"[DEBUG PRINT] Object registration took {time.time() - reg_start:.4f} seconds")

                # Step 5: Add to scene
                print(f"[DEBUG PRINT] Step 5: Adding to scene")
                scene_add_start = time.time()
                try:
                    self._scene.add(tx)
                    print(f"[DEBUG PRINT] Successfully added to scene")
                except Exception as add_error:
                    print(f"[DEBUG PRINT] Error adding to scene: {str(add_error)}")
                    logger.error(f"Scene addition failed: {str(add_error)}")
                    raise
                print(f"[DEBUG PRINT] Scene addition took {time.time() - scene_add_start:.4f} seconds")

                total_time = time.time() - start_time
                print(f"[DEBUG PRINT] Total transmitter setup time: {total_time:.4f} seconds")
                return tx

            except Exception as e:
                print(f"[DEBUG PRINT] Error in add_transmitter:")
                print(f"[DEBUG PRINT] - Error type: {type(e)}")
                print(f"[DEBUG PRINT] - Error message: {str(e)}")
                print("[DEBUG PRINT] - Stack trace:")
                import traceback
                traceback.print_exc()
                
                if 'object_id' in locals():
                    print(f"[DEBUG PRINT] Cleaning up object {object_id}")
                    self._unregister_object(object_id)
                
                logger.error(f"Failed to add transmitter {name}: {e}", exc_info=True)
                raise

            finally:
                print(f"[DEBUG PRINT] Exiting add_transmitter() after {time.time() - start_time:.4f} seconds")
                print(f"[DEBUG PRINT] Final lock state: {self._lock.locked()}")

    def add_ris(self, name: str, position: tf.Tensor, orientation: tf.Tensor,
                num_rows: int, num_cols: int, dtype=tf.complex64) -> RIS:
        """Add a RIS to the scene with enhanced debugging and error handling."""
        print(f"[DEBUG PRINT] Entering add_ris() for '{name}'")

        # Input validation first
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid RIS name")
        if not isinstance(num_rows, int) or not isinstance(num_cols, int):
            raise ValueError("num_rows and num_cols must be integers")
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("num_rows and num_cols must be positive")

        with self._lock:
            print(f"[DEBUG PRINT] Lock acquired for add_ris() - {name}")
            object_id = None
            ris = None

            try:
                # Step 1: Material Creation/Retrieval and Validation
                print(f"[DEBUG PRINT] Creating/retrieving material for RIS '{name}'")
                metal_material = self._get_or_create_material("itu_metal", dtype)
                if not metal_material:
                    raise RuntimeError("Failed to create/retrieve ITU metal material")
                print(f"[DEBUG PRINT] Material '{metal_material.name}' ready for use")

                # Step 2: RIS Object Creation
                print(f"[DEBUG PRINT] Creating RIS object '{name}'")
                ris = RIS(
                    name=name,
                    position=position,
                    orientation=orientation,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    dtype=dtype
                )
                print(f"[DEBUG PRINT] RIS object '{name}' created at {hex(id(ris))}")

                # Step 3: Scene Reference Assignment
                print(f"[DEBUG PRINT] Setting scene reference for '{name}'")
                ris.scene = self._scene
                print(f"[DEBUG PRINT] Scene reference set for '{name}'")

                # Step 4: Material Assignment
                print(f"[DEBUG PRINT] Assigning material '{metal_material.name}' to '{name}'")
                ris.radio_material = metal_material

                # Step 5: Object Registration
                print(f"[DEBUG PRINT] Registering RIS '{name}'")
                object_id = self._register_object(ris, ObjectType.RIS, "itu_metal")
                ris.object_id = object_id
                print(f"[DEBUG PRINT] RIS '{name}' registered with ID {object_id}")

                # Ensure material registry is properly updated
                if "itu_metal" not in self._material_registry:
                    self._material_registry["itu_metal"] = set()
                self._material_registry["itu_metal"].add(object_id)
                print(f"[DEBUG PRINT] Added RIS (ID: {object_id}) to itu_metal material registry")
                print(f"[DEBUG PRINT] Current itu_metal registry: {self._material_registry['itu_metal']}")

                # Step 6: Phase Profile Configuration
                print(f"[DEBUG PRINT] Configuring phase profile for '{name}'")
                try:
                    cell_grid = CellGrid(num_rows=num_rows, num_cols=num_cols, dtype=dtype)
                    phase_profile = DiscretePhaseProfile(cell_grid=cell_grid, dtype=dtype)
                    ris.phase_profile = phase_profile
                    print("[DEBUG PRINT] Phase profile created and assigned successfully")
                except Exception as e:
                    print(f"[DEBUG PRINT] Error creating phase profile: {str(e)}")
                    raise RuntimeError(f"Failed to configure phase profile for '{name}': {str(e)}") from e

                # Step 7: RIS Configuration Validation
                print(f"[DEBUG PRINT] Validating configuration for RIS '{name}'")
                self._validate_ris_configuration(ris)
                print(f"[DEBUG PRINT] RIS configuration for '{name}' validated successfully")

                # Step 8: Scene Addition
                print(f"[DEBUG PRINT] Adding RIS '{name}' to scene")
                try:
                    self._scene.add(ris)
                    print(f"[DEBUG PRINT] RIS '{name}' added to scene successfully")
                    logger.info(f"RIS '{name}' added successfully with ID {object_id}")
                except Exception as e:
                    print(f"[DEBUG PRINT] Fallback: Revalidating and retrying addition for RIS '{name}'. Error: {e}")
                    if object_id not in self._material_registry["itu_metal"]:
                        self._material_registry["itu_metal"].add(object_id)
                    self._scene.add(ris)  # Retry
                    print(f"[DEBUG PRINT] Retry successful: RIS '{name}' added to scene")

                return ris

            except Exception as e:
                error_msg = f"Failed to add RIS '{name}': {str(e)}"
                logger.error(error_msg)
                print(f"[DEBUG PRINT] Error: {error_msg}")

                # Cleanup on failure
                if object_id is not None:
                    try:
                        print(f"[DEBUG PRINT] Cleaning up - unregistering object ID {object_id}")
                        if "itu_metal" in self._material_registry:
                            self._material_registry["itu_metal"].discard(object_id)
                        self._unregister_object(object_id)
                        print(f"[DEBUG PRINT] Object ID {object_id} unregistered successfully")
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup failed: {cleanup_error}")

                raise RuntimeError(error_msg) from e


    def _validate_ris_configuration(self, ris: RIS) -> None:
        """Validate RIS configuration with enhanced checks"""
        try:
            # Check radio material
            if not hasattr(ris, 'radio_material') or ris.radio_material is None:
                raise ValueError("RIS material not properly assigned")
                
            # Check phase profile
            if not hasattr(ris, 'phase_profile') or ris.phase_profile is None:
                raise ValueError("RIS phase profile not configured")
                
            # Verify phase profile size matches RIS dimensions
            expected_size = ris.num_rows * ris.num_cols
            if hasattr(ris.phase_profile, 'size'):
                actual_size = ris.phase_profile.size
                if actual_size != expected_size:
                    raise ValueError(f"Phase profile size mismatch. Expected {expected_size}, got {actual_size}")
                    
            # Check scene reference
            if not hasattr(ris, 'scene') or ris.scene is None:
                raise ValueError("RIS scene reference not set")
                
            print(f"[DEBUG PRINT] RIS configuration validated successfully")
            
        except Exception as e:
            print(f"[DEBUG PRINT] RIS validation failed: {str(e)}")
            raise ValueError(f"RIS configuration validation failed: {str(e)}") from e

    def add_receiver(self, name: str, position: tf.Tensor,
                    orientation: tf.Tensor, dtype=tf.complex64) -> Receiver:
        """Add a receiver to the scene"""
        print(f"[DEBUG PRINT] Entering add_receiver() for '{name}'")
        print(f"[DEBUG PRINT] About to acquire lock in add_receiver() for '{name}'")
        
        with self._lock:
            print(f"[DEBUG PRINT] Lock acquired for add_receiver() - {name}")
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
                
                # Register object without lock since we're already in locked section
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
                if 'object_id' in locals():
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
            print(f"[DEBUG PRINT] Current registry size: {len(self._object_registry)}")
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