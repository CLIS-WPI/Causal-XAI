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
import time

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

    def _get_or_create_material(self, material_name: str, dtype=tf.complex64) -> RadioMaterial:
        """Get existing material or create new one with detailed error handling."""
        print(f"[DEBUG PRINT] Attempting to get/create material: {material_name}")
        print(f"[DEBUG PRINT] Using dtype: {dtype}")
        
        with self._lock:
            try:
                # First try to get existing material
                print(f"[DEBUG PRINT] Checking for existing material: {material_name}")
                material = next((mat for mat in self._scene.materials.values() if mat.name == material_name), None)
                
                if material is not None:
                    print(f"[DEBUG PRINT] Found existing material: {material_name}")
                    print(f"[DEBUG PRINT] Material memory address: {hex(id(material))}")
                    return material
                
                # Create new material if not found
                print(f"[DEBUG PRINT] Creating new material: {material_name}")
                material = RadioMaterial(
                    name=material_name,
                    relative_permittivity=4.5,  # Default value for metal
                    conductivity=0.01,         # Default value for metal
                    dtype=dtype
                )
                
                # Set scene reference
                print(f"[DEBUG PRINT] Setting scene reference for material: {material_name}")
                material.scene = self._scene
                
                # Add to scene
                print(f"[DEBUG PRINT] Adding material to scene: {material_name}")
                self._scene.add(material)
                
                # Initialize empty set in registry if needed
                if material_name not in self._material_registry:
                    print(f"[DEBUG PRINT] Initializing material registry entry for: {material_name}")
                    self._material_registry[material_name] = set()
                
                print(f"[DEBUG PRINT] Successfully created material: {material_name}")
                print(f"[DEBUG PRINT] New material memory address: {hex(id(material))}")
                return material
                
            except Exception as e:
                print(f"[DEBUG PRINT] Error in get_or_create_material: {str(e)}")
                print(f"[DEBUG PRINT] Material name: {material_name}")
                print(f"[DEBUG PRINT] Error type: {type(e)}")
                print("[DEBUG PRINT] Stack trace:")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to get/create material {material_name}: {str(e)}") from e
                    
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
        """Register a new object in the scene with detailed timing, registry validation, deadlock prevention, and debug."""
        print("[DEBUG PRINT] Entering _register_object()")
        print(f"[DEBUG PRINT] Input parameters: obj_type={obj_type}, radio_material={radio_material}")
        print(f"[DEBUG PRINT] Object details: type={type(obj)}, memory_addr={hex(id(obj))}")
        start_time = time.time()

        # Debug lock state
        lock_owner_id = id(threading.current_thread())
        print(f"[DEBUG PRINT] Current thread ID: {lock_owner_id}")
        print(f"[DEBUG PRINT] Lock state: {self._lock.locked()}")
        print(f"[DEBUG PRINT] Current registry state:")
        print(f"[DEBUG PRINT] - Registry size: {len(self._object_registry)}")
        print(f"[DEBUG PRINT] - Existing IDs: {list(self._object_registry.keys())}")

        try:
            # Step 1: Generate Object ID
            print("[DEBUG PRINT] Step 1: Generating object ID - Start")
            id_start = time.time()
            object_id = self._next_id
            self._next_id += 1
            id_duration = time.time() - id_start
            print(f"[DEBUG PRINT] Generated object_id={object_id} in {id_duration:.4f} seconds")

            # Step 2: Registry Validation Check
            print("[DEBUG PRINT] Step 2: Validating object ID in the registry...")
            if object_id in self._object_registry:
                raise RuntimeError(f"Duplicate object_id detected: {object_id}")
            print("[DEBUG PRINT] Object ID validation passed")

            # Step 3: Determine Object Name
            print("[DEBUG PRINT] Step 3: Determining object name")
            name_start = time.time()
            try:
                print(f"[DEBUG PRINT] Object hasattr('name'): {hasattr(obj, 'name')}")
                if hasattr(obj, 'name'):
                    print(f"[DEBUG PRINT] Object name attribute value: {obj.name}")
                name = obj.name if hasattr(obj, 'name') else f"object_{object_id}"
                print(f"[DEBUG PRINT] Final determined name: {name}")
            except Exception as name_error:
                print(f"[DEBUG PRINT] Error getting object name: {str(name_error)}")
                raise
            name_duration = time.time() - name_start
            print(f"[DEBUG PRINT] Name determination completed in {name_duration:.4f} seconds")

            # Step 4: Create SceneObject
            print(f"[DEBUG PRINT] Step 4: Creating SceneObject")
            create_start = time.time()
            scene_object = SceneObject(
                id=object_id,
                name=name,
                obj_type=obj_type,
                radio_material=radio_material,
                reference=obj
            )
            create_duration = time.time() - create_start
            print(f"[DEBUG PRINT] SceneObject created at {hex(id(scene_object))}")
            print(f"[DEBUG PRINT] Creation took {create_duration:.4f} seconds")

            # Step 5: Timeout Check
            elapsed_time = time.time() - start_time
            print(f"[DEBUG PRINT] Current elapsed time: {elapsed_time:.4f} seconds")
            if elapsed_time > 5:
                print(f"[DEBUG PRINT] Operation timed out after {elapsed_time:.4f} seconds")
                raise TimeoutError(f"Timeout while registering object: {name}")

            # Step 6: Add to Object Registry
            print(f"[DEBUG PRINT] Step 6: Adding to registry")
            print(f"[DEBUG PRINT] Registry state before addition:")
            print(f"[DEBUG PRINT] - Current size: {len(self._object_registry)}")
            print(f"[DEBUG PRINT] - Current keys: {list(self._object_registry.keys())}")
            reg_start = time.time()
            self._object_registry[object_id] = scene_object
            reg_duration = time.time() - reg_start
            print(f"[DEBUG PRINT] Registry state after addition:")
            print(f"[DEBUG PRINT] - New size: {len(self._object_registry)}")
            print(f"[DEBUG PRINT] - New keys: {list(self._object_registry.keys())}")
            print(f"[DEBUG PRINT] - Addition took {reg_duration:.4f} seconds")

            total_duration = time.time() - start_time
            print(f"[DEBUG PRINT] Total registration time: {total_duration:.4f} seconds")
            print("[DEBUG PRINT] Registration completed successfully")
            return object_id

        except TimeoutError as e:
            print(f"[DEBUG PRINT] Timeout Error occurred: {str(e)}")
            raise

        except Exception as e:
            print(f"[DEBUG PRINT] Error during registration:")
            print(f"[DEBUG PRINT] - Error type: {type(e)}")
            print(f"[DEBUG PRINT] - Error message: {str(e)}")
            print("[DEBUG PRINT] - Stack trace:")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Registration failed: {str(e)}") from e

        finally:
            elapsed = time.time() - start_time
            print(f"[DEBUG PRINT] _register_object() completed in {elapsed:.4f} seconds")



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
        print(f"[DEBUG PRINT] Attempting to acquire lock in add_ris() for '{name}'")

        with self._lock:
            print(f"[DEBUG PRINT] Lock acquired for add_ris() - {name}")
            try:
                # Create or retrieve the material
                logger.debug(f"Getting or creating material for RIS '{name}'...")
                try:
                    metal_material = self._get_or_create_material("itu_metal", dtype)
                    logger.debug(f"Material 'itu_metal' retrieved or created successfully for {name}")
                except Exception as material_error:
                    logger.error(f"Failed to get or create material for RIS '{name}': {material_error}")
                    raise

                # Create the RIS object
                logger.debug(f"Creating RIS '{name}' with {num_rows} rows and {num_cols} columns...")
                try:
                    ris = RIS(
                        name=name,
                        position=position,
                        orientation=orientation,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        dtype=dtype
                    )
                    logger.debug(f"RIS '{name}' object created successfully")
                    print(f"[DEBUG PRINT] RIS object '{name}' created successfully")
                except Exception as ris_creation_error:
                    logger.error(f"Failed to create RIS '{name}': {ris_creation_error}")
                    raise

                # Set scene reference before assigning material
                logger.debug(f"Setting scene reference for RIS '{name}'...")
                try:
                    ris.scene = self._scene
                    logger.debug(f"Scene reference set for RIS '{name}'")
                    print(f"[DEBUG PRINT] Scene reference assigned to RIS '{name}'")
                except Exception as scene_ref_error:
                    logger.error(f"Failed to set scene reference for RIS '{name}': {scene_ref_error}")
                    raise

                # Register the RIS object in the object registry
                logger.debug(f"Registering RIS '{name}' in the object registry...")
                try:
                    object_id = self._register_object(ris, ObjectType.RIS, "itu_metal")
                    ris.object_id = object_id
                    logger.debug(f"RIS '{name}' registered with object ID {object_id}")
                    print(f"[DEBUG PRINT] RIS '{name}' registered with object ID = {object_id}")
                except Exception as registry_error:
                    logger.error(f"Failed to register RIS '{name}': {registry_error}")
                    raise

                # Assign material after registration
                logger.debug(f"Assigning material to RIS '{name}'...")
                try:
                    ris.radio_material = metal_material
                    logger.debug(f"Material 'itu_metal' assigned to RIS '{name}'")
                except Exception as material_assignment_error:
                    logger.error(f"Failed to assign material to RIS '{name}': {material_assignment_error}")
                    raise

                # Add the RIS to the scene
                logger.debug(f"Adding RIS '{name}' to the scene...")
                try:
                    self._scene.add(ris)
                    logger.debug(f"RIS '{name}' added to the scene successfully")
                    print(f"[DEBUG PRINT] RIS '{name}' added to the scene")
                except Exception as scene_add_error:
                    logger.error(f"Failed to add RIS '{name}' to the scene: {scene_add_error}")
                    raise

                logger.info(f"Successfully added RIS '{name}' with ID {object_id}")
                print(f"[DEBUG PRINT] Exiting add_ris() for '{name}' with success")
                return ris

            except Exception as e:
                logger.error(f"Failed to add RIS '{name}': {e}")
                logger.error("Full error traceback:", exc_info=True)
                print(f"[DEBUG PRINT] ERROR in add_ris() for '{name}': {e}")
                if 'object_id' in locals():
                    logger.debug(f"Cleaning up - unregistering object ID {object_id}")
                    print(f"[DEBUG PRINT] Unregistering object ID {object_id}")
                    self._unregister_object(object_id)
                raise


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