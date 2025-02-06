import tensorflow as tf
import sionna
from sionna.rt import (
    load_scene, 
    Scene,
    PlanarArray, 
    Transmitter, 
    Receiver, 
    RIS, 
    RadioMaterial, 
    SceneObject
)
from sionna.constants import SPEED_OF_LIGHT

print(f"[DEBUG] Using Sionna version: {sionna.__version__}")

def validate_scene(scene):
    """Validate scene initialization"""
    if not hasattr(scene, '_dtype'):
        raise RuntimeError("Scene not properly initialized - use load_scene()")
    
    if scene._dtype not in (tf.complex64, tf.complex128):
        raise ValueError("Invalid dtype")

def init_empty_scene(dtype):
    """Initialize an empty scene with proper MI configuration"""
    print(f"[DEBUG] Initializing empty scene with dtype: {dtype}")
    scene = load_scene("__empty__", dtype=dtype)
    validate_scene(scene)
    print("[DEBUG] Scene initialized successfully")
    return scene

def setup_scene(config):
    """Set up the smart factory simulation scene with ray tracing configuration"""
    print("[DEBUG] Starting scene setup...")
    
    try:
        # 1. Initialize empty scene
        scene = load_scene("__empty__", dtype=config.dtype)
        
        # 2. Set basic properties
        scene.frequency = config.carrier_frequency
        wavelength = SPEED_OF_LIGHT/scene.frequency
        print(f"[DEBUG] Basic properties set. Frequency: {scene.frequency}, Wavelength: {wavelength}")
        
        # Initialize object ID counter
        current_object_id = 0
        
        # 3. Add materials with validation
        if not hasattr(config, 'materials') or not config.materials:
            raise ValueError("Materials configuration is missing or empty")
            
        for mat_name, mat_props in config.materials.items():
            material = RadioMaterial(
                name=mat_props['name'],
                relative_permittivity=mat_props['relative_permittivity'],
                conductivity=mat_props['conductivity'],
                scattering_coefficient=mat_props['scattering_coefficient'],
                xpd_coefficient=mat_props['xpd_coefficient']
            )
            scene.add(material)
        print("[DEBUG] Materials added successfully")

        # 4. Configure antenna arrays
        print("[DEBUG] Setting up antenna arrays...")
        scene.tx_array = PlanarArray(
            num_rows=config.bs_array[0],
            num_cols=config.bs_array[1], 
            vertical_spacing=config.bs_array_spacing,
            horizontal_spacing=config.bs_array_spacing,
            pattern=config.bs_array_pattern,
            polarization=config.bs_polarization,
            dtype=config.dtype
        )
        
        scene.rx_array = PlanarArray(
            num_rows=config.agv_array[0],
            num_cols=config.agv_array[1],
            vertical_spacing=config.agv_array_spacing,
            horizontal_spacing=config.agv_array_spacing,
            pattern=config.agv_array_pattern,
            polarization=config.agv_polarization,
            dtype=config.dtype
        )
        print("[DEBUG] Antenna arrays configured")
        
        # 5. Add radio devices with sequential object IDs
        print("[DEBUG] Adding radio devices...")
        # Add base station
        tx = Transmitter(
            name="bs",
            position=config.bs_position,
            orientation=config.bs_orientation
        )
        tx.object_id = current_object_id
        current_object_id += 1
        scene.add(tx)
        
        # Add AGVs
        for i in range(config.num_agvs):
            rx = Receiver(
                name=f"agv_{i}",
                position=[12.0 - i*4.0, 5.0 + i*10.0, config.agv_height],
                orientation=[0.0, 0.0, 0.0]
            )
            rx.object_id = current_object_id
            current_object_id += 1
            scene.add(rx)
        
        # Add RIS
        try:
            ris = RIS(
                name="ris",
                position=config.ris_position,
                orientation=config.ris_orientation,
                num_rows=config.ris_elements[0],
                num_cols=config.ris_elements[1],
                num_modes=config.ris_modes,
                dtype=config.dtype
            )
            ris.object_id = current_object_id
            current_object_id += 1
            scene.add(ris)
            print("[DEBUG] RIS added successfully")
        except Exception as e:
            print(f"[DEBUG] Error adding RIS: {str(e)}")
            raise
        
        # 6. Add static objects with sequential object IDs
        print("[DEBUG] Adding scene objects...")
        if config.static_scene['walls']:
            # Add floor
            floor = SceneObject(
                name="floor",
                position=[config.room_dim[0]/2, config.room_dim[1]/2, 0],
                orientation=[0.0, 0.0, 0.0]
            )
            floor.object_id = current_object_id
            current_object_id += 1
            scene.add(floor)
            floor.radio_material = config.static_scene['material']
            floor.size = [config.room_dim[0], config.room_dim[1], config.static_scene['wall_thickness']]
            
            # Add ceiling
            ceiling = SceneObject(
                name="ceiling",
                position=[config.room_dim[0]/2, config.room_dim[1]/2, config.room_dim[2]],
                orientation=[0.0, 0.0, 0.0]
            )
            ceiling.object_id = current_object_id
            current_object_id += 1
            scene.add(ceiling)
            ceiling.radio_material = config.static_scene['material']
            ceiling.size = [config.room_dim[0], config.room_dim[1], config.static_scene['wall_thickness']]
            
            # Add walls
            wall_specs = [
                ("wall_north", [config.room_dim[0], config.static_scene['wall_thickness'], config.room_dim[2]], 
                [config.room_dim[0]/2, config.room_dim[1], config.room_dim[2]/2]),
                ("wall_south", [config.room_dim[0], config.static_scene['wall_thickness'], config.room_dim[2]], 
                [config.room_dim[0]/2, 0, config.room_dim[2]/2]),
                ("wall_east", [config.static_scene['wall_thickness'], config.room_dim[1], config.room_dim[2]], 
                [config.room_dim[0], config.room_dim[1]/2, config.room_dim[2]/2]),
                ("wall_west", [config.static_scene['wall_thickness'], config.room_dim[1], config.room_dim[2]], 
                [0, config.room_dim[1]/2, config.room_dim[2]/2])
            ]
            
            for name, size, pos in wall_specs:
                wall = SceneObject(name=name, position=pos, orientation=[0.0, 0.0, 0.0])
                wall.object_id = current_object_id
                current_object_id += 1
                scene.add(wall)
                wall.radio_material = config.static_scene['material']
                wall.size = size
        
        # Add shelves
        for i, pos in enumerate(config.scene_objects['shelf_positions']):
            shelf = SceneObject(
                name=f"shelf_{i}",
                position=pos,
                orientation=[0.0, 0.0, 0.0]
            )
            shelf.object_id = current_object_id
            current_object_id += 1
            scene.add(shelf)
            shelf.radio_material = config.scene_objects['shelf_material']
            shelf.size = config.scene_objects['shelf_dimensions']
        
        print("[DEBUG] Scene setup completed successfully")
        return scene
        
    except Exception as e:
        print(f"[DEBUG] Error in scene setup: {type(e).__name__}: {str(e)}")
        raise