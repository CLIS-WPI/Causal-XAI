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
#######################################
def init_empty_scene(dtype):
    """Initialize an empty scene with proper MI configuration"""
    print(f"[DEBUG] Initializing empty scene with dtype: {dtype}")
    
    # Initialize scene first
    scene = load_scene("__empty__", dtype=dtype)
    
    # Basic scene configuration is handled internally by Sionna
    # No need to manually configure Mitsuba
    
    # Verify initialization
    validate_scene(scene)
    print("[DEBUG] Scene initialized successfully")
    
    return scene
#######################################
def setup_scene(config):
    """Set up the smart factory simulation scene with ray tracing configuration"""
    print("[DEBUG] Starting scene setup...")
    
    try:
        # 1. Initialize empty scene properly using Sionna's API
        scene = load_scene("__empty__", dtype=config.dtype)
        
        # 2. Set basic properties
        scene.frequency = config.carrier_frequency
        wavelength = SPEED_OF_LIGHT/scene.frequency
        print(f"[DEBUG] Basic properties set. Frequency: {scene.frequency}, Wavelength: {wavelength}")
        
        # 3. Add materials FIRST before any objects
        metal = RadioMaterial(
            name="metal",
            relative_permittivity=complex(1.0, -1e7),
            conductivity=1e7,
            scattering_coefficient=0.1,
            xpd_coefficient=10.0
        )
        scene.add(metal)
        
        concrete = RadioMaterial(
            name="concrete",
            relative_permittivity=complex(4.5),
            conductivity=0.01,
            scattering_coefficient=0.2,
            xpd_coefficient=8.0
        )
        scene.add(concrete)
        print("[DEBUG] Materials added successfully")

        # 4. Configure antenna arrays (unchanged)
        print("[DEBUG] Setting up antenna arrays...")
        scene.tx_array = PlanarArray(
            num_rows=config.bs_array[0],
            num_cols=config.bs_array[1], 
            vertical_spacing=config.bs_array_spacing or 0.5*wavelength,
            horizontal_spacing=config.bs_array_spacing or 0.5*wavelength,
            pattern="tr38901",
            polarization="VH",
            dtype=config.dtype
        )
        
        scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=config.agv_array_spacing or 0.5*wavelength,
            horizontal_spacing=config.agv_array_spacing or 0.5*wavelength,
            pattern="iso",
            polarization="V",
            dtype=config.dtype
        )
        print("[DEBUG] Antenna arrays configured")
        
        # 5. Add radio devices (unchanged)
        print("[DEBUG] Adding radio devices...")
        tx = Transmitter(
            name="bs",
            position=config.bs_position,
            orientation=config.bs_orientation
        )
        scene.add(tx)
        
        initial_positions = [
            [12.0, 5.0, config.agv_height],
            [8.0, 15.0, config.agv_height]
        ]
        
        for i, pos in enumerate(initial_positions):
            rx = Receiver(
                name=f"agv_{i}",
                position=pos,
                orientation=[0.0, 0.0, 0.0]
            )
            scene.add(rx)
        
        ris = RIS(
            name="ris",
            position=config.ris_position,
            orientation=config.ris_orientation,
            num_rows=config.ris_elements[0],
            num_cols=config.ris_elements[1],
            num_modes=1,
            dtype=config.dtype
        )
        scene.add(ris)
        print("[DEBUG] Radio devices added")
        
        # 6. Add static objects as Transmitters instead of SceneObjects
        print("[DEBUG] Adding scene objects...")
        room_dims = config.room_dim
        
        # Add floor
        floor = Transmitter(
            name="floor",
            position=[room_dims[0]/2, room_dims[1]/2, 0],
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(floor)
        floor.radio_material = "concrete"
        floor.size = [room_dims[0], room_dims[1], 0.2]
        print("[DEBUG] Floor added")
        
        # Add ceiling
        ceiling = Transmitter(
            name="ceiling",
            position=[room_dims[0]/2, room_dims[1]/2, room_dims[2]],
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(ceiling)
        ceiling.radio_material = "concrete"
        ceiling.size = [room_dims[0], room_dims[1], 0.2]
        print("[DEBUG] Ceiling added")
        
        # Add walls
        wall_specs = [
            ("wall_north", [room_dims[0], 0.2, room_dims[2]], [room_dims[0]/2, room_dims[1], room_dims[2]/2]),
            ("wall_south", [room_dims[0], 0.2, room_dims[2]], [room_dims[0]/2, 0, room_dims[2]/2]),
            ("wall_east", [0.2, room_dims[1], room_dims[2]], [room_dims[0], room_dims[1]/2, room_dims[2]/2]),
            ("wall_west", [0.2, room_dims[1], room_dims[2]], [0, room_dims[1]/2, room_dims[2]/2])
        ]
        
        for name, size, pos in wall_specs:
            wall = Transmitter(
                name=name,
                position=pos,
                orientation=[0.0, 0.0, 0.0]
            )
            scene.add(wall)
            wall.radio_material = "concrete"
            wall.size = size
        print("[DEBUG] Walls added")
        
        # Add shelves
        shelf_dimensions = config.scene_objects.get('shelf_dimensions', [2.0, 1.0, 3.0])
        shelf_positions = [
            [5.0, 5.0, shelf_dimensions[2]/2],
            [15.0, 5.0, shelf_dimensions[2]/2],
            [10.0, 10.0, shelf_dimensions[2]/2],
            [5.0, 15.0, shelf_dimensions[2]/2],
            [15.0, 15.0, shelf_dimensions[2]/2]
        ]
        
        for i, pos in enumerate(shelf_positions):
            shelf = Transmitter(
                name=f"shelf_{i}",
                position=pos,
                orientation=[0.0, 0.0, 0.0]
            )
            scene.add(shelf)
            shelf.radio_material = "metal"
            shelf.size = shelf_dimensions
        print("[DEBUG] Shelves added")
        
        print("[DEBUG] Scene setup completed successfully")
        return scene
        
    except Exception as e:
        print(f"[DEBUG] Error in scene setup: {type(e).__name__}: {str(e)}")
        raise