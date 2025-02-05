import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial, SceneObject
from sionna.constants import SPEED_OF_LIGHT

def setup_scene(config):
    """Set up the smart factory simulation scene with ray tracing configuration"""
    # 1. Validate dtype
    if config.dtype not in (tf.complex64, tf.complex128):
        raise ValueError("dtype must be tf.complex64 or tf.complex128")
    
    # 2. Load scene with proper dtype initialization
    scene = load_scene("__empty__", dtype=config.dtype)
    
    # 3. Set frequency
    scene.frequency = config.carrier_frequency
    
    # 4. Calculate wavelength
    wavelength = SPEED_OF_LIGHT/scene.frequency
    
    # 5. Configure antenna arrays
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
    
    # 6. Add materials first
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
    
    # 7. Add base station
    tx = Transmitter(
        name="bs",
        position=config.bs_position,
        orientation=config.bs_orientation
    )
    scene.add(tx)
    
    # 8. Add AGVs
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
    
    # 9. Add room objects (including boundaries)
    room_dims = config.room_dim
    
    # Floor
    floor = SceneObject(
        name="floor",
        position=[room_dims[0]/2, room_dims[1]/2, 0],
        size=[room_dims[0], room_dims[1], 0.2],
        radio_material=concrete
    )
    scene.add(floor)
    
    # Ceiling
    ceiling = SceneObject(
        name="ceiling",
        position=[room_dims[0]/2, room_dims[1]/2, room_dims[2]],
        size=[room_dims[0], room_dims[1], 0.2],
        radio_material=concrete
    )
    scene.add(ceiling)
    
    # Walls
    wall_specs = [
        ("wall_north", [room_dims[0], 0.2, room_dims[2]], [room_dims[0]/2, room_dims[1], room_dims[2]/2]),
        ("wall_south", [room_dims[0], 0.2, room_dims[2]], [room_dims[0]/2, 0, room_dims[2]/2]),
        ("wall_east", [0.2, room_dims[1], room_dims[2]], [room_dims[0], room_dims[1]/2, room_dims[2]/2]),
        ("wall_west", [0.2, room_dims[1], room_dims[2]], [0, room_dims[1]/2, room_dims[2]/2])
    ]
    
    for name, size, pos in wall_specs:
        wall = SceneObject(
            name=name,
            position=pos,
            size=size,
            radio_material=concrete
        )
        scene.add(wall)
    
    # 10. Add RIS
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
    
    # 11. Add shelves
    shelf_dimensions = config.scene_objects.get('shelf_dimensions', [2.0, 1.0, 3.0])
    shelf_positions = [
        [5.0, 5.0, shelf_dimensions[2]/2],
        [15.0, 5.0, shelf_dimensions[2]/2],
        [10.0, 10.0, shelf_dimensions[2]/2],
        [5.0, 15.0, shelf_dimensions[2]/2],
        [15.0, 15.0, shelf_dimensions[2]/2]
    ]
    
    for i, pos in enumerate(shelf_positions):
        shelf = SceneObject(
            name=f"shelf_{i}",
            position=pos,
            size=shelf_dimensions,
            radio_material=metal
        )
        scene.add(shelf)
    
    # 12. Configure RIS phase profile
    bs_position = tf.constant([config.bs_position], dtype=tf.float32)
    agv_positions = tf.constant(initial_positions, dtype=tf.float32)
    
    for agv_pos in agv_positions:
        ris.phase_gradient_reflector(
            sources=bs_position,
            targets=tf.expand_dims(agv_pos, axis=0)
        )
    
    return scene