import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial, SceneObject
from sionna.constants import SPEED_OF_LIGHT

def setup_scene(config):
    """Set up the smart factory simulation scene with ray tracing configuration"""
    # Create empty scene with ray tracing configuration
    scene = load_scene("__empty__")
    
    # First set the frequency - this is critical for scene initialization
    scene.frequency = config.carrier_frequency
    
    # Calculate wavelength
    wavelength = SPEED_OF_LIGHT/scene.frequency
    
    # Configure ray tracing parameters early
    scene.ray_tracing = {
        'max_depth': config.ray_tracing['max_depth'],
        'diffraction': config.ray_tracing['diffraction'],
        'scattering': config.ray_tracing['scattering']
    }
    
    # Create and add materials first
    metal_material = RadioMaterial(
        name="metal",
        relative_permittivity=complex(1.0, -1e7),
        scattering_coefficient=config.materials['metal_shelves'].get('scattering_coefficient', 0.1),
        xpd_coefficient=config.materials['metal_shelves'].get('xpd_coefficient', 10.0)
    )
    scene.add(metal_material)
    
    concrete_material = RadioMaterial(
        name="concrete",
        relative_permittivity=complex(config.materials['walls'].get('permittivity', 4.5)),
        conductivity=config.materials['walls'].get('conductivity', 0.01),
        scattering_coefficient=0.2,
        xpd_coefficient=8.0
    )
    scene.add(concrete_material)
    
    # Configure antenna arrays
    scene.tx_array = PlanarArray(
        num_rows=config.bs_array[0],
        num_cols=config.bs_array[1],
        vertical_spacing=config.bs_array_spacing or 0.5*wavelength,
        horizontal_spacing=config.bs_array_spacing or 0.5*wavelength,
        pattern="tr38901",
        polarization="VH"
    )
    
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=config.agv_array_spacing or 0.5*wavelength,
        horizontal_spacing=config.agv_array_spacing or 0.5*wavelength,
        pattern="iso",
        polarization="V"
    )
    
    # Add room boundaries first
    room_dims = config.room_dim
    walls = [
        SceneObject(
            name="floor",
            position=[room_dims[0]/2, room_dims[1]/2, 0],
            size=[room_dims[0], room_dims[1], 0.2],
            material="concrete"
        ),
        SceneObject(
            name="ceiling",
            position=[room_dims[0]/2, room_dims[1]/2, room_dims[2]],
            size=[room_dims[0], room_dims[1], 0.2],
            material="concrete"
        ),
        SceneObject(
            name="wall_north",
            position=[room_dims[0]/2, room_dims[1], room_dims[2]/2],
            size=[room_dims[0], 0.2, room_dims[2]],
            material="concrete"
        ),
        SceneObject(
            name="wall_south",
            position=[room_dims[0]/2, 0, room_dims[2]/2],
            size=[room_dims[0], 0.2, room_dims[2]],
            material="concrete"
        ),
        SceneObject(
            name="wall_east",
            position=[room_dims[0], room_dims[1]/2, room_dims[2]/2],
            size=[0.2, room_dims[1], room_dims[2]],
            material="concrete"
        ),
        SceneObject(
            name="wall_west",
            position=[0, room_dims[1]/2, room_dims[2]/2],
            size=[0.2, room_dims[1], room_dims[2]],
            material="concrete"
        )
    ]
    
    # Add walls to scene
    for wall in walls:
        scene.add(wall)
    
    # Add base station
    tx = Transmitter(
        name="bs",
        position=config.bs_position,
        orientation=config.bs_orientation
    )
    scene.add(tx)
    
    # Add AGVs
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
    
    # Add RIS
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
    
    # Add shelves last
    shelf_dimensions = config.scene_objects.get('shelf_dimensions', [2.0, 1.0, 3.0])
    shelf_positions = [
        [5.0, 5.0, shelf_dimensions[2]/2],
        [15.0, 5.0, shelf_dimensions[2]/2],
        [10.0, 10.0, shelf_dimensions[2]/2],
        [5.0, 15.0, shelf_dimensions[2]/2],
        [15.0, 15.0, shelf_dimensions[2]/2]
    ]
    
    for i, position in enumerate(shelf_positions):
        shelf = SceneObject(
            name=f"shelf_{i}",
            position=position,
            orientation=[0.0, 0.0, 0.0],
            size=shelf_dimensions,
            material="metal"
        )
        scene.add(shelf)
    
    # Configure RIS phase profile after all objects are added
    bs_position = tf.constant([config.bs_position], dtype=tf.float32)
    agv_positions = tf.constant(initial_positions, dtype=tf.float32)
    
    for agv_pos in agv_positions:
        ris.phase_gradient_reflector(
            sources=bs_position,
            targets=tf.expand_dims(agv_pos, axis=0)
        )
    
    # Final verification
    scene._check_scene(check_materials=True)
    
    return scene