# scene_setup.py
# scene_setup.py
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial, SceneObject
from sionna.constants import SPEED_OF_LIGHT

def setup_scene(config):
    """Set up the smart factory simulation scene"""
    # Create empty scene
    scene = load_scene("__empty__")
    scene.frequency = 28e9  # Set to 28 GHz
    
    # Calculate wavelength
    wavelength = SPEED_OF_LIGHT/scene.frequency
    
    # Configure BS antenna array (16x4 UPA at 28 GHz)
    scene.tx_array = PlanarArray(
        num_rows=16,
        num_cols=4,
        vertical_spacing=0.5*wavelength,
        horizontal_spacing=0.5*wavelength,
        pattern="tr38901",  
        polarization="VH"   
    )
    
    # Configure AGV antenna array (1x1)
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5*wavelength,
        horizontal_spacing=0.5*wavelength,
        pattern="iso",      
        polarization="V"    
    )
    
    # Create metal material for the shelves
    metal_material = RadioMaterial(
        name="metal",
        relative_permittivity=complex(1.0, -1e7),  # Changed to complex permittivity
        scattering_coefficient=0.1,
        xpd_coefficient=10.0
    )
    scene.add(metal_material)
    
    # Add base station with object_id=0
    tx = Transmitter(
        name="bs",
        position=[10.0, 0.5, 4.5],  
        orientation=[0.0, 0.0, 0.0],
        object_id=0
    )
    scene.add(tx)
    
    # Define initial AGV positions
    initial_positions = [
        [12.0, 5.0, 0.5],   # AGV1
        [8.0, 15.0, 0.5]    # AGV2
    ]
    
    # Add receivers (AGVs) with sequential object_ids
    for i, pos in enumerate(initial_positions):
        rx = Receiver(
            name=f"agv_{i}",
            position=pos,
            orientation=[0.0, 0.0, 0.0],
            object_id=i+1  # Start from 1
        )
        scene.add(rx)
    
    # Add RIS with object_id after AGVs
    ris = RIS(
        name="ris",
        position=[10.0, 19.5, 2.5],
        orientation=[0.0, 0.0, 0.0],
        num_rows=8,
        num_cols=8,
        element_spacing=0.5*wavelength,
        dtype=tf.complex64,
        object_id=len(initial_positions)+1
    )
    scene.add(ris)
    
    # Configure RIS phase profile
    bs_position = tf.constant([[10.0, 0.5, 4.5]], dtype=tf.float32)
    agv_position = tf.constant([[12.0, 5.0, 0.5]], dtype=tf.float32)
    ris.phase_gradient_reflector(sources=bs_position, targets=agv_position)
    
    # Add metallic shelves with fixed positions
    shelf_dimensions = [2.0, 1.0, 3.0]  # [length, width, height] in meters
    
    shelf_positions = [
        [5.0, 5.0, shelf_dimensions[2]/2],    
        [15.0, 5.0, shelf_dimensions[2]/2],
        [10.0, 10.0, shelf_dimensions[2]/2],
        [5.0, 15.0, shelf_dimensions[2]/2],
        [15.0, 15.0, shelf_dimensions[2]/2]
    ]

    # Create and add shelves with sequential object_ids
    start_shelf_id = len(initial_positions) + 2  # Start after BS, AGVs, and RIS
    for i, position in enumerate(shelf_positions):
        shelf = SceneObject(  # Changed from Transmitter to SceneObject
            name=f"shelf_{i}",
            position=position,
            orientation=[0.0, 0.0, 0.0],
            size=shelf_dimensions,
            material="metal",
            object_id=start_shelf_id + i
        )
        scene.add(shelf)
    
    # Verify scene configuration
    scene._check_scene(check_materials=True)
    
    return scene