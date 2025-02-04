# scene_setup.py
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial
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
    
    # Add base station
    tx = Transmitter(
        name="bs",
        position=[10.0, 0.5, 4.5],  
        orientation=[0.0, 0.0, 0.0]
    )
    scene.add(tx)
    
    # Create metal material for the shelves
    metal_material = RadioMaterial(
        name="metal",
        relative_permittivity=1.0,
        conductivity=1e7  # High conductivity for metal
    )
    scene.add(metal_material)
    
    # Define initial AGV positions
    initial_positions = [
        [12.0, 5.0, 0.5],   # AGV1
        [8.0, 15.0, 0.5]    # AGV2
    ]
    
    # Add receivers (AGVs)
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
        position=[10.0, 19.5, 2.5],
        orientation=[0.0, 0.0, 0.0],
        num_rows=8,
        num_cols=8,
        vertical_spacing=0.5*wavelength,    # Changed from element_spacing
        horizontal_spacing=0.5*wavelength,  # Added horizontal spacing
        dtype=tf.complex64
    )

    scene.add(ris)
    
    # Configure RIS phase profile to reflect signals from BS to first AGV
    # Using only first AGV position to match expected shape [1, 3]
    bs_position = tf.constant([[10.0, 0.5, 4.5]], dtype=tf.float32)  # Shape [1, 3]
    agv_position = tf.constant([[12.0, 5.0, 0.5]], dtype=tf.float32)  # Shape [1, 3]
    
    # Configure RIS phase profile
    ris.phase_gradient_reflector(sources=bs_position, targets=agv_position)
    
    # Add metallic shelves with fixed positions
    shelf_dimensions = {
        'length': 2.0,  # x-dimension in meters
        'width': 1.0,   # y-dimension in meters
        'height': 3.0   # z-dimension in meters
    }
    
    shelf_positions = [
        [5.0, 5.0, shelf_dimensions['height']/2],    
        [15.0, 5.0, shelf_dimensions['height']/2],
        [10.0, 10.0, shelf_dimensions['height']/2],
        [5.0, 15.0, shelf_dimensions['height']/2],
        [15.0, 15.0, shelf_dimensions['height']/2]
    ]

    # Create and add shelves
    for i, position in enumerate(shelf_positions):
        shelf = Transmitter(
            name=f"shelf_{i}",
            position=position,
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(shelf)
        shelf.radio_material = "metal"
        shelf.size = [
            shelf_dimensions['length'],
            shelf_dimensions['width'], 
            shelf_dimensions['height']
        ]
    
    return scene