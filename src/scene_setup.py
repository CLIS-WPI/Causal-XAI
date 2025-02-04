# scene_setup.py
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RIS, RadioMaterial

def setup_scene(config):
    """Set up the smart factory simulation scene"""
    # Create empty scene
    scene = load_scene("__empty__")
    scene.frequency = 28e9  # Set to 28 GHz
    
    # Configure BS antenna array (16x4 UPA at 28 GHz)
    scene.tx_array = PlanarArray(
        num_rows=16,
        num_cols=4,
        vertical_spacing=0.5*3e8/28e9,  # Half wavelength at 28 GHz
        horizontal_spacing=0.5*3e8/28e9,
        pattern="tr38901",  
        polarization="VH"   
    )
    
    # Configure AGV antenna array (1x1)
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5*3e8/28e9,
        horizontal_spacing=0.5*3e8/28e9,
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
    
    # Add RIS
    ris = RIS(
        name="ris",
        position=[10.0, 19.5, 2.5],  
        orientation=[0.0, 0.0, 0.0],
        num_rows=8,
        num_cols=8
    )
    scene.add(ris)
    
    # First create a metal material for the shelves
    from sionna.rt import RadioMaterial
    metal_material = RadioMaterial(
        name="shelf_metal",
        relative_permittivity=1.0,
        conductivity=1e7  # High conductivity for metal
    )
    scene.add(metal_material)
    
    # Define shelf dimensions
    shelf_dimensions = {
        'length': 2.0,  # x-dimension in meters
        'width': 1.0,   # y-dimension in meters
        'height': 3.0   # z-dimension in meters
    }
    
    # Add metallic shelves with fixed positions
    shelf_positions = [
        [5.0, 5.0, shelf_dimensions['height']/2],    # Adjust z-position to half height
        [15.0, 5.0, shelf_dimensions['height']/2],
        [10.0, 10.0, shelf_dimensions['height']/2],
        [5.0, 15.0, shelf_dimensions['height']/2],
        [15.0, 15.0, shelf_dimensions['height']/2]
    ]

        # Create and add shelves using Transmitter objects
    for i, position in enumerate(shelf_positions):
        # Create a metal box for each shelf using a Transmitter as a placeholder
        shelf = Transmitter(
            name=f"shelf_{i}",
            position=position,
            orientation=[0.0, 0.0, 0.0]
        )
        
        # Add the shelf to the scene once
        scene.add(shelf)
        
        # Set the radio material
        shelf.radio_material = "shelf_metal"
        
        # Set physical dimensions for the shelf
        shelf.size = [
            shelf_dimensions['length'],
            shelf_dimensions['width'], 
            shelf_dimensions['height']
        ]

        
        # Add shelf to scene
        scene.add(shelf)
    
    # Add initial AGV positions
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
    
    return scene