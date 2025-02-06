import os
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
########
def setup_scene(config, ply_dir='meshes'):
    """
    Setup scene with PLY-based geometries
    
    Args:
        config: Configuration object
        ply_dir: Directory containing PLY files
    
    Returns:
        Configured Sionna scene object
    """
    try:
        # Initialize empty scene
        scene = load_scene("__empty__", dtype=config.dtype)
        scene.frequency = config.carrier_frequency

        # Configure antenna arrays (existing method)
        scene.tx_array = PlanarArray(
            num_rows=config.bs_array[0],
            num_cols=config.bs_array[1],
            vertical_spacing=config.bs_array_spacing,
            horizontal_spacing=config.bs_array_spacing,
            pattern="tr38901",
            polarization="VH",
            dtype=config.dtype
        )

        scene.rx_array = PlanarArray(
            num_rows=config.agv_array[0],
            num_cols=config.agv_array[1],
            vertical_spacing=config.agv_array_spacing,
            horizontal_spacing=config.agv_array_spacing,
            pattern="iso",
            polarization="V",
            dtype=config.dtype
        )

        # Create radio materials
        metal_material = RadioMaterial(
            name="metal",
            relative_permittivity=1.0,
            conductivity=1e7,
            scattering_coefficient=0.1,
            xpd_coefficient=0.0
        )
        scene.add(metal_material)

        concrete_material = RadioMaterial(
            name="concrete",
            relative_permittivity=4.0,
            conductivity=0.01,
            scattering_coefficient=0.2,
            xpd_coefficient=0.5
        )
        scene.add(concrete_material)

        # Add room boundary objects from PLY files
        room_boundaries = [
            ('floor', 'floor.ply', concrete_material),
            ('ceiling', 'ceiling.ply', concrete_material),
            ('wall_xp', 'wall_xp.ply', concrete_material),
            ('wall_xm', 'wall_xm.ply', concrete_material),
            ('wall_yp', 'wall_yp.ply', concrete_material),
            ('wall_ym', 'wall_ym.ply', concrete_material)
        ]

        for name, ply_file, material in room_boundaries:
            try:
                room_object = SceneObject(
                    name=f"{name}-concrete",
                    filename=os.path.join(ply_dir, ply_file)
                )
                room_object.radio_material = material
                scene.add(room_object)
                print(f"[SUCCESS] Room object {name} added from {ply_file}")
            except Exception as e:
                print(f"[WARNING] Failed to add room object {name}: {str(e)}")

        # Add shelves from PLY files
        for i, pos in enumerate(config.scene_objects['shelf_positions']):
            try:
                shelf_object = SceneObject(
                    name=f"shelf_{i}-metal",
                    filename=os.path.join(ply_dir, f'shelf_{i}.ply')
                )
                shelf_object.radio_material = metal_material
                scene.add(shelf_object)
                print(f"[SUCCESS] Shelf {i} added from shelf_{i}.ply")
            except Exception as e:
                print(f"[WARNING] Failed to add shelf {i}: {str(e)}")

        # Add base station, AGVs, and RIS (existing method)
        tx = Transmitter(
            name="bs",
            position=config.bs_position,
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(tx)

        for i in range(config.num_agvs):
            position = [12.0 - i*4.0, 5.0 + i*10.0, 0.5]
            rx = Receiver(
                name=f"agv_{i}",
                position=position,
                orientation=[0.0, 0.0, 0.0]
            )
            scene.add(rx)

        ris = RIS(
            name="ris",
            position=config.ris_position,
            num_rows=config.ris_elements[0],
            num_cols=config.ris_elements[1],
            num_modes=config.ris_modes,
            orientation=config.ris_orientation,
            dtype=config.dtype
        )
        scene.add(ris)

        return scene

    except Exception as e:
        print(f"[CRITICAL ERROR] Scene setup failed: {str(e)}")
        raise RuntimeError(f"Scene setup error: {str(e)}") from e
########
def _validate_config(config):
    """
    Validate critical configuration parameters
    
    Args:
        config: Configuration object to validate
    
    Raises:
        ValueError: If critical parameters are missing or invalid
    """
    required_attrs = [
        'room_dim', 'bs_position', 'ris_position', 
        'bs_array', 'agv_array', 'num_agvs',
        'carrier_frequency', 'dtype'
    ]
    
    missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
    
    if missing_attrs:
        raise ValueError(f"Missing required configuration attributes: {missing_attrs}")

def _initialize_scene(config):
    """
    Initialize empty Sionna scene
    
    Args:
        config: Configuration object
    
    Returns:
        Initialized Sionna scene
    """
    print("[DEBUG] Initializing scene...")
    
    scene = load_scene("__empty__", dtype=config.dtype)
    scene.frequency = config.carrier_frequency
    
    print(f"[DEBUG] Scene initialized with frequency: {scene.frequency} Hz")
    return scene

def _configure_antenna_arrays(scene, config):
    """
    Configure Base Station and Mobile Robot antenna arrays
    
    Args:
        scene: Sionna scene object
        config: Configuration object
    """
    print("[DEBUG] Configuring antenna arrays...")
    
    # Base Station Antenna Array
    scene.tx_array = PlanarArray(
        num_rows=config.bs_array[0],
        num_cols=config.bs_array[1],
        vertical_spacing=config.bs_array_spacing,
        horizontal_spacing=config.bs_array_spacing,
        pattern="tr38901",
        polarization="VH",
        dtype=config.dtype
    )
    
    # Mobile Robot (AGV) Antenna Array
    scene.rx_array = PlanarArray(
        num_rows=config.agv_array[0],
        num_cols=config.agv_array[1],
        vertical_spacing=config.agv_array_spacing,
        horizontal_spacing=config.agv_array_spacing,
        pattern="iso",
        polarization="V",
        dtype=config.dtype
    )
    
    print("[SUCCESS] Antenna arrays configured")

def _add_base_station(scene, config):
    """
    Add base station transmitter to the scene
    
    Args:
        scene: Sionna scene object
        config: Configuration object
    """
    print("[DEBUG] Adding base station...")
    
    tx = Transmitter(
        name="bs",
        position=config.bs_position,
        orientation=[0.0, 0.0, 0.0]
    )
    scene.add(tx)
    
    print(f"[SUCCESS] Base station added at {config.bs_position}")

def _add_mobile_robots(scene, config):
    """
    Add mobile robots (AGVs) to the scene
    
    Args:
        scene: Sionna scene object
        config: Configuration object
    """
    print("[DEBUG] Adding mobile robots...")
    
    for i in range(config.num_agvs):
        # Predefined initial positions for AGVs
        position = [12.0 - i*4.0, 5.0 + i*10.0, 0.5]
        
        rx = Receiver(
            name=f"agv_{i}",
            position=position,
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(rx)
        
        print(f"[SUCCESS] Mobile robot {i} added at {position}")

def _add_ris(scene, config):
    """
    Add Reconfigurable Intelligent Surface to the scene
    
    Args:
        scene: Sionna scene object
        config: Configuration object
    """
    print("[DEBUG] Adding RIS...")
    
    ris = RIS(
        name="ris",
        position=config.ris_position,
        num_rows=config.ris_elements[0],
        num_cols=config.ris_elements[1],
        num_modes=config.ris_modes,
        orientation=config.ris_orientation,
        dtype=config.dtype
    )
    scene.add(ris)
    
    print(f"[SUCCESS] RIS added at {config.ris_position}")

def _create_radio_materials(scene):
    """
    Create radio materials for scene objects
    
    Args:
        scene: Sionna scene object
    
    Returns:
        Tuple of metal and concrete radio materials
    """
    print("[DEBUG] Creating radio materials...")
    
    # Metal material for shelves
    metal_material = RadioMaterial(
        name="metal",
        relative_permittivity=1.0,
        conductivity=1e7,
        scattering_coefficient=0.1,
        xpd_coefficient=0.0
    )
    scene.add(metal_material)
    
    # Concrete material for walls and floor
    concrete_material = RadioMaterial(
        name="concrete",
        relative_permittivity=4.0,
        conductivity=0.01,
        scattering_coefficient=0.2,
        xpd_coefficient=0.5
    )
    scene.add(concrete_material)
    
    print("[SUCCESS] Radio materials created")
    return metal_material, concrete_material

def _add_scene_objects_from_ply(scene, config, metal_material, concrete_material):
    """
    Add scene objects using PLY files
    
    Args:
        scene: Sionna scene object
        config: Configuration object
        metal_material: Metal radio material
        concrete_material: Concrete radio material
    """
    print("[DEBUG] Adding scene objects from PLY files...")
    
    # Define PLY files for room boundaries
    room_boundary_plys = [
        ('floor', 'floor.ply', concrete_material),
        ('ceiling', 'ceiling.ply', concrete_material),
        ('wall_xp', 'wall_xp.ply', concrete_material),
        ('wall_xm', 'wall_xm.ply', concrete_material),
        ('wall_yp', 'wall_yp.ply', concrete_material),
        ('wall_ym', 'wall_ym.ply', concrete_material)
    ]
    
    # Add room boundary objects
    for name, ply_file, material in room_boundary_plys:
        try:
            room_object = SceneObject(
                name=f"{name}-concrete",
                filename=os.path.join('meshes', ply_file)
            )
            room_object.radio_material = material
            scene.add(room_object)
            print(f"[SUCCESS] Room object {name} added from {ply_file}")
        except Exception as e:
            print(f"[WARNING] Failed to add room object {name}: {str(e)}")
    
    # Add shelves
    for i, pos in enumerate(config.scene_objects['shelf_positions']):
        try:
            shelf_object = SceneObject(
                name=f"shelf_{i}-metal",
                filename=os.path.join('meshes', f'shelf_{i}.ply')
            )
            shelf_object.radio_material = metal_material
            scene.add(shelf_object)
            print(f"[SUCCESS] Shelf {i} added from shelf_{i}.ply")
        except Exception as e:
            print(f"[WARNING] Failed to add shelf {i}: {str(e)}")
    
    print("[SUCCESS] Scene objects setup completed")

# Ensure this script can be run directly for testing
if __name__ == "__main__":
    from config import SmartFactoryConfig
    config = SmartFactoryConfig()
    setup_scene(config)