import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.sionna_ply_generator import SionnaPLYGenerator

def main():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the meshes directory path
    meshes_dir = os.path.join(current_dir, 'src/meshes')
    
    # Ensure meshes directory exists
    os.makedirs(meshes_dir, exist_ok=True)
    
    # Generate PLY files for the smart factory scenario
    SionnaPLYGenerator.generate_factory_geometries(
        room_dims=[20, 20, 5],  # Room dimensions
        shelf_dims=[2, 1, 4],   # Shelf dimensions
        output_dir=meshes_dir   # Explicitly use full path
    )
    
    # Verify file generation
    print("Attempting to generate PLY files...")
    print(f"Meshes directory: {meshes_dir}")
    
    # List files in the meshes directory
    try:
        files = os.listdir(meshes_dir)
        print("Generated files:")
        for file in files:
            print(file)
        
        if not files:
            print("WARNING: No files were generated!")
    except Exception as e:
        print(f"Error listing files: {e}")

if __name__ == "__main__":
    main()