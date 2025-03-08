import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_obstacle(position, dimensions):
    """Create vertices for a 3D obstacle"""
    x, y, z = position
    l, w, h = dimensions
    
    vertices = np.array([
        [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
        [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
    ])
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    
    return faces

def create_curved_path():
    """Create a smooth curved path for AGV2 with updated corner"""
    t = np.linspace(0, 1, 100)
    
    points = [
        (17, 18),  # Start (top right)
        (3, 18),   # Top left
        (3, 4),    # Bottom left (تغییر به y=4)
        (17, 4),   # Bottom right (تغییر به y=4)
        (17, 18)   # Back to start
    ]
    
    x_coords = []
    y_coords = []
    
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        if i == 0:  # Start to top left
            xc = (x1 + x2)/2
            yc = y1
        elif i == 1:  # Top left to bottom left
            xc = x1
            yc = (y1 + y2)/2
        elif i == 2:  # Bottom left to bottom right
            xc = (x1 + x2)/2
            yc = y1
        else:  # Bottom right to start
            xc = x2
            yc = (y1 + y2)/2
            
        x = (1-t)**2 * x1 + 2*(1-t)*t * xc + t**2 * x2
        y = (1-t)**2 * y1 + 2*(1-t)*t * yc + t**2 * y2
        
        x_coords.extend(x)
        y_coords.extend(y)
    
    return x_coords, y_coords

def plot_factory():
    print("Setting up plot...")
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 5)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('X (m)', labelpad=10)
    ax.set_ylabel('Y (m)', labelpad=10)
    ax.set_zlabel('Z (m)', labelpad=10)
    ax.set_title('Smart Factory Layout with AGV Trajectories', pad=20)
    
    # Plot floor
    print("Creating floor...")
    xx, yy = np.meshgrid(np.linspace(0, 20, 2), np.linspace(0, 20, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    # Plot obstacles (O1 و O2 نزدیک‌تر به BS)
    print("Creating obstacles...")
    obstacles = [
        ((6, 6, 0), (4, 1, 4)),    # O1 - تغییر از [4, 4] به [6, 6]
        ((12, 6, 0), (2, 1, 4)),   # O2 - تغییر از [12, 4] به [12, 6]
        ((4, 10, 0), (1, 2, 5)),   # O3
        ((14, 12, 0), (1, 2, 5)),  # O4
        ((8, 16, 0), (4, 1, 5))    # O5
    ]
    
    for i, ((x, y, z), (l, w, h)) in enumerate(obstacles):
        faces = create_obstacle((x, y, z), (l, w, h))
        obstacle = Poly3DCollection(faces, 
                                  facecolors='lightgray', 
                                  alpha=0.6,
                                  edgecolor='black',
                                  linewidth=1)
        ax.add_collection3d(obstacle)
        ax.text(x+l/2, y+w/2, z+h, f'O{i+1}', fontsize=10)
        
        ax.plot([10, x+l/2], [10, y+w/2], [4.8, 2], 
                'r-', alpha=0.5, linewidth=1)
    
    # Plot base station
    print("Adding base station...")
    ax.scatter(10, 10, 4.8, color='red', s=200, marker='^', label='Base Station')
    
    # Plot AGV 1 trajectory
    print("Creating AGV 1 path...")
    agv1_x = [2, 18]
    agv1_y = [3, 3]
    agv1_z = [0.5, 0.5]
    ax.plot(agv1_x, agv1_y, agv1_z, 'b-', linewidth=2, label='AGV 1 Path')
    ax.scatter(2, 3, 0.5, color='blue', s=150, marker='o', label='AGV 1')
    
    # Plot AGV 2 trajectory
    print("Creating AGV 2 path...")
    x_coords, y_coords = create_curved_path()
    z_coords = [0.5] * len(x_coords)
    ax.plot(x_coords, y_coords, z_coords, 'g-', linewidth=2, label='AGV 2 Path')
    ax.scatter(17, 18, 0.5, color='green', s=150, marker='o', label='AGV 2')
    
    # Set view and legend
    print("Configuring view settings...")
    ax.view_init(elev=30, azim=225)
    ax.legend(bbox_to_anchor=(1.15, 0.5), loc='center left')
    
    try:
        ax.set_box_aspect([1, 1, 0.25])
        print("Using set_box_aspect for plot proportions")
    except AttributeError:
        print("Warning: set_box_aspect not available.")
        ax.set_aspect('auto')
    
    return fig, ax

def main():
    print("Starting factory visualization...")
    try:
        fig, ax = plot_factory()
        print("Plot created successfully!")
        
        plt.tight_layout()
        output_file = 'factory_layout_independent_paths.png'
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
        
        try:
            plt.show()
            print("Plot displayed successfully")
        except Exception as e:
            print(f"Could not display plot: {e}")
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()