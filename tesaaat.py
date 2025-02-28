import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_snapshot():
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. AGV Positions Plot (Top Left)
    ax1 = plt.subplot(221)
    agv_positions = np.array([
        [2.0, 2.0, 2.5],
        [3.0, 12.0, 2.5]
    ])
    
    # Plot factory boundaries
    ax1.plot([0, 20, 20, 0, 0], [0, 0, 20, 20, 0], 'k-', label='Factory Bounds')
    # Plot AGV positions
    ax1.scatter(agv_positions[:, 0], agv_positions[:, 1], c='red', s=100, label='AGVs')
    # Plot BS position
    ax1.scatter(10, 0.5, c='blue', s=100, label='Base Station')
    
    ax1.set_title('Factory Layout')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. LOS Conditions Plot (Top Right)
    ax2 = plt.subplot(222)
    los_conditions = np.array([0, 0])  # From the data
    conditions = ['AGV 1', 'AGV 2']
    colors = ['red' if x == 0 else 'green' for x in los_conditions]
    ax2.bar(conditions, [1, 1], color=colors)
    ax2.set_title('Line of Sight Conditions')
    ax2.set_ylabel('Status')
    for i, condition in enumerate(conditions):
        status = 'NLOS' if los_conditions[i] == 0 else 'LOS'
        ax2.text(i, 0.5, status, ha='center')
    
    # 3. Channel Matrix Magnitude Plot (Bottom Left)
    ax3 = plt.subplot(223)
    # Taking a slice of the complex channel matrix
    channel_slice = np.abs(np.array([[-1.76277968e-06+6.30380271e-07j,
                                    -1.75792570e-06+6.42250427e-07j,
                                    -1.75299476e-06+6.54082896e-07j]]))
    ax3.plot(channel_slice[0], label='Channel Magnitude')
    ax3.set_title('Channel Matrix Magnitude (Sample)')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Magnitude')
    ax3.grid(True)
    
    # 4. Path Delays Plot (Bottom Right)
    ax4 = plt.subplot(224)
    path_delays = np.array([1.0478072e-08, 3.3886419e-08, 3.3886419e-08])
    ax4.hist(path_delays, bins=20, color='blue', alpha=0.7)
    ax4.set_title('Path Delays Distribution')
    ax4.set_xlabel('Delay (s)')
    ax4.set_ylabel('Count')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

# Generate and save the plot
fig = plot_simulation_snapshot()
plt.savefig('simulation_snapshot.png', dpi=300, bbox_inches='tight')
plt.show()