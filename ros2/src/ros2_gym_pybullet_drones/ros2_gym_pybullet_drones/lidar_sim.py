import numpy as np
import os
os.environ["DISPLAY"] = ":0"  # Force X11 display
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Alternative to GLX
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_avia_lidar(num_scans=5, show_plot=True):
    """Simulate Livox Avia LiDAR and generate point cloud using PyBullet"""
    # Connect to PyBullet
    # physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless mode
    physics_client = p.connect(p.GUI, options="--opengl2")  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load simple environment
    p.loadURDF("plane.urdf")
    p.loadURDF("cube.urdf", [2, 0, 0.5])
    p.loadURDF("sphere2.urdf", [0, 3, 1])
    
    # LiDAR parameters (Livox Avia-like)
    fov_v = np.radians(40)  # 40 deg vertical FOV
    fov_h = np.radians(70)  # 70 deg horizontal FOV
    max_range = 50.0
    min_range = 0.1
    w1 = 763.82589     # Motor frequency 1 (rad/s)
    w2 = -488.41293788 # Motor frequency 2 (rad/s)
    
    # Sensor position and orientation
    sensor_pos = np.array([0, 0, 1.0])  # 1m above origin
    sensor_orn = p.getQuaternionFromEuler([0, 0, 0])  # Facing forward
    
    point_cloud = []
    
    # Simulate multiple scans
    for scan_idx in range(num_scans):
        # Time-dependent pattern generation
        t = scan_idx * 0.1  # 10Hz scan rate
        
        # Generate non-repetitive pattern (Livox Avia rose curve)
        for i in range(240):  # 240 points per scan (simplified)
            # Rose curve equation
            theta = w1 * t + i * 0.01
            phi = w2 * t + i * 0.01
            azimuth = 0.5 * fov_h * (np.cos(theta) + np.cos(phi))
            elevation = 0.5 * fov_v * (np.sin(theta) + np.sin(phi))
            
            # Calculate ray direction
            ray_dir = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth), 
                np.sin(elevation)
            ])
            
            # Transform to world frame
            rotation_matrix = np.array(p.getMatrixFromQuaternion(sensor_orn)).reshape(3, 3)
            ray_dir_world = rotation_matrix.dot(ray_dir)
            
            # Raycast
            ray_to = sensor_pos + ray_dir_world * max_range
            ray_result = p.rayTest(sensor_pos, ray_to)[0]
            
            if ray_result[0] != -1:  # If hit
                hit_distance = ray_result[2] * max_range
                if min_range < hit_distance < max_range:
                    hit_point = sensor_pos + ray_dir_world * hit_distance
                    point_cloud.append(hit_point)
        
        # Step simulation (for moving objects)
        p.stepSimulation()
        time.sleep(0.05)
    
    # Convert to numpy array
    point_cloud = np.array(point_cloud)
    
    # Visualization
    if show_plot and len(point_cloud) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], 
                  s=2, c=point_cloud[:,2], cmap='viridis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Livox Avia Simulation - {len(point_cloud)} Points')
        plt.show()
    
    p.disconnect()
    return point_cloud

# Run the simulation
if __name__ == "__main__":
    import time
    start_time = time.time()
    pc = simulate_avia_lidar(num_scans=10)
    print(f"Generated {len(pc)} points in {time.time()-start_time:.2f} seconds")