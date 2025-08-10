import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def point_to_aabb_distance(center, dims, point):
    cx, cy, cz = center
    w, h, l = dims
    px, py, pz = point

    xmin, xmax = cx - w / 2, cx + w / 2
    ymin, ymax = cy - h / 2, cy + h / 2
    zmin, zmax = cz - l / 2, cz + l / 2

    qx = np.clip(px, xmin, xmax)
    qy = np.clip(py, ymin, ymax)
    qz = np.clip(pz, zmin, zmax)

    closest_point = (qx, qy, qz)
    distance = np.linalg.norm([qx - px, qy - py, qz - pz])
    return distance, closest_point

def draw_aabb(ax, center, dims):
    cx, cy, cz = center
    w, h, l = dims
    xmin, xmax = cx - w / 2, cx + w / 2
    ymin, ymax = cy - h / 2, cy + h / 2
    zmin, zmax = cz - l / 2, cz + l / 2

    corners = np.array([[xmin, ymin, zmin],
                        [xmax, ymin, zmin],
                        [xmax, ymax, zmin],
                        [xmin, ymax, zmin],
                        [xmin, ymin, zmax],
                        [xmax, ymin, zmax],
                        [xmax, ymax, zmax],
                        [xmin, ymax, zmax]])

    faces = [[corners[j] for j in [0,1,2,3]],
             [corners[j] for j in [4,5,6,7]],
             [corners[j] for j in [0,1,5,4]],
             [corners[j] for j in [2,3,7,6]],
             [corners[j] for j in [1,2,6,5]],
             [corners[j] for j in [0,3,7,4]]]

    box = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=0.2)
    ax.add_collection3d(box)

def main():
    # Define the box and point
    center = (0, 0, 0)
    dims = (4, 2, 6)  # width, height, length
    point = (5, 0, 1)

    # Compute distance and closest point
    distance, closest = point_to_aabb_distance(center, dims, point)
    print(f"Distance: {distance:.3f}")
    print(f"Closest point on box: {closest}")

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_aabb(ax, center, dims)

    # Plot original point
    ax.scatter(*point, c='r', label='Point', s=50)
    ax.scatter(*closest, c='g', label='Closest Point', s=50)

    # Line between point and closest point
    ax.plot([point[0], closest[0]], [point[1], closest[1]], [point[2], closest[2]], 'k--', label='Distance')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Distance from Point to AABB')
    ax.legend()
    ax.set_box_aspect([1,1,1])

    # Auto-scale limits
    all_points = np.array([point, closest])
    for axis in 'xyz':
        getattr(ax, f'set_{axis}lim')(all_points.min() - 2, all_points.max() + 2)

    plt.show()

if __name__ == '__main__':
    main()
