import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import HalfspaceIntersection, ConvexHull

def ray_polyhedron_intersection(ray_origin, ray_dir, planes):
    t_min = 0.0
    t_max = np.inf
    for plane in planes:
        n = plane[:3]
        d = plane[3]
        s = np.dot(n, ray_dir)
        o = np.dot(n, ray_origin) + d

        if np.abs(s) < 1e-8:
            if o >= 0:
                return None  # Ray parallel and outside
            else:
                continue
        t = -o / s
        if s > 0:
            t_max = min(t_max, t)
        else:
            t_min = max(t_min, t)

    if t_min > t_max:
        return None

    return ray_origin + t_max * ray_dir

# Define planes of a cube: ax + by + cz + d < 0
planes = np.array([[1,        0,        0, -6.96812],
[-1,        0,        0, -1.89106],
[0,        1,        0, -6.23366],
[0,       -1,        0, -2.26596],
[0,        0,        1,       -3],
[0,        0,       -1,      0.8]])
# planes = np.array([
#     [ 1,  0,  0, -1],  # x < 1
#     [-1,  0,  0, -1],  # x > -1
#     [ 0,  1,  0, -1],  # y < 1
#     [ 0, -1,  0, -1],  # y > -1
#     [ 0,  0,  1, -1],  # z < 1
#     [ 0,  0, -1, -1]   # z > -1
# ])

# Point inside cube
ray_origin = np.array([2.10894, 1.73404, 3.09121])
ray_dir = np.array([0.205353, 0.419614, 0.884169])
ray_dir = ray_dir / np.linalg.norm(ray_dir)  # normalize

# Find intersection point
intersection_point = ray_polyhedron_intersection(ray_origin, ray_dir, planes)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate polyhedron vertices from halfspaces
hs = HalfspaceIntersection(planes, interior_point=ray_origin)
hull = ConvexHull(hs.intersections)

# Draw polyhedron
for simplex in hull.simplices:
    triangle = hs.intersections[simplex]
    ax.add_collection3d(Poly3DCollection([triangle], facecolors='cyan', edgecolors='k', alpha=0.3))

# Draw ray
t_vals = np.linspace(0, 2, 100)
ray_points = ray_origin[None, :] + t_vals[:, None] * ray_dir
ax.plot(ray_points[:,0], ray_points[:,1], ray_points[:,2], 'r--', label='Ray')

# Draw intersection point
if intersection_point is not None:
    ax.scatter(*intersection_point, color='red', s=50, label='Intersection')
else:
    print("No intersection with polyhedron boundary.")

# Format plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.legend()
ax.set_title("Ray-Polyhedron Intersection")

plt.show()
