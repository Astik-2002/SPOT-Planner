import trimesh
import numpy as np

def create_torus(R=1.0, r=0.3, segments=64, rings=32, filename="torus.obj"):
    theta = np.linspace(0, 2 * np.pi, segments)
    phi = np.linspace(0, 2 * np.pi, rings)
    theta, phi = np.meshgrid(theta, phi)

    # Parametric equations
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    faces = []
    for i in range(rings - 1):
        for j in range(segments - 1):
            a = i * segments + j
            b = a + 1
            c = a + segments
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)
    print(f"Torus mesh saved to {filename}")

create_torus(R=2.0, r=0.2, filename="/home/astik/gym-pybullet-drones/gym_pybullet_drones/assets/torus.obj")
