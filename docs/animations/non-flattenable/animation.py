import matplotlib
matplotlib.use('TkAgg')  # Choose 'TkAgg' or 'Qt5Agg' as the backend

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# Create a 3D hinge vertex with at least 5 faces
vertices = np.array([
    [0, 0, 0],  # Vertex (hinge point)
    [0.5, 0, 0],
    [1, 0, 0],
    [0.5, 0.25, 0.25],
    [0.5, -0.25, 0.25],
])

faces = [
    [0, 1, 3],
    [1, 2, 3],
    [0, 1, 4],
    [1, 2, 4],
]

# Create a plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the faces
poly3d = [[vertices[vert_idx] for vert_idx in face] for face in faces]
plot = ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

# make a visible red point at vertex 1
ax.scatter(vertices[1][0], vertices[1][1], vertices[1][2], color='red', s=100)

# Set plot limits and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([min(vertices[:, 0]), max(vertices[:, 0])])
ax.set_ylim([min(vertices[:, 1]), max(vertices[:, 1])])
ax.set_zlim([min(vertices[:, 2]), max(vertices[:, 2])])
ax.view_init(elev=50, azim=45)

def rotate(point, angle):
    new_point = np.zeros(3)
    new_point[0] = point[0]
    new_point[1] = point[1] * np.cos(angle) - point[2] * np.sin(angle)
    new_point[2] = point[1] * np.sin(angle) + point[2] * np.cos(angle)
    return new_point
    

def animate(frame, frames=10):
    vertices_new = np.array(vertices)

    vertices_new[3] = rotate(vertices[3], -np.pi/frames/4 * min(frame, frames))
    vertices_new[4] = rotate(vertices[4], np.pi/frames/4 * min(frame, frames))

    # vertices2[3:5, 2] -= vertices[3][2]/frames * (frame+1)
    poly3d = [[vertices_new[vert_idx] for vert_idx in face] for face in faces]
    # ax.view_init(elev=50, azim=45 + frame*15)
    plot.set_verts(poly3d)

ani = FuncAnimation(fig, animate, frames=15, interval=200, repeat=True)

# To save the animation as a GIF
ani.save('animations/non-flattenable/non-flattenable.gif', writer='imagemagick')

plt.show()
