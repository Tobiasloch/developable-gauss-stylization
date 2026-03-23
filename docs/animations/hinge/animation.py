import matplotlib
matplotlib.use('TkAgg')  # Choose 'TkAgg' or 'Qt5Agg' as the backend

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

# Create a 3D hinge vertex with at least 5 faces
z_max = 0.5
y_max = 1.25
vertices = np.array([
    [0,     0,      0],  # Vertex (hinge point)
    [0.5,   0,      0],
    [1,     0,      0],
    [0.2,     y_max*.8,  z_max*.8],
    [0.5,   y_max,  z_max],
    [.8,     y_max*.8,  z_max*.8],
    [0.2,     -y_max*.8,  z_max*.8],
    [0.5,   -y_max,  z_max],
    [.8,     -y_max*.8,  z_max*.8],
])

faces = [
    [0, 1, 3],
    [1, 3, 4],
    [1, 4, 5],
    [1, 2, 5],
    [1, 6, 7],
    # [2, 4, 5],
    # [0, 6, 7],
    [0, 1, 6],
    [1, 7, 8],
    [1, 2, 8],
]

# Define UV coordinates for the vertices
uv_coords = np.array([
    [0.5, 0.5],
    [0.25, 0.5],
    [0.75, 0.5],
    [0.5, 0.75],
    [0.5, 0.25],
])

# Load a checkerboard pattern image
texture = mpimg.imread('images/checkboard.png')


# Create a plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot the faces
poly3d = [[vertices[vert_idx] for vert_idx in face] for face in faces]
plot = ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

# make a visible red point at vertex 1
ax.scatter(vertices[1][0], vertices[1][1], vertices[1][2], color='red', s=100)

# Set plot limits and labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
ax.set_xlim([min(vertices[:, 0]), max(vertices[:, 0])])
ax.set_ylim([-y_max, y_max])
ax.set_zlim([0, y_max])
ax.view_init(elev=50, azim=45)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])


def rotate(point, angle):
    new_point = np.zeros(3)
    new_point[0] = point[0]
    new_point[1] = point[1] * np.cos(angle) - point[2] * np.sin(angle)
    new_point[2] = point[1] * np.sin(angle) + point[2] * np.cos(angle)
    return new_point
    

def animate(frame, stop=15, start=5):
    vertices_new = np.array(vertices)

    frame_of_animation = min(frame, stop)- start
    frames = stop -start

    if (frame >= start):
        for r, angle in [(range(3,3+3), -np.pi/frames/8 * frame_of_animation), (range(6,6+3), np.pi/frames/8 * frame_of_animation)]:
            for v in r:
                vertices_new[v] = rotate(vertices[v], angle)

    # vertices2[3:5, 2] -= vertices[3][2]/frames * (frame+1)
    poly3d = [[vertices_new[vert_idx] for vert_idx in face] for face in faces]
    # ax.view_init(elev=50, azim=45 + frame*15)
    plot.set_verts(poly3d)

# draw a dotted line from [-0.125, 0, 0] to [1.125, 0, 0]
ax.plot([-0.125, 1.125], [0, 0], [0, 0], color='black', linestyle='dotted', linewidth=5)

ani = FuncAnimation(fig, animate, frames=20, interval=200, repeat=True)

# To save the animation as a GIF
ani.save('animations/hinge/hinge_animation.gif', writer='imagemagick')

plt.show()
