import matplotlib
matplotlib.use('TkAgg')  # Choose 'TkAgg' or 'Qt5Agg' as the backend

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from sklearn.decomposition import PCA


INITIAL_PYRAMID_FOLDER = "initial-pyramid"
PYRAMID_WITH_PCAS_FOLDER = "pyramid-with-pcas"
FACE_NORMALS_PCA_FOLDER = "face-normals-pca"

folders = [
    INITIAL_PYRAMID_FOLDER,
    PYRAMID_WITH_PCAS_FOLDER,
    FACE_NORMALS_PCA_FOLDER,
    "pyramid-with-pcas-projected-onto-first-pc",
    "pyramid-with-pcas-projected-onto-first-two-pcs",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Define the coordinates of the vertices of the pyramid
vertices = [
    [0.5, 0.5, 0.6],  # Apex of the pyramid
    [0, 0, 0],      # Base vertex 1
    [1, 0, 0],      # Base vertex 2
    [1, 1, 0],      # Base vertex 3
    [0, 1, 0],      # Base vertex 4
]
vertices_single_pc_projection = [[0.5,0.5,0], *vertices[1:]]
vertices_two_pc_projection = [
     vertices[0],
     [0,0,vertices[0][2]],
     vertices[2],
     [1,1,vertices[0][2]],
     vertices[4],
]

# write a function that linearly interpolates between vertices and vertices_single_pc_projection
def interpolate_vertices(vertices, vertices_single_pc_projection, t):
    return np.array(vertices) + t * (np.array(vertices_single_pc_projection) - np.array(vertices)) 

# do an animation and interpolate between vertices and vertices_two_pc_projection


def pyramid_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[0], vertices[3], vertices[4]],
        [vertices[0], vertices[4], vertices[1]],
        # [vertices[1], vertices[2], vertices[3], vertices[4]],  # Base face
    ]

# Define the faces (triangles) of the pyramid using vertex indices
faces = pyramid_faces(vertices)
faces_single_pc_projection = pyramid_faces(vertices_single_pc_projection)
faces_two_pc_projection = pyramid_faces(vertices_two_pc_projection)

face_normals = np.zeros((len(faces), 3))
for i, face in enumerate(faces):
    v1 = np.array(face[1]) - np.array(face[0], dtype=np.float64)
    v2 = np.array(face[2]) - np.array(face[0], dtype=np.float64)
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    face_normals[i] = normal

pcas = np.array([
    [0.393443,  0.0,          0.0],
    [0.0,         0.393443,   0.0],
    [0.0,         0.0,          0.546448]
], dtype=np.float64)

# Compute principal components using PCA
pca = PCA()
pca.fit(face_normals)

def plot_pyramid(angle, file_path, pcas, 
                 show_face_normals=True, 
                 face_normal_center=None,
                 plot_pcas=False, 
                 show_pc_plane=False, 
                 show_target_normals=False,
                 show_pyramid=True,
                 normal_length=0.3,
                 show_plot=False,
                 show_target_mesh=False
                 ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pcas_normalised = pcas / np.linalg.norm(pcas, axis=1)[:, None]

    if show_pyramid:
        # Create a Poly3DCollection object to plot the pyramid
        pyramid = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.5, zorder=1)

        # Add the pyramid to the plot
        ax.add_collection3d(pyramid)

    if show_target_mesh:
        # Create a Poly3DCollection object to plot the pyramid
        pyramid = None
        if len(pcas) == 1:
            pyramid = Poly3DCollection(faces_single_pc_projection, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.5, zorder=1)
        elif len(pcas) == 2:
            pyramid = Poly3DCollection(faces_two_pc_projection, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.5, zorder=1)

        # Add the pyramid to the plot
        if not pyramid is None:
            ax.add_collection3d(pyramid)

    if show_pc_plane and pcas.shape[0] >= 2:
        # Plot the plane spanned by the first two principal components
        plane_point = vertices[0]
        normal = np.cross(pcas[0], pcas[1])
        d = -np.dot(normal, plane_point)
        zz, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        if normal[2] == 0:
            xx = np.full_like(zz, vertices[0][2])
        else:
            xx = (-normal[2] * zz - normal[1] * yy - d) * 1.0 / normal[0]
        xx -= 0.1
        ax.plot_surface(xx, yy, zz, color='blue', alpha=0.3, zorder=1)
    
    if plot_pcas:
        # Plot principal components
        for i in range(pcas.shape[0]):
            pc = pcas[i]
            ax.quiver(*vertices[0], pc[0], pc[1], pc[2], color='red', length=0.5, zorder=2)

    # Calculate and plot the face normals
    for i, face in enumerate(faces):
        normal = face_normals[i]
        normal /= np.linalg.norm(normal) / normal_length

        center = np.mean(face, axis=0)
        if face_normal_center is not None:
            center = face_normal_center
        if show_face_normals:
            ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], color='black', length=1, zorder=2)


        if show_target_normals:
            target_normal = normal @ pcas_normalised.T @ pcas_normalised
            target_normal /= np.linalg.norm(target_normal) / normal_length
            ax.quiver(center[0], center[1], center[2], target_normal[0], target_normal[1], target_normal[2], color='green', length=1, zorder=2)

            if show_face_normals:
                # draw an arrow that shows the rotation between the normal and the target normal
                p1 = center + normal
                p2 = (center + target_normal) - p1
                ax.quiver(*p1, *p2, linestyle='--', color='red', length=1, linewidth=0.5, zorder=2)

    # Set plot limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    # ax.set_axis_off()

    # Rotate the plot
    ax.view_init(elev=20, azim=angle)

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)  # Save each frame as a separate image
    if show_plot:
        plt.show()
    plt.close()

angles = np.arange(60, 150, 15)

# TODO: Plot the mesh after the style was applied
plot_pyramid(60, f"pyramid-target-1pc.png", pcas[2:],
             show_face_normals=False, plot_pcas=False, show_pc_plane=False,
                show_target_normals=False, show_pyramid=False,
                show_target_mesh=True, show_plot=True
             )
plot_pyramid(60, f"pyramid-target-2pc.png", pcas[1:],
             show_face_normals=False, plot_pcas=False, show_pc_plane=False,
                show_target_normals=False, show_pyramid=False,
                show_target_mesh=True, show_plot=True
             )

# Generate 5 images at different angles
for i, angle in enumerate(angles):
    plot_pyramid(angle % 360, f"{INITIAL_PYRAMID_FOLDER}/rotating-pyramid-frame-{i}.png", pcas[1:])

# Generate 5 images at different angles
for i, angle in enumerate(angles):
    plot_pyramid(angle % 360, f"{PYRAMID_WITH_PCAS_FOLDER}/rotating-pyramid-frame-{i}.png", pcas[1:], plot_pcas=True, show_pc_plane=True)

# project onto first pc
for i, angle in enumerate(angles):
    plot_pyramid(angle % 360, f"{folders[3]}/rotating-pyramid-frame-{i}.png", pcas[2:], show_face_normals=True, plot_pcas=True, show_pc_plane=False, show_target_normals=True)

# project onto first two pcs
for i, angle in enumerate(angles):
    plot_pyramid(angle % 360, f"{folders[4]}/rotating-pyramid-frame-{i}.png", pcas[1:], 
                 show_face_normals=True, plot_pcas=True, show_pc_plane=True, 
                 show_target_normals=True)




# generate 5 images of the face normals
# for i, angle in enumerate(np.arange(60, 150, 15)):
#     plot_pyramid(angle % 360, f"{FACE_NORMALS_PCA_FOLDER}/face-normals-pca-frame-{i}.png", pcas[1:], show_face_normals=True, 
#                  plot_pcas=True, show_pc_plane=True, 
#                  show_target_normals=True, show_pyramid=False,
#                  face_normal_center=vertices[0])

