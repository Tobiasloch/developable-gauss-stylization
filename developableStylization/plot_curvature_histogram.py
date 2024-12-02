import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os, sys
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

executable_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder = executable_folder

files = ["output-curvature.csv", "output-original-curvature.csv"]
paths = [os.path.join(folder, files[0]), os.path.join(folder, files[1])]

args = list(sys.argv)
save_path = None
if "-s" in args:
    args.remove("-s")
    save_path = "curvHist.png"

single = False
if "--single" in args:
    args.remove("--single")
    single = True

first_binsize = False
if "--first-binsize" in args:
    args.remove("--first-binsize")
    first_binsize = True

if "--help" in args:
    print("""
    Developable Surface Approximation via Guass Stylization
    
    Options:
    -s      save the file
""")
    exit()

if len(args) > 1:
    paths[0] = args[1]
    if len(args) > 2:
        paths[1] = args[2]

epsilon = 1e-6

# load data from gaussStylization_ImGui/build/curvature.csv which contains the curvature of each vertex in each line
curvature = np.loadtxt(paths[0], delimiter=",")
curvature = np.log(np.abs(curvature) + epsilon)
if not single:
    curvature_original = np.loadtxt(paths[1], delimiter=",")
    curvature_original = np.log(np.abs(curvature_original) + epsilon)
else:
    curvature_original = curvature

r = curvature.max() - curvature.min()
step = r/50
# plot histogram of curvature
if not single:
    plt.hist(curvature_original, bins=[curvature.min()+(step*i) for i in range(50)] if first_binsize else 50, label="input")
plt.hist(curvature, bins=50, alpha=0.5, label="output")
# plt.xlim(-10, 10)

plt.xticks([min(curvature_original.min(), curvature.min()), max(curvature_original.max(), curvature.max())], ['low curvature', 'high curvature'], fontsize=20)

if not single:
    plt.legend(fontsize=20)
# plt.title("Curvature histogram")
# plt.xlabel("Curvature")
# plt.ylabel("Count")

if save_path is not None:
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()
