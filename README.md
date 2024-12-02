# Piecewise Developable Surface Approximation with Gauss Stylization

This is a fork of the "[Gauss Stylization](https://cybertron.cg.tu-berlin.de/projects/gaussStylization/)" program by Maximilian Kohlbrenner, Ugo Finnendahl, Tobias Djuren and Marc Alexa. It is modified, such that it applies a piecewise developable style onto a given triangle mesh. It also contains code from "[Cubic Stylization](https://www.dgp.toronto.edu/projects/cubic-stylization/)" by [Hsueh-Ti Derek Liu](https://www.dgp.toronto.edu/~hsuehtil/) and [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/). It provides a [ImGui](https://github.com/ocornut/imgui) to interactively create Piecewise Developable surfaces.

The code for the developable face normal preference generation is in `utils_gauss/gauss_style_developable.cpp`. An explanation on the provided technique can be found at [this page](https://cybertron.cg.tu-berlin.de/p/cgp-ss23/gauss-stylization-developable/)

### Compilation
To compile the application, please type these commands in the terminal
```
cd developableStylization
mkdir build
cd build
cmake ..
make
```
This will create the executable of the developable stylization. To start the application, please run
```
./developableStylization_bin
```
where the example meshes are provided in `/meshes`. Instructions of how to control the Gauss Stylization is listed on the side of the GUI. The exports are placed in the developableStylization folder.

