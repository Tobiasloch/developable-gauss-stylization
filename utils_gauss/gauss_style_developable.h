#ifndef GAUSS_STYLE_DEVELOPABLE_H_
#define GAUSS_STYLE_DEVELOPABLE_H_

#include <Eigen/Dense>
#include <igl/per_face_normals.h>
#include <igl/vertex_triangle_adjacency.h>
// #include <igl/faces_around_vertex.h>
#include <igl/adjacency_list.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/slice.h>
#include <igl/svd3x3.h>

#include "gauss_style_data.h"
#include "normalize_g.h"

// use the namespace Eigen for convenience
using namespace Eigen;

void generate_g(    
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &F,
    gauss_style_data &gauss_data,
    std::vector<Eigen::MatrixXd> &vertex_principle_components);

void calculateGaussianCurvature(
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &F,
    Eigen::VectorXd &K);

#endif