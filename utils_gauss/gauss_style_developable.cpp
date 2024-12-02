#include "gauss_style_developable.h"

void performPCA(const Eigen::MatrixXd &normals, Eigen::MatrixXd &projected_normals, double collapse_threshhold, MatrixXd &vertex_principle_components)
{
  // Compute the covariance matrix
  Eigen::MatrixXd covariance = (normals.transpose() * normals)/(normals.rows());

  // Perform eigenvalue decomposition
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);
  if (eigensolver.info() != Eigen::Success)
  {
    std::cout << "Eigenvalue decomposition failed\n";
    return;
  }
  Eigen::MatrixXd eigenvalues = eigensolver.eigenvalues();
  Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

  // this is only for visualization, not needed for the algorithm
  vertex_principle_components = eigenvectors;
  for (int i = 0; i < vertex_principle_components.cols(); ++i) {
    vertex_principle_components.col(i).normalize();
    vertex_principle_components.col(i) *= eigenvalues(i);
  }

  // Project the original face normals onto the first or first two principal components
  MatrixXd principle_components;
  if (eigenvalues(eigenvectors.cols()-2) < collapse_threshhold) {
    // take only the first component into the calculation
    principle_components = eigenvectors.rightCols(1);
  } else {
    // also use the second component
    principle_components = eigenvectors.rightCols(2);
  }
  projected_normals = (normals * principle_components) * principle_components.transpose();
  projected_normals.rowwise().normalize();
}

void init_style(Eigen::Ref<Eigen::MatrixXi> F, gauss_style_data &gauss_data) {
  gauss_data.mu = std::vector<double>(F.rows(), gauss_data.mu_default);
  gauss_data.sigma = std::vector<double>(F.rows(), gauss_data.sigma_default);
  gauss_data.caxiscontrib = std::vector<double>(F.rows(), 0.5);
  gauss_data.n_weights = std::vector<Eigen::VectorXd>(F.rows(), Eigen::VectorXd(0));
  gauss_data.r_weights = std::vector<Eigen::VectorXd>(F.rows(), Eigen::VectorXd(0));
  gauss_data.style_R = std::vector<Eigen::VectorXd>(F.rows(), Eigen::VectorXd(0));
  gauss_data.FGroups = Eigen::VectorXi::LinSpaced(F.rows(), 0, F.rows() - 1);
}


// Liu Paper code: https://github.com/HTDerekLiu/normal_driven_cpp/blob/main/utils/developable_normals.cpp
void generate_g(
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &F,
    gauss_style_data &gauss_data,
    std::vector<MatrixXd> &vertex_principle_components
)
{
  vertex_principle_components.resize(V.rows());
  Eigen::MatrixXd FN;
  igl::per_face_normals(V, F, FN);

  init_style(F, gauss_data);
  
  gauss_data.style_N = std::vector<Eigen::MatrixXd>(F.rows(), Eigen::MatrixXd::Zero(1, 3));
  std::vector<std::list<Eigen::MatrixXd>> preferred_face_normals(F.rows());

  for (int i = 0; i < V.rows(); i++)
  {
    std::vector<int> faces = gauss_data.adjFList[i];
    Eigen::VectorXi faces_vec = Map<VectorXi, Eigen::Unaligned>(faces.data(), faces.size());;
    Eigen::MatrixXd projected_normals;

    if (faces.size() == 0) continue;
    else if (faces.size() == 1)  {
      projected_normals = FN.row(faces[0]);
    } else {
      // generate normal Matrix
      Eigen::MatrixXd face_normals(faces.size(), 3);
      igl::slice(FN,faces_vec,1,face_normals);

      for (int j = 0; j < faces.size(); ++j)
      {
        Eigen::MatrixXd face_normal = FN.row(faces[j]);
        face_normal.normalize();
        face_normals.row(j) = face_normal;
      }
      MatrixXd principle_components;
      performPCA(face_normals, projected_normals, gauss_data.collapse_threshhold, principle_components);
      vertex_principle_components[i] = principle_components;

      if (projected_normals.hasNaN()) {
        std::cout << "projected_normals has NaNs" << std::endl;
        projected_normals = face_normals;
      }
    }

    for (int j = 0; j < faces.size(); ++j)
    {
      gauss_data.style_N[faces[j]] += projected_normals.row(j);
    }
    
    
  }

  // normalise style_n
  for (int i = 0; i < gauss_data.style_N.size(); i++)
  {
    if (gauss_data.style_N[i].isZero()) continue;
    gauss_data.style_N[i].row(0).normalize();
  }

  normalize_g(gauss_data);
}


// code from https://github.com/libigl/libigl/blob/main/tutorial/202_GaussianCurvature/main.cpp
void calculateGaussianCurvature(
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &F,
    Eigen::VectorXd &K) {
      // Compute integral of Gaussian curvature
      igl::gaussian_curvature(V,F,K);
      // Compute mass matrix
      SparseMatrix<double> M,Minv;
      igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
      igl::invert_diag(M,Minv);
      // Divide by area to get integral average
      K = (Minv*K).eval();
    }
