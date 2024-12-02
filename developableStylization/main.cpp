#define IGL_VIEWER_VIEWER_QUIET
#include <igl/unproject_onto_mesh.h>
#include <igl/snap_points.h>
#include <igl/upsample.h>

#include <igl/opengl/glfw/imgui/ext/ImGuiMenu.h>
// SELECTION (HOTFIX version)
#include <igl/opengl/glfw/imgui/ext/SelectionPlugin.h>

#include <igl/AABB.h>
#include <igl/screen_space_selection.h>

#include <imgui/imgui.h>

#include <Eigen/Core>

// #include <ctime>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cstdlib>

#include "get_bounding_box.h"
#include "normalize_unitbox.h"

#include "gauss_style_data.h"
#include "gauss_style_precomputation.h"
#include "gauss_style_single_iteration.h"
#include "gauss_style_developable.h"


// SELECTION HOTFIX (https://github.com/alecjacobson/libigl-issue-1656-hot-fix/blob/main/main.cpp)
namespace igl{ namespace opengl{ namespace glfw{ namespace imgui{
class PrePlugin: public igl::opengl::glfw::imgui::ImGuiMenu
{
public:
  PrePlugin(){};
  IGL_INLINE virtual bool pre_draw() override { ImGuiMenu::pre_draw(); return false;}
  IGL_INLINE virtual bool post_draw() override { return false;}
};
class PostPlugin: public igl::opengl::glfw::imgui::ImGuiMenu
{
public:
  PostPlugin(){};
  IGL_INLINE virtual bool pre_draw() override { return false;}
  IGL_INLINE virtual bool post_draw() override {  ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); return false;}
};
}}}}


#ifndef MESH_PATH
#define MESH_PATH "../../meshes/"
#endif

#ifndef OUTPUT_PATH
#define OUTPUT_PATH "../"
#endif

#define WIDTH 960
#define HEIGHT 1079

// state of the mode
struct Mode
{
  Eigen::MatrixXd CV; // point constraints. This is code needed for the gauss stylization, but it not needed for the developability approximation
  bool running = false;
  bool single_step = false;
  unsigned int meshNr = 6;
  int iter = 1;
  double normal_length = 0.025;
  double normal_width = 2;
  bool show_preferred_face_normals = false;
  bool show_gaussian_curvature = false;
  bool gaussian_curvature_sqrt_scale = true;
  bool show_pc_per_vertex = false;
  bool show_face_normals = false;
  int iterations = 0;
  int iterations_max = 1000;
  bool enable_iterations_max = false;
  int gaussian_curvature_max_power = 10;
  bool wireframe_mode = false;
  float face_color[3] = {0.6,0.6,0.6};
  double rotation_angle = 45.0;
  bool show_original_mesh = false;
} state;

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

  // data initialization
  // ---------------------------------------------------------------------------
  gauss_style_data developable_style;

  // load mesh
  // ---------------------------------------------------------------------------
	MatrixXd V, U, VO;
	MatrixXi F;

  // collect all files in MESH_PATH
  std::vector<string> fileNames;
  std::vector<filesystem::path> paths;
  copy(filesystem::directory_iterator(MESH_PATH), filesystem::directory_iterator(), std::back_inserter(paths));
  sort(paths.begin(), paths.end());
  for (const auto& entry : paths) {
    // extract file names
    stringstream nameStream;
    nameStream << entry.filename();
    string name = nameStream.str();
    fileNames.push_back(name.substr(1, name.length() - 6));
  };

  VectorXd originalCurvature;
  auto loadMesh = [&](int nr)
  {
    
    string filename = fileNames.at(nr);
	  string file = MESH_PATH + filename + ".obj";
	  igl::readOBJ(file, V, F);
    normalize_unitbox(V);
    RowVector3d meanV = V.colwise().mean();
    V = V.rowwise() - meanV;
    U = V;
    VO = V;

    // calculate gaussian colormap
    calculateGaussianCurvature(V, F, originalCurvature);
    state.iterations = 0;
  };
  loadMesh(state.meshNr);


  // Set the F(0,0) to be the contrained point (point constaints are not supported)
  state.CV = MatrixXd();
  state.CV.resize(1,3);
  RowVector3d new_c = V.row(F(0,0));
  state.CV.row(0) << new_c;

  // initialize viewer and plugins (selection HOTFIX, see definition of PRE, POST above)
  // ---------------------------------------------------------------------------
  igl::opengl::glfw::Viewer viewer;

  igl::opengl::glfw::imgui::PrePlugin PRE;
  igl::opengl::glfw::imgui::PostPlugin POST;
  igl::opengl::glfw::imgui::ext::ImGuiMenu menu;
  viewer.plugins.push_back(&PRE);
	viewer.plugins.push_back(&menu);
  viewer.plugins.push_back(&POST);

  viewer.data().point_size = 5;
  viewer.core().viewport = Vector4f(WIDTH, 0, WIDTH, HEIGHT);
  int right_view = viewer.core_list[0].id;
  int left_view = viewer.append_core(Vector4f(WIDTH, 0, WIDTH, HEIGHT));


  // colors
  // ---------------------------------------------------------------------------
  const RowVector3d red(250.0/255, 114.0/255, 104.0/255);
  const RowVector3d green(100.0/255, 255/255, 104.0/255);
  const RowVector3d blue(149.0/255, 217.0/255, 244.0/255);
  const RowVector3d orange(250.0/255, 240.0/255, 104.0/255);
  const RowVector3d black(0., 0., 0.);
  const RowVector3d gray(200.0/255, 200.0/255, 200.0/255);
  const auto darkblue = blue * 0.5;
  const auto darkred = red * 0.5;

  // color palette 1
  MatrixXd cp1 (9, 3);
  cp1 << 38./255., 70./255., 83./255.,
        42./255., 157./255., 143./255.,
        233./255., 196./255., 106./255.,
        244./255., 162./255., 97./255.,
        231./255., 111./255., 81./255.,
        0./255., 157./255., 143./255.,
        233./255., 0/255., 106./255.,
        244./255., 162./255., 0./255.,
        231./255., 111./255., 255./255.;

  // ---------------------------------------------------------------------------
  // ---------------------- main draw function ---------------------------------
  // ---------------------------------------------------------------------------
  std::vector<MatrixXd> vertex_principle_components;
  
  const auto &draw = [&]()
  {
    // single iteration step
    // -------------------------------------------------------------------------
    if (state.iterations >= state.iterations_max && state.enable_iterations_max) {
      state.running = false;
    }
    if (state.running)
    {
      state.show_original_mesh = false;
      vertex_principle_components.clear();
      generate_g(U, F, developable_style, vertex_principle_components);

      gauss_style_single_iteration(V, U, F, developable_style, state.iter);

      if (state.single_step) {
        state.running = false;
        state.single_step = false;
      }

      state.iterations++;
    }

    // idle mode
    // -------------------------------------------------------------------------
    viewer.data().clear();
    viewer.data().face_based = true;
    viewer.data().show_lines = (state.wireframe_mode) ? 4294967295 : 0;
    if (state.show_original_mesh)
    {
      viewer.data().set_mesh(V,F);
      viewer.data().set_colors(RowVector3d(state.face_color[0], state.face_color[1], state.face_color[2]));
    }
    // running
    // -------------------------------------------------------------------------
    else
    { 
      viewer.data().set_mesh(U,F);
      viewer.data().set_colors(RowVector3d(state.face_color[0], state.face_color[1], state.face_color[2]));
    }

    // draw input normals
    // -------------------------------------------------------------------------
    RowVector3d origin(0., 0., 0.);
    RowVector3d color = darkblue;

    if (state.show_preferred_face_normals) {
      for (int j = 0; j < developable_style.FGroups.size(); j++) {
        int group = developable_style.FGroups[j];
        for (unsigned i = 0; i < developable_style.style_N[group].rows(); i++)
        {
          Eigen::MatrixXd style = developable_style.style_N[group] * state.normal_length;
          // display the normals on top of the face with index state.group
          Eigen::RowVector3d faceCenter = U.row(F(j, 0)) + U.row(F(j, 1)) + U.row(F(j, 2));
          faceCenter /= 3;

          viewer.data().add_edges(faceCenter, faceCenter + style.row(i), orange);
        }
      }
    }

    if (state.show_face_normals) {
      for (int j = 0; j < F.rows(); j++) {
        // display the normals on top of the face with index state.group
        Eigen::RowVector3d faceCenter = U.row(F(j, 0)) + U.row(F(j, 1)) + U.row(F(j, 2));
        faceCenter /= 3;
        Eigen::RowVector3d edge1 = U.row(F(j, 1)) - U.row(F(j, 0));
        Eigen::RowVector3d edge2 = U.row(F(j, 2)) - U.row(F(j, 0));
        Eigen::RowVector3d normal = edge1.cross(edge2);
        normal.normalize();

        viewer.data().add_edges(faceCenter, faceCenter + (normal*state.normal_length), color);
      }
    }

    if (state.show_gaussian_curvature) {
      VectorXd K;
      if (state.show_original_mesh) {
        calculateGaussianCurvature(V, F, K);
      } else {
        calculateGaussianCurvature(U, F, K);
      }

      float max = pow(2, state.gaussian_curvature_max_power);
      viewer.data().set_data(K, -max, max, igl::COLOR_MAP_TYPE_VIRIDIS, 21);
    }

    if (state.show_pc_per_vertex) {
      for (unsigned i = 0; i < vertex_principle_components.size(); i++)
      {
        MatrixXd pcs = vertex_principle_components[i];
        for (unsigned j = 0; j < pcs.cols() && j < 2; j++)
        {
          Eigen::MatrixXd pc = pcs.col(pcs.cols()-1-j).transpose();
          Eigen::RowVector3d pc_color = green;
          if (j == 1) pc_color = red;

          viewer.data().add_edges(U.row(i), U.row(i) + (pc*state.normal_length), pc_color);
          viewer.data().add_points(U.row(i), black);
        }
      }
    }

    viewer.data().line_width = state.normal_width;
    
    // draw origin
    // -------------------------------------------------------------------------
    viewer.data().add_points(origin.transpose(), black);
  };
  
  // reset
  // ---------------------------------------------------------------------------
  const auto & reset = [&](bool reset_constraints=true){
    state.running = false;
    state.iterations = 0;

    V = VO;
    U = V;
    draw();
  };

  // ---------------------------------------------------------------------------
  // ------------------------- Interactions ------------------------------------
  // ---------------------------------------------------------------------------

  // mesh rotation parameters
  // ---------------------------------------------------------------------------
  Matrix3d Rx, Ry, Rz;

  const auto generateRotationMatrices = [&](double theta_x, double theta_y, double theta_z) {
    Rx << 1., 0., 0.,
          0., cos(theta_x/180.*3.14), -sin(theta_x/180.*3.14),
          0., sin(theta_x/180.*3.14), cos(theta_x/180.*3.14);

    Ry << cos(theta_y/180.*3.14), 0., sin(theta_y/180.*3.14),
          0., 1., 0.,
          -sin(theta_y/180.*3.14), 0, cos(theta_y/180.*3.14);

    Rz << cos(theta_z/180.*3.14), -sin(theta_z/180.*3.14), 0.,
          sin(theta_z/180.*3.14), cos(theta_z/180.*3.14), 0.,
          0., 0., 1;
  };

  generateRotationMatrices(state.rotation_angle, state.rotation_angle, state.rotation_angle);

  const auto rotate = [&](Matrix3d Ri)
  {
    U = (Ri * U.transpose()).transpose();
  };
  // when key pressed do
  // ---------------------------------------------------------------------------
  viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod) {
    switch(key) {
      case 'R':
      case 'r': {
        reset(false);
        break;
      }
      case 'Q':
      case 'q': {
        rotate(Rx);
        break;
      }
      case 'W':
      case 'w': {
        rotate(Rx.transpose());
        break;
      }
      case 'A':
      case 'a': {
        rotate(Ry);
        break;
      }
      case 'S':
      case 's': {
        rotate(Ry.transpose());
        break;
      }
      case 'X':
      case 'x': {
        rotate(Rz);
        break;
      }
      case 'Y':
      case 'y': {
        rotate(Rz.transpose());
        break;
      }
      case 'i':
      case 'I': {
        state.single_step = true;
      }
      case ' ': {
        // start/stop stylization

        if (!state.running) {
          // set constrained points and pre computation
          U = V;
          igl::snap_points(state.CV, V, developable_style.b);
          developable_style.bc.setZero(developable_style.b.size(), 3);
          for (int ii = 0; ii < developable_style.b.size(); ii++)
            developable_style.bc.row(ii) = V.row(developable_style.b(ii));
          gauss_style_precomputation(V, F, developable_style);
        }
        state.running = !state.running;
        break;
      }

      default:
        return false;
    }
    draw();
    return true;
  };

  // on resize
  viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) {
    v.core().viewport = Vector4f(0, 0, w, h);
    v.core(right_view).viewport = Vector4f(w, 0, 0, h);
    return true;
  };

  // default mode: keep drawing the current mesh
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &)->bool {
    if(viewer.core().is_animating)
        draw();
    return false;
  };

  // ---------------------------------------------------------------------------
  // -------------------------- GUI Windows ------------------------------------
  // ---------------------------------------------------------------------------
  // draw additional windows
  char outputName[128] = "output";
  menu.callback_draw_viewer_window = []() {};
  menu.callback_draw_custom_window = [&]()
  {
    // Define next window position + size
    {
      ImGui::SetNextWindowPos(ImVec2(0.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(275, 850), ImGuiCond_FirstUseEver);
      ImGui::Begin(
        "Settings", nullptr,
        ImGuiWindowFlags_NoSavedSettings
      );
    }
    {
      // How to use
      if (ImGui::CollapsingHeader("Instructions", ImGuiTreeNodeFlags_DefaultOpen))
      {
      ImGui::Text("Instructions");
        ImGui::BulletText("[space]  start stylization");
        ImGui::BulletText("I           single step");
        ImGui::BulletText("R           reset ");
        ImGui::BulletText("Q/W     rotate x-axis");
        ImGui::BulletText("A/S       rotate y-axis");
        ImGui::BulletText("Y/X       rotate z-axis");
        ImGui::BulletText("Y/X       rotate z-axis");
        ImGui::Text(" ");
      }
      if (ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen))
      {
        ImGui::Text("Number of Vertices: ");ImGui::SameLine();ImGui::Text("%lu", V.rows());
        ImGui::Text("Number of Faces: ");ImGui::SameLine();ImGui::Text("%lu", F.rows());

        ImGui::Text("Stylization Iterations: %d", state.iterations);

        // calculate the vertex change between U and U_last and U and V
        double vertex_change_from_V = 0;
        for (int i = 0; i < U.rows(); i++) {
          vertex_change_from_V += (U.row(i) - V.row(i)).norm();
        }
        
        ImGui::Text("Total Vertex Change from original: %f", vertex_change_from_V);
        ImGui::Text(" ");
      }
    }
    {
      ImGui::PushItemWidth(-150);

      ImGui::Separator();
      if (ImGui::Combo("Input Mesh", (int*)&state.meshNr, fileNames)) {
        reset();
        loadMesh(state.meshNr);
        draw();
      }

      if (ImGui::CollapsingHeader("Gauss Stylization Parameters", ImGuiTreeNodeFlags_DefaultOpen))
      {

        ImGui::SliderInt("ADMM Iterations", &state.iter, 1, 10);

        ImGui::DragScalar("lambda", ImGuiDataType_Double, &developable_style.lambda, 2e-1, 0, 0, "%.1e");
        ImGui::DragScalar("mu", ImGuiDataType_Double, &developable_style.mu_default, 2e-1, 0, 0, "%.1e");
        ImGui::DragScalar("sigma", ImGuiDataType_Double, &developable_style.sigma_default, 2e-1, 0, 0, "%.1e");

        ImGui::Checkbox("", &state.enable_iterations_max);
        ImGui::SameLine();

        if (!state.enable_iterations_max) {
          ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
          ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        }
        ImGui::DragInt("max Iteration", &state.iterations_max, 1, 0, INT_MAX);
        if (!state.enable_iterations_max) {
          ImGui::PopItemFlag();
          ImGui::PopStyleVar();
        }
      }
      
      
      if (ImGui::CollapsingHeader("Developability Parameters", ImGuiTreeNodeFlags_DefaultOpen))
      {
          ImGui::DragScalar("theta", ImGuiDataType_Double, &developable_style.collapse_threshhold, 2e-1, 0, 0, "%.1e");
      }

      if (ImGui::CollapsingHeader("Visualization")) {
        // set face color
        ImGui::ColorEdit3("Face Color", state.face_color, ImGuiColorEditFlags_NoInputs);
        ImGui::Checkbox("Enable Wireframe Mode", &state.wireframe_mode);
        ImGui::Checkbox("Show gaussian curvature", &state.show_gaussian_curvature);
        ImGui::SliderInt("curvature max (log2 scale)", &state.gaussian_curvature_max_power, 1, 15);
        if (ImGui::Checkbox("Show Original Mesh", &state.show_original_mesh)) {
          if (state.show_original_mesh) state.running = false;
        }

        ImGui::Checkbox("Show face normals", &state.show_face_normals);
        ImGui::Checkbox("Show last Preferences", &state.show_preferred_face_normals);

        ImGui::Checkbox("Show last principle components", &state.show_pc_per_vertex);

        ImGui::DragScalar("normal length", ImGuiDataType_Double, &state.normal_length, 2e-2, 0, 0, "%.1e");
        ImGui::DragScalar("normal width", ImGuiDataType_Double, &state.normal_width, 2e-2, 0, 0, "%.1e");
        
        if (ImGui::DragScalar("rotation angle", ImGuiDataType_Double, &state.rotation_angle, 2e-2, 0, 0)) {
          generateRotationMatrices(state.rotation_angle, state.rotation_angle, state.rotation_angle);
        }
      }
    }

    {
      if (ImGui::CollapsingHeader("Export")) {
        // output file name
        ImGui::InputText(".obj", outputName, IM_ARRAYSIZE(outputName));

        if (ImGui::Button("save output mesh", ImVec2(-1, 0))) {
          string outputFile = OUTPUT_PATH;
          outputFile.append(outputName);
          outputFile.append(".obj");

          igl::writeOBJ(outputFile, U, F);

          string inputFile = OUTPUT_PATH;
          inputFile.append(outputName);
          inputFile.append("-original.obj");
          igl::writeOBJ(inputFile, V, F);
        }

        const auto & save_curvature = [&](MatrixXd& vertices, std::string filename){
          VectorXd K;
          calculateGaussianCurvature(vertices, F, K);

          std::ofstream outputFile(filename);
          Eigen::IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
          outputFile << K.format(CSVFormat);
          outputFile.close();
        };
        if (ImGui::Button("save curvature", ImVec2(-1, 0))) {
          string curvatureOutput = OUTPUT_PATH;
          string curvatureOutputOriginal = OUTPUT_PATH;
          curvatureOutput.append(outputName);
          curvatureOutput.append("-curvature.csv");
          curvatureOutputOriginal.append(outputName);
          curvatureOutputOriginal.append("-original-curvature.csv");

          save_curvature(V, curvatureOutputOriginal);
          save_curvature(U, curvatureOutput);
        }
      }

    }

    ImGui::End();
  };

  // initialize the scene
  {
    viewer.data().set_mesh(V,F);
    // viewer.data().show_lines = (state.wireframe_mode) ? 1 : 0;
    viewer.data().point_size = 5;
    viewer.core().is_animating = true;
    viewer.data().face_based = true;
    Vector4f backColor;
    backColor << 1., 1., 1., 1.;
    viewer.core().background_color = backColor;

    draw();
    viewer.launch(true,false,"Developable Surface Approximation with Gauss Stylization", WIDTH*2, HEIGHT);
  }

}
