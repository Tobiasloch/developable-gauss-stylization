#ifndef IGL_OPENGL_GFLW_IMGUI_EXT_SELECTIONPLUGIN_H
#define IGL_OPENGL_GFLW_IMGUI_EXT_SELECTIONPLUGIN_H
#include <igl/igl_inline.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imguizmo/ImGuizmo.h>
#include <Eigen/Dense>
#include <vector>

namespace igl{ namespace opengl{ namespace glfw{ namespace imgui{
namespace ext{ 

class SelectionPlugin: public igl::opengl::glfw::imgui::ImGuiMenu
{
public:
  // customizable hotkeys
  std::string MARQUEE_KEY = "Mm";
  // leave 'L' for show_lines in viewer
  std::string LASSO_KEY = "l";
  std::string OFF_KEY = "Vv";
  enum Mode
  {
    OFF                 = 0,
    RECTANGULAR_MARQUEE = 1,
    ELLIPTICAL_MARQUEE  = 2,
    POLYGONAL_LASSO     = 3,
    LASSO               = 4,
    NUM_MODES           = 5
  } mode = RECTANGULAR_MARQUEE;
  bool is_down = false;
  bool has_moved_since_down = false;
  bool is_drawing = false;
  Eigen::Matrix<float,2,2> M = Eigen::Matrix<float,2,2>::Zero();
  std::vector<Eigen::RowVector2f> L;
  // callback called when slection is completed (usually on mouse_up)
  std::function<void(void)> callback;
  // callback called after mode is changed 
  std::function<void(Mode)> callback_post_mode_change;
  // Initilize with rotate operation on an identity transform (at origin)
  SelectionPlugin(){};
  IGL_INLINE virtual void init(igl::opengl::glfw::Viewer *_viewer) override;
  IGL_INLINE virtual bool pre_draw() override;
  IGL_INLINE virtual bool post_draw() override;
  IGL_INLINE virtual bool mouse_down(int button, int modifier) override;
  IGL_INLINE virtual bool mouse_up(int button, int modifier) override;
  IGL_INLINE virtual bool mouse_move(int mouse_x, int mouse_y) override;
  IGL_INLINE virtual bool key_pressed(unsigned int key, int modifiers) override;
  IGL_INLINE void clear();
  // helpers
  IGL_INLINE static void circle(const Eigen::Matrix<float,2,2> & M,  std::vector<Eigen::RowVector2f> & L);
  IGL_INLINE static void rect(const Eigen::Matrix<float,2,2> & M,  std::vector<Eigen::RowVector2f> & L);
  IGL_INLINE static Eigen::RowVector2f xy(const Viewer * v);
};

}}}}}

#ifndef IGL_STATIC_LIBRARY
#include "SelectionPlugin.cpp"
#endif
#endif
