#include "ImGuizmoPlugin.h"
#include <imgui/imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_fonts_droid_sans.h>
#include <GLFW/glfw3.h>

namespace igl{ namespace opengl{ namespace glfw{ namespace imgui{
namespace ext{ 

IGL_INLINE void ImGuizmoPlugin::init(igl::opengl::glfw::Viewer *_viewer)
{
  ImGuiMenu::init(_viewer);
}
IGL_INLINE bool ImGuizmoPlugin::pre_draw() 
{
  if(!visible){ return false; }
  //ImGuiMenu::pre_draw();
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGuizmo::BeginFrame();
  ImGui::PopStyleVar();
  return false;
}
IGL_INLINE bool ImGuizmoPlugin::post_draw() 
{
  if(!visible){ return false; }
  // Don't draw the Viewer's default menu: draw just the ImGuizmo
  Eigen::Matrix4f view = (viewer->core().view / viewer->core().camera_zoom);
  Eigen::Matrix4f proj = viewer->core().proj;
  if(viewer->core().orthographic){ view(2,3) -= 1000;/* Hack depth */ }
  // ImGuizmo doesn't like a lot of scaling in view, shift it to T
  const float z = viewer->core().camera_base_zoom;
  const Eigen::Matrix4f S = 
    (Eigen::Matrix4f()<< z,0,0,0, 0,z,0,0, 0,0,z,0, 0,0,0,1).finished();
  view = (view * S.inverse()).eval();
  const Eigen::Matrix4f T0 = T;
  T = (S * T).eval();
  ImGuiIO& io = ImGui::GetIO();
  ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
  ImGuizmo::Manipulate(
    view.data(),proj.data(),operation,ImGuizmo::LOCAL,T.data(),NULL,NULL);
  // invert scaling that was shifted onto T
  T = (S.inverse() * T).eval();
  const float diff = (T-T0).array().abs().maxCoeff();
  // Only call if actually changed; otherwise, triggers on all mouse events
  if( diff > 2e-6) { callback(T); }
  //ImGui::Render();
  //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  return false;
}

IGL_INLINE bool ImGuizmoPlugin::mouse_down(int button, int modifier)
{
  if(callback_mouse_down && callback_mouse_down(*this,button,modifier)) return true;
  return ImGuiMenu::mouse_down(button,modifier);
}

IGL_INLINE bool ImGuizmoPlugin::mouse_up(int button, int modifier)
{
  if(callback_mouse_up && callback_mouse_up(*this,button,modifier)) return true;
  return ImGuiMenu::mouse_up(button,modifier);
}

IGL_INLINE bool ImGuizmoPlugin::mouse_move(int mouse_x, int mouse_y)
{
  if(callback_mouse_move && callback_mouse_move(*this,mouse_x,mouse_y)) return true;
  return ImGuiMenu::mouse_move(mouse_x,mouse_y);
}

IGL_INLINE bool ImGuizmoPlugin::mouse_scroll(float delta_y)
{
  return ImGuiMenu::mouse_scroll(delta_y);
}


}}}}}
