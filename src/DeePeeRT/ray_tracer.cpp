
#include "xdg/DeePeeRT/ray_tracer.h"

namespace xdg {

  DeePeeRTRayTracer::DeePeeRTRayTracer() = default;
  DeePeeRTRayTracer::~DeePeeRTRayTracer() = default;


  std::pair<TreeID, TreeID>
  DeePeeRTRayTracer::register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                                    MeshID volume)
  {
    // set up ray tracing tree for boundary faces of the volume
    TreeID faces_tree = create_surface_tree(mesh_manager, volume);
    // set up point location tree for any volumetric elements. TODO - currently not supported with DeePeeRT
    TreeID element_tree = create_element_tree(mesh_manager, volume);
    return {faces_tree, element_tree}; // return TREE_NONE for element tree until implmemented
  }

  SurfaceTreeID
  DeePeeRTRayTracer::create_surface_tree(const std::shared_ptr<MeshManager>& mesh_manager,    
                                       MeshID volume)
  {
    warning("Surface trees not currently supported with DeePeeRT ray tracer");
    return TREE_NONE;
  };

  ElementTreeID

}