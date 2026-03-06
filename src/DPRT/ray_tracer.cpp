#include "xdg/DPRT/ray_tracer.h"
// #include "dprt/dprt.in.h"

namespace xdg {

DPRTRayTracer::DPRTRayTracer() = default;
DPRTRayTracer::~DPRTRayTracer() = default;

void DPRTRayTracer::init()
{
  if (context_ != nullptr) return;
  context_ = dprtContextCreate(DPRT_CONTEXT_GPU, 0); // Create a GPU context using the first available GPU
}

std::pair<TreeID, TreeID>
DPRTRayTracer::register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                               MeshID volume)
{
  TreeID surface_tree = create_surface_tree(mesh_manager, volume);
  TreeID element_tree = create_element_tree(mesh_manager, volume);
  return {surface_tree, element_tree};
}

TreeID DPRTRayTracer::create_surface_tree(const std::shared_ptr<MeshManager>& mesh_manager, MeshID volume_id)
{
  // DPRT geometry creation requires a valid context; ensure it exists even if caller did not invoke init() yet.
  init();

  SurfaceTreeID tree = next_surface_tree_id();
  surface_trees_.push_back(tree);
  auto volume_surfaces = mesh_manager->get_volume_surfaces(volume_id);

  std::vector<DPRTTriangles> surface_meshes_list;

  for (const auto &surf : volume_surfaces) {
    auto vertices = mesh_manager->get_surface_vertices(surf);
    auto indices = mesh_manager->get_surface_connectivity(surf);    
    
    std::vector<DPRTvec3> vertexArray;
    vertexArray.reserve(vertices.size());
    for (const auto &vertex : vertices) {
      vertexArray.push_back({vertex.x, vertex.y, vertex.z});
    }
    
    std::vector<DPRTint3> indexArray;
    indexArray.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) {
      indexArray.push_back(
        DPRTint3{indices[i], indices[i + 1], indices[i + 2]}
      );
    }

    // Create a DPRT triangle mesh for this surface (BLAS)
    DPRTTriangles surface_mesh = dprtCreateTriangles(context_, surf, vertexArray.data(), vertexArray.size(), indexArray.data(), indexArray.size());
    surface_meshes_list.push_back(surface_mesh);
  }

  // Create a DPRT group for this volume, containing all of its surface meshes
  DPRTGroup volume_group = dprtCreateTrianglesGroup(context_, surface_meshes_list.data(), surface_meshes_list.size());
  // Create a DPRT model for this volume, with instances for each surface (TLAS)
  DPRTModel model = dprtCreateModel(context_, &volume_group, nullptr, 1);

  surface_tree_to_model_[tree] = model;
  
  return tree;
}

TreeID DPRTRayTracer::create_element_tree(const std::shared_ptr<MeshManager>& mesh_manager, MeshID volume_id)
{
  warning("DPRT element tree creation is not implemented.");
  return TREE_NONE;
}

void DPRTRayTracer::create_global_surface_tree()
{
  warning("DPRT global surface tree creation is not implemented.");
}

void DPRTRayTracer::create_global_element_tree()
{
  warning("DPRT global element tree creation is not implemented.");
}

bool DPRTRayTracer::point_in_volume(TreeID,
                                    const Position&,
                                    const Direction*,
                                    const std::vector<MeshID>*) const
{
  fatal_error("DPRT point_in_volume() is not implemented.");
}

std::pair<double, MeshID> DPRTRayTracer::ray_fire(TreeID,
                                                  const Position&,
                                                  const Direction&,
                                                  const double,
                                                  HitOrientation,
                                                  std::vector<MeshID>* const)
{
  fatal_error("DPRT ray_fire() is not implemented.");
}

MeshID DPRTRayTracer::find_element(const Position&) const
{
  fatal_error("DPRT find_element() is not implemented.");
}

MeshID DPRTRayTracer::find_element(TreeID, const Position&) const
{
  fatal_error("DPRT find_element(tree, point) is not implemented.");
}

std::pair<double, MeshID> DPRTRayTracer::closest(TreeID, const Position&)
{
  fatal_error("DPRT closest() is not implemented.");
}

bool DPRTRayTracer::occluded(TreeID, const Position&, const Direction&, double&) const
{
  fatal_error("DPRT occluded() is not implemented.");
}

void DPRTRayTracer::dpr_trace(TreeID tree,
                              DPRTRay* rays,
                              DPRTHit* hits,
                              size_t num_rays)
{
  if (num_rays == 0) return;

  auto model = surface_tree_to_model_.at(tree);
  dprtTrace(model, rays, hits, static_cast<int>(num_rays)); // Launch rays against the model
}

// void DPRTRayTracer::ray_fire_prepared(const size_t, const double, HitOrientation)
// {
//   fatal_error("DPRT ray_fire_prepared() is not implemented.");
// }

// void DPRTRayTracer::populate_rays_external(size_t, const RayPopulationCallback&)
// {
//   fatal_error("DPRT populate_rays_external() is not implemented.");
// }

} // namespace xdg
