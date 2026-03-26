#include "xdg/DPRT/ray_tracer.h"
#include <omp.h>


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

  if (volume_surfaces.empty()) {
    fatal_error("Volume {} has no surfaces to register in DPRT.", volume_id);
  }

  std::vector<DPRTTriangles> surface_meshes_list;
  surface_meshes_list.reserve(volume_surfaces.size());

  for (const auto &surf : volume_surfaces) {
    auto vertices = mesh_manager->get_surface_vertices(surf);
    auto indices = mesh_manager->get_surface_connectivity(surf);

    if (vertices.empty()) {
      fatal_error("Surface {} for volume {} has no vertices.", surf, volume_id);
    }
    if (indices.size() % 3 != 0) {
      fatal_error("Surface {} for volume {} has {} connectivity entries; expected a multiple of 3 for triangles.",
                  surf, volume_id, indices.size());
    }
    
    std::vector<DPRTvec3> vertexArray;
    vertexArray.reserve(vertices.size());
    for (const auto &vertex : vertices) {
      vertexArray.push_back({vertex.x, vertex.y, vertex.z});
    }
    
    std::vector<DPRTint3> indexArray;
    indexArray.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) {
      const int i0 = indices[i];
      const int i1 = indices[i + 1];
      const int i2 = indices[i + 2];
      const int vertex_count = static_cast<int>(vertices.size());
      if (i0 < 0 || i0 >= vertex_count ||
          i1 < 0 || i1 >= vertex_count ||
          i2 < 0 || i2 >= vertex_count) {
        fatal_error("Surface {} for volume {} has out-of-range triangle indices ({}, {}, {}) with {} surface vertices.",
                    surf, volume_id, i0, i1, i2, vertices.size());
      }
      indexArray.push_back(
        DPRTint3{i0, i1, i2}
      );
    }
    
    // Create a DPRT triangle mesh for this surface (BLAS)
    DPRTTriangles surface_mesh = dprtCreateTriangles(context_, surf, vertexArray.data(), vertexArray.size(), indexArray.data(), indexArray.size());
    surface_meshes_list.push_back(surface_mesh);
  }

  // Create a DPRT group for this volume, containing all of its surface meshes
  DPRTGroup volume_group = dprtCreateTrianglesGroup(context_, surface_meshes_list.data(), surface_meshes_list.size());
  // // Create a DPRT model for this volume, with instances for each surface (TLAS)
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

bool DPRTRayTracer::point_in_volume(TreeID tree,
                                    const Position& point,
                                    const Direction* direction,
                                    const std::vector<MeshID>* exclude_primitives) const
{
  // TODO: DPRT does not currently support exclude_primitives in this query path.

  auto model = surface_tree_to_model_.at(tree);

  Direction directionUsed = (direction != nullptr) ? Direction{direction->x, direction->y, direction->z} 
                            : Direction{1. / std::sqrt(2.0), 1. / std::sqrt(2.0), 0.0};

/*
  DPRT also does not expose programmable hit/intersection shaders, so we
  cannot inject custom point-in-volume logic into traversal.

  As a workaround, we trace the same ray twice:
  - `DPRT_CULL_BACK` keeps only entering intersections
  - `DPRT_CULL_FRONT` keeps only exiting intersections

  We then classify the point by comparing the nearest entering and exiting hits:
  - neither hit: outside
  - exiting hit only: inside
  - entering hit only: outside
  - both hit: inside if `exitHit.t < enterHit.t`
*/

  DPRTHit hits[2]{};
  for (auto& h : hits) {
    h.primID = -1;
    h.instID = -1;
    h.geomUserData = 0;
    h.t = INFTY;
  }

  // We can however use the same ray with different rayflags applied for culling entering vs exiting hits
  DPRTRay ray{};
  ray.origin = {point[0], point[1], point[2]};
  ray.direction = {directionUsed[0], directionUsed[1], directionUsed[2]};
  ray.tMax = INFTY;
  ray.tMin = -1e-12; // small negative tolerance to avoid self intersections in PIV queries (tests on boundaries fail otherwise :/)
  // TODO - is the same negative tolerance needed for ray_fire??

  const int host_device = omp_get_initial_device();
  const int gpu_device = 0;

  auto* d_ray = static_cast<DPRTRay*>(omp_target_alloc(sizeof(DPRTRay), gpu_device));
  auto* d_hits = static_cast<DPRTHit*>(omp_target_alloc(2 * sizeof(DPRTHit), gpu_device));

  omp_target_memcpy(d_ray, &ray, sizeof(DPRTRay), 0, 0, gpu_device, host_device);
  omp_target_memcpy(d_hits, hits, 2 * sizeof(DPRTHit), 0, 0, gpu_device, host_device);

  constexpr int numRays = 1;
  DPRTHit* d_enterHit = d_hits;
  DPRTHit* d_exitHit = d_hits + 1;

  dprtTrace(model, d_ray, d_enterHit, numRays, DPRT_CULL_BACK);  // nearest entering hit
  dprtTrace(model, d_ray, d_exitHit, numRays, DPRT_CULL_FRONT);  // nearest exiting hit

  omp_target_memcpy(hits, d_hits, 2 * sizeof(DPRTHit), 0, 0, host_device, gpu_device);

  // Recover the two hit structs
  DPRTHit& enterHit = hits[0];
  DPRTHit& exitHit  = hits[1];

  omp_target_free(d_ray, gpu_device);
  omp_target_free(d_hits, gpu_device);

  const bool hasEnter = enterHit.primID != -1; // check if entering ray returns a hit
  const bool hasExit = exitHit.primID != -1; // check if exiting ray returns a hit

  if (!hasEnter && !hasExit) return false; // neither hit: outside
  if (!hasEnter && hasExit) return true; // exiting hit only: inside
  if (hasEnter && !hasExit) return false; // entering hit only: outside
  return exitHit.t < enterHit.t; // both hit: inside if `exitHit.t < enterHit.t`
}

std::pair<double, MeshID> DPRTRayTracer::ray_fire(SurfaceTreeID tree,
                                                  const Position& origin,
                                                  const Direction& direction,
                                                  const double dist_limit,
                                                  HitOrientation orientation,
                                                  std::vector<MeshID>* const exclude_primitves)
{
  // TODO: DPRT does not currently support exclude_primitives in this query path.

  auto model = surface_tree_to_model_.at(tree);
  DPRTRay ray;
  ray.origin = {origin[0], origin[1], origin[2]};
  ray.direction = {direction[0], direction[1], direction[2]};
  ray.tMax = dist_limit;
  ray.tMin = 0.0;
  
  DPRTHit hit;
  hit.instID = -1;
  hit.primID = -1;
  hit.geomUserData = 0;
  hit.t = INFTY;
  hit.u = 0.0;
  hit.v = 0.0;

  int64_t rayFlags = DPRT_FLAGS_NONE; 
  if (orientation == HitOrientation::ENTERING)
    rayFlags = DPRT_CULL_BACK;
  else if (orientation == HitOrientation::EXITING) 
    rayFlags = DPRT_CULL_FRONT;

  const int host_device = omp_get_initial_device();
  const int gpu_device = 0;
  auto* d_ray = static_cast<DPRTRay*>(omp_target_alloc(sizeof(DPRTRay), gpu_device));
  auto* d_hit = static_cast<DPRTHit*>(omp_target_alloc(sizeof(DPRTHit), gpu_device));
  
  omp_target_memcpy(d_ray, &ray, sizeof(DPRTRay), 0, 0, gpu_device, host_device);
  omp_target_memcpy(d_hit, &hit, sizeof(DPRTHit), 0, 0, gpu_device, host_device);

  constexpr int numRays = 1;

  dprtTrace(model, d_ray, d_hit, numRays, rayFlags);

  omp_target_memcpy(&hit, d_hit, sizeof(DPRTHit), 0, 0, host_device, gpu_device);
  omp_target_free(d_ray, gpu_device);
  omp_target_free(d_hit, gpu_device);

  MeshID surface_hit = static_cast<MeshID>(hit.geomUserData);
  if (surface_hit == 0) 
    surface_hit = ID_NONE;

  return {hit.t, surface_hit};
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

// Assumes rays populated by an external application
void DPRTRayTracer::batch_ray_fire(TreeID tree,
                                   DPRTRay* d_rays,
                                   DPRTHit* d_hits,
                                   size_t num_rays)
{
  if (num_rays == 0) {
    warning("Warning number of rays passed to ray_fire is 0. No work to be done."); 
    return;
  } 
  auto model = surface_tree_to_model_.at(tree);
  dprtTrace(model, d_rays, d_hits, static_cast<int>(num_rays)); // Launch rays against the model
}

} // namespace xdg
