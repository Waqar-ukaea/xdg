#include "xdg/cuBQL/ray_tracer.h"
#include "xdg/error.h"

#include <omp.h>
#include "cuBQL/builder/omp.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"
namespace xdg {

CuBQLRayTracer::CuBQLRayTracer() = default;

CuBQLRayTracer::~CuBQLRayTracer() = default;

void CuBQLRayTracer::init()
{
}

std::pair<TreeID, TreeID>
CuBQLRayTracer::register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                                MeshID volume)
{
  TreeID surface_tree = create_surface_tree(mesh_manager, volume);
  TreeID element_tree = create_element_tree(mesh_manager, volume);
  return {surface_tree, element_tree};
}

TreeID
CuBQLRayTracer::create_surface_tree(const std::shared_ptr<MeshManager>& mesh_manager,
                                    MeshID volume_id)
{

  int gpu_id = 0; // TODO - how to manage GPU IDs in a multi-GPU system?
  int host_id = omp_get_initial_device();


  // cuBQL::box3d *d_boxes = nullptr; // box3d is an alias for box_t<double, 3>
  // int num_boxes = 0;

  // cuBQL::bvh3d bvh; // bvh3d is an alias for BinaryBVH<double, 3> 
  // // bvh_t is an alias for BinaryBVH<T, D>, so bvh_t<double, 3> is also BinaryBVH<double, 3>

  cuBQL::BuildConfig buildParams; // It looks like spatial median is the only supported method for omp builder
  // // Looks like the cuBQL::gpuBuilder is only pulled in when cubql is built with CUDA support 
  
  // cuBQL::build_omp_target(bvh, d_boxes, num_boxes, buildParams, gpu_id); // build bvh on GPU 0 using OpenMP target offloading

  SurfaceTreeID tree = next_surface_tree_id();
  surface_trees_.push_back(tree);
  auto volume_surfaces = mesh_manager->get_volume_surfaces(volume_id);

  for (const auto &surf : volume_surfaces) {
    cuBQL::bvh3d bvh; // bvh3d is an alias for BinaryBVH<double, 3> 

    auto num_faces = mesh_manager->num_surface_faces(surf);
    auto vertices = mesh_manager->get_surface_vertices(surf);
    auto indices = mesh_manager->get_surface_connectivity(surf);
    
    // Get host side storage for vertices using cubql friendly types 
    std::vector<cuBQL::vec3d> h_vertices;
    h_vertices.reserve(vertices.size());
    for (const auto& vertex : vertices) {
      h_vertices.emplace_back(vertex.x, vertex.y, vertex.z);
    }

    // Get host side storage for indices using cubql friendly types
    std::vector<cuBQL::vec3i> h_indices;
    h_indices.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) {
      h_indices.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
    }

    // copy vertices and indices to device
    auto* d_vertices = static_cast<cuBQL::vec3d*>
      (omp_target_alloc(h_vertices.size() * sizeof(cuBQL::vec3d), gpu_id));
    omp_target_memcpy(d_vertices, h_vertices.data(), 
      h_vertices.size() * sizeof(cuBQL::vec3d), 0, 0, gpu_id, host_id);

    auto* d_indices = static_cast<cuBQL::vec3i*>
      (omp_target_alloc(h_indices.size() * sizeof(cuBQL::vec3i), gpu_id));
    omp_target_memcpy(d_indices, h_indices.data(), 
      h_indices.size() * sizeof(cuBQL::vec3i), 0, 0, gpu_id, host_id);

    // Create device storage for triangle AABBs to be computed in parallel on GPU
    auto* d_aabbs = static_cast<cuBQL::box3d*>
      (omp_target_alloc(h_indices.size() * sizeof(cuBQL::box3d), gpu_id));

    // Create AABBs for each triangle in parallel on GPU
    // TODO - Should this be its own function?
    // TODO - How can this be extended for tets and other element types?
    #pragma omp target device(gpu_id) is_device_ptr(d_vertices, d_indices, d_aabbs)
    #pragma omp teams distribute parallel for
    for (uint32_t primID = 0; primID < h_indices.size(); ++primID) {
      cuBQL::vec3i indices = d_indices[primID];

      cuBQL::vec3d A = d_vertices[indices.x];
      cuBQL::vec3d B = d_vertices[indices.y];
      cuBQL::vec3d C = d_vertices[indices.z];

      cuBQL::box3d aabb;
      aabb.extend(A);
      aabb.extend(B);
      aabb.extend(C);

      d_aabbs[primID] = aabb;
    } // This is our AABB population kernel written with OpenMP target offloading pragmas to run in parallel on the GPU

    // Construct the bvh on the gpu using the AABBs with openmp pathway
    cuBQL::build_omp_target(bvh, d_aabbs, num_faces, buildParams, gpu_id); 
    surface_bvhs_.push_back(bvh); // Store the BVH for this surface tree
  }
  return TREE_NONE;
}

TreeID
CuBQLRayTracer::create_element_tree(const std::shared_ptr<MeshManager>&,
                                    MeshID)
{
  fatal_error("Element trees not currently supported with cuBQL ray tracer");
  return TREE_NONE;
}

void CuBQLRayTracer::create_global_surface_tree()
{
  fatal_error("Global surface trees not currently supported with cuBQL ray tracer");
}

void CuBQLRayTracer::create_global_element_tree()
{
  fatal_error("Global element trees not currently supported with cuBQL ray tracer");
}

MeshID CuBQLRayTracer::find_element(const Position&) const
{
  fatal_error("Element queries not currently supported with cuBQL ray tracer");
  return ID_NONE;
}

MeshID CuBQLRayTracer::find_element(TreeID, const Position&) const
{
  fatal_error("Element queries not currently supported with cuBQL ray tracer");
  return ID_NONE;
}

bool CuBQLRayTracer::point_in_volume(TreeID,
                                     const Position&,
                                     const Direction*,
                                     const std::vector<MeshID>*) const
{
  fatal_error("Point-in-volume queries not currently supported with cuBQL ray tracer");
  return false;
}

std::pair<double, MeshID>
CuBQLRayTracer::ray_fire(TreeID,
                         const Position& origin,
                         const Direction& direction,
                         const double tmax,
                         HitOrientation hitOrientation,
                         std::vector<MeshID>* const)
{

  // For a closest hit query this is the traversal template we want to make use of
  cuBQL::shrinkingRadiusQuery::forEachPrim(prim_lambda, node_lambda, surface_bvhs_[0], tmax)

  fatal_error("Ray-fire queries not currently supported with cuBQL ray tracer");
  return {INFTY, ID_NONE};
}

std::pair<double, MeshID>
CuBQLRayTracer::closest(TreeID, const Position&)
{
  fatal_error("Closest queries not currently supported with cuBQL ray tracer");
  return {INFTY, ID_NONE};
}

bool CuBQLRayTracer::occluded(TreeID,
                              const Position&,
                              const Direction&,
                              double&) const
{
  fatal_error("Occlusion queries not currently supported with cuBQL ray tracer");
  return false;
}

} // namespace xdg
