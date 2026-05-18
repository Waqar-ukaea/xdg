#include "xdg/cuBQL/ray_tracer.h"
#include "xdg/error.h"
#include "xdg/geometry/plucker.h"
#include "xdg/available_device_probe.h"

#include <omp.h>
#include "cuBQL/builder/omp.h"
#include "cuBQL/math/Ray.h"
#include "cuBQL/queries/triangleData/Triangle.h"
#include "cuBQL/queries/triangleData/math/rayTriangleIntersections.h"
#include "cuBQL/traversal/rayQueries.h"

namespace xdg {

CuBQLRayTracer::CuBQLRayTracer()
{
  if (!system_has_omp_target_device()) {
    fatal_error("No OpenMP target capable device found; cannot initialize cuBQL ray tracer.");
  }

  context_.gpuID = 0; // TODO - support selecting among multiple OpenMP target devices.
  context_.hostID = omp_get_initial_device();
}

CuBQLRayTracer::~CuBQLRayTracer()
{
  for (auto& [tree, tlas] : tree_to_volume_tlas_) {
    tlas.release();
  }
}

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

  const auto& context = context_;
  const int gpu_id = context.gpuID;


  // cuBQL::box3d *d_boxes = nullptr; // box3d is an alias for box_t<double, 3>
  // int num_boxes = 0;

  // cuBQL::bvh3d bvh; // bvh3d is an alias for BinaryBVH<double, 3> 
  // // bvh_t is an alias for BinaryBVH<T, D>, so bvh_t<double, 3> is also BinaryBVH<double, 3>

  cuBQL::BuildConfig buildParams; // It looks like spatial median is the only supported method for omp builder
  // // Looks like the cuBQL::gpuBuilder is only pulled in when cubql is built with CUDA support 
  
  // cuBQL::build_omp_target(bvh, d_boxes, num_boxes, buildParams, gpu_id); // build bvh on GPU 0 using OpenMP target offloading

  SurfaceTreeID tree = next_surface_tree_id();
  surface_trees_.push_back(tree);
  surface_tree_to_volume_[tree] = volume_id;
  auto volume_surfaces = mesh_manager->get_volume_surfaces(volume_id);
  std::vector<cuBQL::box3d> h_tlas_boxes;
  std::vector<CuBQLVolumeTLAS::SurfaceInstanceDD> h_surface_instances;
  std::vector<CuBQLSurfaceBLAS> h_surface_blases;
  h_tlas_boxes.reserve(volume_surfaces.size());
  h_surface_instances.reserve(volume_surfaces.size());
  h_surface_blases.reserve(volume_surfaces.size());

  for (const auto &surf : volume_surfaces) {
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

    // Get storage for primitive refs so a hit can be mapped back to a mesh face.
    std::vector<MeshID> h_primitive_refs = mesh_manager->get_surface_faces(surf);

    // copy vertices and indices to device
    auto* d_vertices = static_cast<cuBQL::vec3d*>
      (omp_target_alloc(h_vertices.size() * sizeof(cuBQL::vec3d), gpu_id));
    omp_target_memcpy(d_vertices, h_vertices.data(), 
      h_vertices.size() * sizeof(cuBQL::vec3d), 0, 0, gpu_id, context.hostID);

    auto* d_indices = static_cast<cuBQL::vec3i*>
      (omp_target_alloc(h_indices.size() * sizeof(cuBQL::vec3i), gpu_id));
    omp_target_memcpy(d_indices, h_indices.data(), 
      h_indices.size() * sizeof(cuBQL::vec3i), 0, 0, gpu_id, context.hostID);

    auto* d_primitive_refs = static_cast<MeshID*>
      (omp_target_alloc(h_primitive_refs.size() * sizeof(MeshID), gpu_id));
    omp_target_memcpy(d_primitive_refs, h_primitive_refs.data(),
      h_primitive_refs.size() * sizeof(MeshID), 0, 0, gpu_id, context.hostID);

    // Create device storage for triangle AABBs to be computed in parallel on GPU
    auto* d_aabbs = static_cast<cuBQL::box3d*>
      (omp_target_alloc(h_indices.size() * sizeof(cuBQL::box3d), gpu_id));
    const auto num_primitives = static_cast<uint32_t>(h_indices.size());

    // Create AABBs for each triangle in parallel on GPU
    // TODO - Should this be its own function?
    // TODO - How can this be extended for tets and other element types?
    #pragma omp target device(gpu_id) is_device_ptr(d_vertices, d_indices, d_aabbs)
    #pragma omp teams distribute parallel for
    for (uint32_t primID = 0; primID < num_primitives; ++primID) {
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
    cuBQL::bvh3d bvh; // bvh3d is an alias for BinaryBVH<double, 3>
    cuBQL::build_omp_target(bvh, d_aabbs, num_faces, buildParams, gpu_id); 

    omp_target_free(d_aabbs, gpu_id);

    auto [forward_parent, reverse_parent] = mesh_manager->get_parent_volumes(surf);
    if (volume_id != forward_parent && volume_id != reverse_parent) {
      fatal_error("Volume {} is not a parent of surface {}", volume_id, surf);
    }

    CuBQLSurfaceMesh surface_mesh;
    surface_mesh.surface_id = surf;
    surface_mesh.d_vertices = d_vertices;
    surface_mesh.d_indices = d_indices;
    surface_mesh.d_primitive_refs = d_primitive_refs;
    surface_mesh.num_vertices = h_vertices.size();
    surface_mesh.num_triangles = num_faces;
    surface_mesh.gpu_id = gpu_id;

    CuBQLSurfaceBLAS surface_blas;
    surface_blas.bvh = bvh;
    surface_blas.mesh = surface_mesh;
    surface_blas.num_prims = num_faces;
    surface_blas.gpu_id = gpu_id;

    // Store BLAS bounding boxes to build TLAS
    const auto surface_bounding_box = mesh_manager->surface_bounding_box(surf);
    cuBQL::box3d surface_bounds;
    surface_bounds.lower = cuBQL::vec3d(surface_bounding_box.min_x,
                                        surface_bounding_box.min_y,
                                        surface_bounding_box.min_z);
    surface_bounds.upper = cuBQL::vec3d(surface_bounding_box.max_x,
                                        surface_bounding_box.max_y,
                                        surface_bounding_box.max_z);

    h_tlas_boxes.push_back(surface_bounds);
    h_surface_instances.push_back({surface_blas.get_device_data()});
    h_surface_blases.push_back(surface_blas);
  }

  if (h_surface_instances.empty()) {
    fatal_error("Volume {} has no surfaces; cannot build cuBQL surface tree", volume_id);
  }

  auto* d_tlas_boxes = static_cast<cuBQL::box3d*>
    (omp_target_alloc(h_tlas_boxes.size() * sizeof(cuBQL::box3d), gpu_id));
  omp_target_memcpy(d_tlas_boxes,
                    h_tlas_boxes.data(),
                    h_tlas_boxes.size() * sizeof(cuBQL::box3d),
                    0,
                    0,
                    gpu_id,
                    context.hostID);

  auto* d_surface_instances = static_cast<CuBQLVolumeTLAS::SurfaceInstanceDD*>
    (omp_target_alloc(h_surface_instances.size() * sizeof(CuBQLVolumeTLAS::SurfaceInstanceDD), gpu_id));
  omp_target_memcpy(d_surface_instances,
                    h_surface_instances.data(),
                    h_surface_instances.size() * sizeof(CuBQLVolumeTLAS::SurfaceInstanceDD),
                    0,
                    0,
                    gpu_id,
                    context.hostID);

  cuBQL::BuildConfig tlasBuildParams(1);
  tlasBuildParams.maxAllowedLeafSize = 1;

  CuBQLVolumeTLAS volume_tlas;
  volume_tlas.num_surface_instances = static_cast<uint32_t>(h_surface_instances.size());
  volume_tlas.gpu_id = gpu_id;
  volume_tlas.d_surface_instances = d_surface_instances;
  volume_tlas.surface_blases = std::move(h_surface_blases);
  cuBQL::build_omp_target(volume_tlas.bvh,
                          d_tlas_boxes,
                          volume_tlas.num_surface_instances,
                          tlasBuildParams,
                          gpu_id);

  omp_target_free(d_tlas_boxes, gpu_id);

  tree_to_volume_tlas_.emplace(tree, std::move(volume_tlas));
  
  return tree;
}

TreeID
CuBQLRayTracer::create_element_tree(const std::shared_ptr<MeshManager>&,
                                    MeshID)
{
  warning("Element trees not currently supported with cuBQL ray tracer");
  return TREE_NONE;
}

void CuBQLRayTracer::create_global_surface_tree()
{
  warning("Global surface trees not currently supported with cuBQL ray tracer");
}

void CuBQLRayTracer::create_global_element_tree()
{
  warning("Global element trees not currently supported with cuBQL ray tracer");
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
CuBQLRayTracer::ray_fire(TreeID tree,
                         const Position& origin,
                         const Direction& direction,
                         const double tmax,
                         HitOrientation hitOrientation,
                         std::vector<MeshID>* const exclude_primitives)
{
  const auto& context = context_;
  const int gpu_id = context.gpuID;

  MeshID* d_exclude_primitives = nullptr;
  int exclude_count = 0;
  if (exclude_primitives && !exclude_primitives->empty()) {
    exclude_count = static_cast<int>(exclude_primitives->size());
    d_exclude_primitives = static_cast<MeshID*>
      (omp_target_alloc(exclude_count * sizeof(MeshID), gpu_id));
    omp_target_memcpy(d_exclude_primitives,
                      exclude_primitives->data(),
                      exclude_count * sizeof(MeshID),
                      0,
                      0,
                      gpu_id,
                      context.hostID);
  }

  double closest_distance = tmax;
  MeshID closest_surface = ID_NONE;
  MeshID closest_primitive = ID_NONE;
  auto* d_surface_hit = static_cast<CubqlHit*>
    (omp_target_alloc(sizeof(CubqlHit), gpu_id));

  const CuBQLVolumeTLAS& volume_tlas = tree_to_volume_tlas_.at(tree);
  const auto volume_tlas_dd = volume_tlas.get_device_data();

  CubqlHit surface_hit;
  surface_hit.distance = tmax;
  omp_target_memcpy(d_surface_hit,
                    &surface_hit,
                    sizeof(CubqlHit),
                    0,
                    0,
                    gpu_id,
                    context.hostID);

  const int orientation = static_cast<int>(hitOrientation);
  const cuBQL::vec3d ray_origin(origin.x, origin.y, origin.z);
  const cuBQL::vec3d ray_direction(direction.x, direction.y, direction.z);

  #pragma omp target device(gpu_id) \
    is_device_ptr(d_exclude_primitives, d_surface_hit)
  {
    cuBQL::ray3d world_ray;
    world_ray.origin = ray_origin;
    world_ray.direction = ray_direction;
    world_ray.tMin = 0.0;
    world_ray.tMax = d_surface_hit->distance;

    CuBQLVolumeTLAS::SurfaceInstanceDD surface_instance;

    // lambda to go from tlas->blas
    auto enter_blas = [=, &surface_instance, &world_ray]
      (cuBQL::ray3d& out_ray, cuBQL::bvh3d& out_bvh, int instance_id)
    {
      surface_instance = volume_tlas_dd.surface_instances[instance_id];
      out_ray = world_ray; // No ray transformation because everything is in world coords with xdg
      out_bvh = surface_instance.surface_blas.bvh; // Get the BLAS for the surface instance we hit in the TLAS
    };


    // lambda for primitive intersection
    // - culls excluded primitives
    // - culls backfacing or frontfacing hits based on hitOrientation
    // - calls plucker intersection 
    auto xdg_plucker_intersect_prim = [=, &world_ray, &surface_instance]
      (uint32_t prim_id) -> double
    {
      const CuBQLSurfaceMesh::DD mesh = surface_instance.surface_blas.mesh;
      const MeshID primitive_ref = mesh.primitive_refs[prim_id];

      for (int i = 0; i < exclude_count; ++i) {
        if (d_exclude_primitives[i] == primitive_ref) {
          return world_ray.tMax;
        }
      }

      const cuBQL::vec3i index = mesh.indices[prim_id];
      cuBQL::vec3d vertices[3] = {
        mesh.vertices[index.x],
        mesh.vertices[index.y],
        mesh.vertices[index.z]
      };

      cuBQL::vec3d normal = cuBQL::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
      if (surface_instance.reverse_sense) {
        normal = -normal;
      }

      const double normal_dot_direction = dot(normal, world_ray.direction);
      bool culled = false;
      if (orientation == static_cast<int>(HitOrientation::EXITING)) {
        culled = normal_dot_direction < 0.0;
      } else if (orientation == static_cast<int>(HitOrientation::ENTERING)) {
        culled = normal_dot_direction >= 0.0;
      }
      if (culled) {
        return world_ray.tMax;
      }

      // Reuse same plucker intersection function applied to other ray tracers
      auto intersection = plucker_ray_tri_intersect(vertices,
                                                    world_ray.origin,
                                                    world_ray.direction,
                                                    world_ray.tMax,
                                                    world_ray.tMin,
                                                    false,
                                                    0);
      if (intersection.hit) {
        d_surface_hit->distance = intersection.t;
        d_surface_hit->surface = mesh.surface_id;
        d_surface_hit->primitive = primitive_ref;
        world_ray.tMax = intersection.t;
      }

      return world_ray.tMax;
    };

    // We must provide the lambda to run on exit but it is empty for our use case 
    // since there is no need for cleanup or transforming the ray back to world space
    auto leave_blas = []() -> void {}; 

    // Two level Traversal template from cuBQL 
    cuBQL::shrinkingRayQuery::twoLevel::forEachPrim(enter_blas,
                                                    leave_blas,
                                                    xdg_plucker_intersect_prim,
                                                    volume_tlas_dd.bvh,
                                                    world_ray);
    
  }

  omp_target_memcpy(&surface_hit,
                    d_surface_hit,
                    sizeof(CubqlHit),
                    0,
                    0,
                    context.hostID,
                    gpu_id);
  
  if (surface_hit.primitive != ID_NONE) {
    closest_distance = surface_hit.distance;
    closest_surface = surface_hit.surface;
    closest_primitive = surface_hit.primitive;
  }

  omp_target_free(d_surface_hit, gpu_id);

  if (d_exclude_primitives) {
    omp_target_free(d_exclude_primitives, gpu_id);
  }

  if (closest_surface == ID_NONE) {
    return {INFTY, ID_NONE};
  }

  if (exclude_primitives) {
    exclude_primitives->push_back(closest_primitive);
  }

  return {closest_distance, closest_surface};
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
