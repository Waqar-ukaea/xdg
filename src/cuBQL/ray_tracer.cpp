#include "xdg/cuBQL/ray_tracer.h"
#include "xdg/error.h"

#include <omp.h>
#include "cuBQL/builder/omp.h"
#include "cuBQL/math/Ray.h"
#include "cuBQL/queries/triangleData/Triangle.h"
#include "cuBQL/queries/triangleData/math/rayTriangleIntersections.h"
#include "cuBQL/traversal/rayQueries.h"

namespace xdg {

struct CuBQLPluckerIntersectionResult {
  bool hit {false};
  double t {0.0};
};

static inline __cubql_both double cubql_ray_hit_tolerance(double t)
{
  constexpr double tolerance = 64.0 * 2.2204460492503131e-16;
  return tolerance * (1.0 + cuBQL::abst(t));
}

static inline __cubql_both bool cubql_plucker_first(cuBQL::vec3d a,
                                                    cuBQL::vec3d b)
{
  if (a.x < b.x) return true;
  if (a.x > b.x) return false;

  if (a.y < b.y) return true;
  if (a.y > b.y) return false;

  return a.z < b.z;
}

static inline __cubql_both double cubql_plucker_edge_test(cuBQL::vec3d vertex_a,
                                                          cuBQL::vec3d vertex_b,
                                                          cuBQL::vec3d ray,
                                                          cuBQL::vec3d ray_normal)
{
  double pip;
  if (cubql_plucker_first(vertex_a, vertex_b)) {
    const cuBQL::vec3d edge = vertex_b - vertex_a;
    const cuBQL::vec3d edge_normal = cross(edge, vertex_a);
    pip = dot(ray, edge_normal) + dot(ray_normal, edge);
  } else {
    const cuBQL::vec3d edge = vertex_a - vertex_b;
    const cuBQL::vec3d edge_normal = cross(edge, vertex_b);
    pip = dot(ray, edge_normal) + dot(ray_normal, edge);
    pip = -pip;
  }

  constexpr double dbl_zero_tol = 20.0 * 2.2204460492503131e-16;
  if (cuBQL::abst(pip) < dbl_zero_tol) {
    pip = 0.0;
  }
  return pip;
}

static inline __cubql_both CuBQLPluckerIntersectionResult
cubql_plucker_ray_tri_intersect(cuBQL::vec3d vertices[3],
                                cuBQL::vec3d origin,
                                cuBQL::vec3d direction,
                                double t_max,
                                double t_min)
{
  const cuBQL::vec3d ray_a = direction;
  const cuBQL::vec3d ray_b = cross(direction, origin);

  const double plucker_coord_0 =
    cubql_plucker_edge_test(vertices[0], vertices[1], ray_a, ray_b);

  const double plucker_coord_1 =
    cubql_plucker_edge_test(vertices[1], vertices[2], ray_a, ray_b);

  if ((0.0 < plucker_coord_0 && 0.0 > plucker_coord_1) ||
      (0.0 > plucker_coord_0 && 0.0 < plucker_coord_1)) {
    return {};
  }

  const double plucker_coord_2 =
    cubql_plucker_edge_test(vertices[2], vertices[0], ray_a, ray_b);

  if ((0.0 < plucker_coord_1 && 0.0 > plucker_coord_2) ||
      (0.0 > plucker_coord_1 && 0.0 < plucker_coord_2) ||
      (0.0 < plucker_coord_0 && 0.0 > plucker_coord_2) ||
      (0.0 > plucker_coord_0 && 0.0 < plucker_coord_2)) {
    return {};
  }

  if (plucker_coord_0 == 0.0 &&
      plucker_coord_1 == 0.0 &&
      plucker_coord_2 == 0.0) {
    return {};
  }

  const double inverse_sum =
    1.0 / (plucker_coord_0 + plucker_coord_1 + plucker_coord_2);

  const cuBQL::vec3d intersection =
    plucker_coord_0 * inverse_sum * vertices[2] +
    plucker_coord_1 * inverse_sum * vertices[0] +
    plucker_coord_2 * inverse_sum * vertices[1];

  int idx = 0;
  double max_abs_dir = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (cuBQL::abst(direction[i]) > max_abs_dir) {
      idx = i;
      max_abs_dir = cuBQL::abst(direction[i]);
    }
  }

  double dist_out = (intersection[idx] - origin[idx]) / direction[idx];

  const double u = plucker_coord_2 * inverse_sum;
  const double v = plucker_coord_0 * inverse_sum;
  if (u < 0.0 || v < 0.0 || (u + v) > 1.0) {
    return {};
  }

  if (dist_out < t_min || dist_out > t_max) {
    return {};
  }

  return {true, dist_out};
}

CuBQLRayTracer::CuBQLRayTracer() = default;

CuBQLRayTracer::~CuBQLRayTracer()
{
  for (auto& surface_bvh : surface_bvhs_) {
    cuBQL::omp::Context context(surface_bvh.gpu_id);

    if (surface_bvh.bvh.nodes || surface_bvh.bvh.primIDs) {
      cuBQL::omp::freeBVH(surface_bvh.bvh, &context);
    }

    if (surface_bvh.d_vertices) {
      omp_target_free(surface_bvh.d_vertices, surface_bvh.gpu_id);
    }
    if (surface_bvh.d_indices) {
      omp_target_free(surface_bvh.d_indices, surface_bvh.gpu_id);
    }
    if (surface_bvh.d_normals) {
      omp_target_free(surface_bvh.d_normals, surface_bvh.gpu_id);
    }
    if (surface_bvh.d_primitive_refs) {
      omp_target_free(surface_bvh.d_primitive_refs, surface_bvh.gpu_id);
    }
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
  auto& surface_bvh_indices = tree_to_surface_bvh_indices_[tree];
  auto volume_surfaces = mesh_manager->get_volume_surfaces(volume_id);

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
    std::vector<MeshID> h_primitive_refs;
    h_primitive_refs.reserve(num_faces);

    std::vector<cuBQL::vec3d> h_normals;
    h_normals.reserve(num_faces);

    auto [forward_parent, reverse_parent] = mesh_manager->get_parent_volumes(surf);
    for (const auto& face : mesh_manager->get_surface_faces(surf)) {
      h_primitive_refs.push_back(face);

      auto normal = mesh_manager->face_normal(face);
      if (volume_id == reverse_parent) {
        normal = -normal;
      } else if (volume_id != forward_parent) {
        fatal_error("Volume {} is not a parent of surface {}", volume_id, surf);
      }
      h_normals.emplace_back(normal.x, normal.y, normal.z);
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

    auto* d_normals = static_cast<cuBQL::vec3d*>
      (omp_target_alloc(h_normals.size() * sizeof(cuBQL::vec3d), gpu_id));
    omp_target_memcpy(d_normals, h_normals.data(),
      h_normals.size() * sizeof(cuBQL::vec3d), 0, 0, gpu_id, host_id);

    auto* d_primitive_refs = static_cast<MeshID*>
      (omp_target_alloc(h_primitive_refs.size() * sizeof(MeshID), gpu_id));
    omp_target_memcpy(d_primitive_refs, h_primitive_refs.data(),
      h_primitive_refs.size() * sizeof(MeshID), 0, 0, gpu_id, host_id);

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

    CuBQLSurfaceBVH surface_bvh;
    surface_bvh.surface = surf;
    surface_bvh.bvh = bvh;
    surface_bvh.d_vertices = d_vertices;
    surface_bvh.d_indices = d_indices;
    surface_bvh.d_normals = d_normals;
    surface_bvh.d_primitive_refs = d_primitive_refs;
    surface_bvh.num_vertices = h_vertices.size();
    surface_bvh.num_faces = num_faces;
    surface_bvh.gpu_id = gpu_id;

    surface_bvh_indices.push_back(surface_bvhs_.size());
    surface_bvhs_.push_back(surface_bvh);
  }
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
CuBQLRayTracer::ray_fire(TreeID tree,
                         const Position& origin,
                         const Direction& direction,
                         const double tmax,
                         HitOrientation hitOrientation,
                         std::vector<MeshID>* const exclude_primitives)
{
  int gpu_id = 0; // TODO - how to manage GPU IDs in a multi-GPU system?
  int host_id = omp_get_initial_device();

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
                      host_id);
  }

  double closest_distance = tmax;
  MeshID closest_surface = ID_NONE;
  MeshID closest_primitive = ID_NONE;
  auto* d_surface_hit = static_cast<CuBQLRayHit*>
    (omp_target_alloc(sizeof(CuBQLRayHit), gpu_id));

  const auto& surface_bvh_indices = tree_to_surface_bvh_indices_.at(tree);
  for (const auto surface_bvh_index : surface_bvh_indices) {
    const auto& surface_bvh = surface_bvhs_.at(surface_bvh_index);

    cuBQL::bvh3d bvh = surface_bvh.bvh;
    const cuBQL::vec3d* d_vertices = surface_bvh.d_vertices;
    const cuBQL::vec3i* d_indices = surface_bvh.d_indices;
    const cuBQL::vec3d* d_normals = surface_bvh.d_normals;
    const MeshID* d_primitive_refs = surface_bvh.d_primitive_refs;

    CuBQLRayHit surface_hit;
    surface_hit.distance = closest_distance + cubql_ray_hit_tolerance(closest_distance);
    omp_target_memcpy(d_surface_hit,
                      &surface_hit,
                      sizeof(CuBQLRayHit),
                      0,
                      0,
                      gpu_id,
                      host_id);

    const int orientation = static_cast<int>(hitOrientation);
    const cuBQL::vec3d ray_origin(origin.x, origin.y, origin.z);
    const cuBQL::vec3d ray_direction(direction.x, direction.y, direction.z);

    #pragma omp target device(gpu_id) \
      is_device_ptr(d_vertices, d_indices, d_normals, d_primitive_refs, d_exclude_primitives, d_surface_hit)
    {
      cuBQL::ray3d ray(ray_origin, ray_direction, 0.0, d_surface_hit->distance);

      auto intersect_prim = [=, &ray]
        (uint32_t prim_id) -> double
      {
        const MeshID primitive_ref = d_primitive_refs[prim_id];

        for (int i = 0; i < exclude_count; ++i) {
          if (d_exclude_primitives[i] == primitive_ref) {
            return ray.tMax;
          }
        }

        const cuBQL::vec3i index = d_indices[prim_id];
        cuBQL::vec3d vertices[3] = {
          d_vertices[index.x],
          d_vertices[index.y],
          d_vertices[index.z]
        };

        const double normal_dot_direction = dot(d_normals[prim_id], ray.direction);
        bool culled = false;
        if (orientation == static_cast<int>(HitOrientation::EXITING)) {
          culled = normal_dot_direction < 0.0;
        } else if (orientation == static_cast<int>(HitOrientation::ENTERING)) {
          culled = normal_dot_direction >= 0.0;
        }
        if (culled) {
          return ray.tMax;
        }

        auto intersection = cubql_plucker_ray_tri_intersect(vertices,
                                                            ray.origin,
                                                            ray.direction,
                                                            ray.tMax,
                                                            ray.tMin);
        if (intersection.hit) {
          d_surface_hit->distance = intersection.t;
          d_surface_hit->primitive = primitive_ref;
          ray.tMax = intersection.t;
        }

        return ray.tMax;
      };

      cuBQL::shrinkingRayQuery::forEachPrim(intersect_prim, bvh, ray);
    }

    omp_target_memcpy(&surface_hit,
                      d_surface_hit,
                      sizeof(CuBQLRayHit),
                      0,
                      0,
                      host_id,
                      gpu_id);

    const double hit_tolerance = cubql_ray_hit_tolerance(closest_distance);
    const bool closer_hit = surface_hit.distance < closest_distance - hit_tolerance;
    const bool tied_hit = cuBQL::abst(surface_hit.distance - closest_distance) <= hit_tolerance;
    const bool lower_surface_tie = closest_surface == ID_NONE ||
      surface_bvh.surface < closest_surface;

    if (surface_hit.primitive != ID_NONE && (closer_hit || (tied_hit && lower_surface_tie))) {
      closest_distance = surface_hit.distance;
      closest_surface = surface_bvh.surface;
      closest_primitive = surface_hit.primitive;
    }
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
