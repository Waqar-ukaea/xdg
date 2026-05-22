#include "xdg/cuBQL/intersection.h"
#include "xdg/geometry/plucker.h"

#include <omp.h>

#include "cuBQL/math/Ray.h"
#include "cuBQL/traversal/rayQueries.h"

namespace xdg {

CuBQLSurfaceHit
intersect_surface_tree(const cubql::Context& context,
                       const CuBQLVolumeTLAS& volume_tlas,
                       const CuBQLRay& ray,
                       HitOrientation hit_orientation,
                       const std::vector<MeshID>* exclude_primitives)
{
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

  auto* d_surface_hit = static_cast<CuBQLSurfaceHit*>
    (omp_target_alloc(sizeof(CuBQLSurfaceHit), gpu_id));

  CuBQLSurfaceHit surface_hit;
  surface_hit.distance = ray.tmax;
  omp_target_memcpy(d_surface_hit,
                    &surface_hit,
                    sizeof(CuBQLSurfaceHit),
                    0,
                    0,
                    gpu_id,
                    context.hostID);

  const auto volume_tlas_dd = volume_tlas.get_device_data();
  const int orientation = static_cast<int>(hit_orientation);

  #pragma omp target device(gpu_id) \
    is_device_ptr(d_exclude_primitives, d_surface_hit)
  {
    cuBQL::ray3d world_ray;
    world_ray.origin = ray.origin;
    world_ray.direction = ray.direction;
    world_ray.tMin = ray.tmin;
    world_ray.tMax = d_surface_hit->distance;

    CuBQLVolumeTLAS::SurfaceInstanceDD surface_instance;

    auto enter_blas = [=, &surface_instance, &world_ray]
      (cuBQL::ray3d& out_ray, cuBQL::bvh3d& out_bvh, int instance_id)
    {
      surface_instance = volume_tlas_dd.surface_instances[instance_id];
      out_ray = world_ray;
      out_bvh = surface_instance.surface_blas.bvh;
    };

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

      cuBQL::vec3d normal = cuBQL::cross(vertices[1] - vertices[0],
                                         vertices[2] - vertices[0]);
      if (surface_instance.reverse_sense) {
        normal = -normal;
      }

      const double normal_dot_direction = dot(normal, world_ray.direction);
      if (orientation_cull(normal_dot_direction,
                           static_cast<HitOrientation>(orientation))) {
        return world_ray.tMax;
      }

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
        d_surface_hit->piv = normal_dot_direction > 0.0 ? INSIDE : OUTSIDE;
        world_ray.tMax = intersection.t;
      }

      return world_ray.tMax;
    };

    auto leave_blas = []() -> void {};

    cuBQL::shrinkingRayQuery::twoLevel::forEachPrim(enter_blas,
                                                    leave_blas,
                                                    xdg_plucker_intersect_prim,
                                                    volume_tlas_dd.bvh,
                                                    world_ray);
  }

  omp_target_memcpy(&surface_hit,
                    d_surface_hit,
                    sizeof(CuBQLSurfaceHit),
                    0,
                    0,
                    context.hostID,
                    gpu_id);

  omp_target_free(d_surface_hit, gpu_id);

  if (d_exclude_primitives) {
    omp_target_free(d_exclude_primitives, gpu_id);
  }

  return surface_hit;
}

} // namespace xdg
