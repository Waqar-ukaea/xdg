#include "xdg/gprt/ray_tracer.h"
#include "xdg/gprt/moab_util.h"

namespace xdg {

GPRTRayTracer::GPRTRayTracer()
{
  context_ = gprtContextCreate();
}

GPRTRayTracer::~GPRTRayTracer()
{
  gprtContextDestroy(context_);
}

// Ray tracer interface stub methods to be implemented
void GPRTRayTracer::init() {
  // TODO: Init GPRT context and modules
}

// void create_gprt_geoms(const std::shared_ptr<MeshManager> mesh_manager,
//                        MeshID volume_id)
// {
//   // loop over surfaes in mesh manager
//   for (auto surface : mesh_manager)
//   {
//     // flat_vertices = mesh_manager->get_surface_vertices(surface);
//     // flat_indices = mesh_manager->get_surface_faces(surface);  
//     // std::vector<T> vertices;
//     // std::vector<uint3> indices;
//     // vertex buffer -> gprtDeviceBufferCreate<float3>(context, vertices.size(), vertices);
//     // connectivity_buffer -> gprtDeviceBufferCreate<uint3>(context, NUM_INDICES, indices);
//     // gprt_geoms.push_back(gprtGeomCreate<TrianglesGeomData>(context, trianglesGeomType));
//   }
// }

// // We create the BLAS for the triangles in a given volume and the TLAS instance for that BLAS
// void create_accel_from_vol(MeshID volume)
// {
//   // get gprt_geoms from the volume
//   //
// //   blas_ = gprtTriangleAccelCreate(context_, gprt_geoms.size(), gprt_geoms.data());
// //   gprtAccelBuild(context, blas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
// //   tlas_ = gprtInstanceAccelCreate(context_, 1, &blas_);
// //   gprtAccelBuild(context, tlas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
// }



TreeID GPRTRayTracer::register_volume(const std::shared_ptr<MeshManager> mesh_manager, MeshID volume) {
  /*
  Loop through all surfaces in volume
  */
  // TODO: Register GPRT geometry
  // Create BLAS for triangles in the volume
  // Create TLAS instance for the BLAS
  // Store the geometry in the tree_to_accel_map_

  return {}; // placeholder
}

bool GPRTRayTracer::point_in_volume(TreeID scene,
                                     const Position& point,
                                     const Direction* direction,
                                     const std::vector<MeshID>* exclude_primitives) const {
  // TODO: Point containment logic
  return false;
}

std::pair<double, MeshID> GPRTRayTracer::ray_fire(TreeID scene,
                                                  const Position& origin,
                                                  const Direction& direction,
                                                  const double dist_limit,
                                                  HitOrientation orientation,
                                                  std::vector<MeshID>* const exclude_primitives) {
  // TODO: Ray cast logic
  return {0.0, 0};
}

void GPRTRayTracer::closest(TreeID scene,
                            const Position& origin,
                            double& dist,
                            MeshID& triangle) {
  // TODO: Closest hit logic with triangle
  dist = -1.0;
  triangle = -1;
}

void GPRTRayTracer::closest(TreeID scene,
                            const Position& origin,
                            double& dist) {
  // TODO: Closest hit logic
  dist = -1.0;
}

bool GPRTRayTracer::occluded(TreeID scene,
                             const Position& origin,
                             const Direction& direction,
                             double& dist) const {
  // TODO: Occlusion logic
  dist = -1.0;
  return false;
}

} // namespace xdg
