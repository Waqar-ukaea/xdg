#include <algorithm> // for find

#include "xdg/closest.h"
#include "xdg/triangle_ref.h"
#include "xdg/geometry_data.h"
#include "xdg/plucker.h"
#include "xdg/ray.h"

namespace xdg
{

bool orientation_cull(const Direction& ray_dir, const Direction& normal, HitOrientation orientation)
{
  double dot_prod = ray_dir.dot(normal);
  if (orientation == HitOrientation::EXITING) {
    if (dot_prod < 0.0) return true;
  }
  else if (orientation == HitOrientation::ENTERING) {
    if (dot_prod >= 0.0) return true;
  }
  return false;
}

bool primitive_mask_cull(RTCDRayHit* rayhit) {
  if (!rayhit->ray.exclude_primitives) return false;

  RTCDRay& ray = rayhit->ray;
  RTCDHit& hit = rayhit->hit;

  // if the primitive mask is set, cull if the primitive is not in the mask
  return std::find(ray.exclude_primitives->begin(), ray.exclude_primitives->end(), hit.primID) != ray.exclude_primitives->end();
}

void TriangleBoundsFunc(RTCBoundsFunctionArguments* args)
{
  const GeometryUserData* user_data = (const GeometryUserData*)args->geometryUserPtr;
  const MeshManager* mesh_manager = user_data->mesh_manager;

  const TriangleRef& tri_ref = user_data->tri_ref_buffer[args->primID];
  BoundingBox bounds = mesh_manager->element_bounding_box(tri_ref.triangle_id);

  args->bounds_o->lower_x = bounds.min_x - user_data->box_bump;
  args->bounds_o->lower_y = bounds.min_y - user_data->box_bump;
  args->bounds_o->lower_z = bounds.min_z - user_data->box_bump;
  args->bounds_o->upper_x = bounds.max_x + user_data->box_bump;
  args->bounds_o->upper_y = bounds.max_y + user_data->box_bump;
  args->bounds_o->upper_z = bounds.max_z + user_data->box_bump;
}


void TriangleIntersectionFunc(RTCIntersectFunctionNArguments* args) {
  const GeometryUserData* user_data = (const GeometryUserData*)args->geometryUserPtr;
  const MeshManager* mesh_manager = user_data->mesh_manager;

  const TriangleRef& tri_ref = user_data->tri_ref_buffer[args->primID];

  auto vertices = mesh_manager->triangle_vertices(tri_ref.triangle_id);

  RTCDRayHit* rayhit = (RTCDRayHit*)args->rayhit;
  RTCDRay& ray = rayhit->ray;
  RTCDHit& hit = rayhit->hit;

  Position ray_origin = {ray.dorg[0], ray.dorg[1], ray.dorg[2]};
  Direction ray_direction = {ray.ddir[0], ray.ddir[1], ray.ddir[2]};

  // local variable for distance to the triangle intersection
  double plucker_dist;
  bool hit_tri = plucker_ray_tri_intersect(vertices, ray_origin, ray_direction, plucker_dist);

  if (!hit_tri) return;

  if (plucker_dist > rayhit->ray.dtfar) return;

  Direction normal = (vertices[1] - vertices[0]).cross((vertices[2] - vertices[0]));
  normal.normalize();
  // if this is a normal ray fire, flip the normal as needed
  if (tri_ref.sense == Sense::REVERSE && rayhit->ray.rf_type != RayFireType::FIND_VOLUME) normal = -normal;

  if (rayhit->ray.rf_type == RayFireType::VOLUME) {
   if (orientation_cull(rayhit->ray.ddir, normal, rayhit->ray.orientation)) return;
   if (primitive_mask_cull(rayhit)) return;
  }

  // if we've gotten through all of the filters, set the ray information
  rayhit->ray.set_tfar(plucker_dist);
  // zero-out barycentric coords
  rayhit->hit.u = 0.0;
  rayhit->hit.v = 0.0;
  // set the hit information
  rayhit->hit.geomID = args->geomID;
  rayhit->hit.primID = args->primID;
  rayhit->hit.tri_ref = &tri_ref;

  rayhit->hit.dNg[0] = normal[0];
  rayhit->hit.dNg[1] = normal[1];
  rayhit->hit.dNg[2] = normal[2];
}


void TriangleOcclusionFunc(RTCOccludedFunctionNArguments* args) {
  const GeometryUserData* user_data = (const GeometryUserData*) args->geometryUserPtr;
  const MeshManager* mesh_manager = user_data->mesh_manager;
  const TriangleRef& tri_ref = user_data->tri_ref_buffer[args->primID];

  auto vertices = mesh_manager->triangle_vertices(tri_ref.triangle_id);

  // get the double precision ray from the args
  RTCDRay* ray = (RTCDRay*) args->ray;

  double plucker_dist;
  if (plucker_ray_tri_intersect(vertices, ray->dorg, ray->ddir, plucker_dist)) {
    ray->set_tfar(-INFTY);
  }
}

bool TriangleClosestFunc(RTCPointQueryFunctionArguments* args) {
  RTCGeometry g = rtcGetGeometry(*(RTCScene*)args->userPtr, args->geomID);
  // get the array of DblTri's stored on the geometry
  const GeometryUserData* user_data = (const GeometryUserData*) rtcGetGeometryUserData(g);

  const MeshManager* mesh_manager = user_data->mesh_manager;

  const TriangleRef& tri_ref = user_data->tri_ref_buffer[args->primID];
  auto vertices = mesh_manager->triangle_vertices(tri_ref.triangle_id);

  RTCDPointQuery* query = (RTCDPointQuery*) args->query;
  Position p {query->dblx, query->dbly, query->dblz};

  Position result = closest_location_on_triangle(vertices, p);

  double dist = (result - p).length();
  if ( dist < query->dradius) {
    query->radius = dist;
    query->dradius = dist;
    query->primID = args->primID;
    query->geomID = args->geomID;
    return true;
  } else {
    return false;
  }
}

} // namespace xdg


