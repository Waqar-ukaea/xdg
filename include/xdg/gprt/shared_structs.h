#include "gprt.h"
#include "../device_ray.h"
#include "../shared_enums.h"
#include "../geometry/dp_math.h"

struct GPRTPrimitiveRef
{
  int id; // ID of the primitive
  int sense;
};

struct dblRay 
{
  double3 origin;
  double3 direction;
  double tMin; // Minimum distance for ray intersection
  double tMax; // Maximum distance for ray intersection
  int32_t* exclude_primitives; // Optional for excluding primitives
  int32_t exclude_count;           // Number of excluded primitives
  xdg::HitOrientation hitOrientation;
  int volume_tree; // TreeID of the volume being queried
  SurfaceAccelerationStructure volume_accel; // The volume accel 
};

struct dblHit 
{
  double distance;
  int surf_id;
  int primitive_id;
  xdg::PointInVolume piv; // Point in volume check result (0 for outside, 1 for inside)
};

/* Shader data associated with native single-precision triangle geometry. */
struct TriangleGeomData {
  float3 *normals;
  int surf_id;
  int forward_vol;
  int reverse_vol;
  dblRay *ray;
  xdg::XDGRayHit *ray_hits;
  int forward_tree;
  int reverse_tree;
  GPRTPrimitiveRef* primitive_refs;
  xdg::SurfaceBoundaryCondition boundary_condition;
};

struct dblRayGenData {
  dblRay *ray;
  dblHit *hit;
};

struct XDGRayHitRayGenData {
  xdg::XDGRayHit *ray_hits;
  SurfaceAccelerationStructure *volume_accels;
  int volume_accel_count;
};

/* A small structure of constants that can change every frame without rebuilding the
  shader binding table. (must be 128 bytes or less) */

struct RayFirePushConstants {
  xdg::HitOrientation hitOrientation;
  int batch_mode;
};
