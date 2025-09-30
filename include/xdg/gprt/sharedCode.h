#include "gprt.h"

#define AA 3 // used for antialiasing

using double3 = std::array<double, 3>;
using double4 = std::array<double, 4>;

/* Inputs for each ray */
struct RayInput {
  float3 origin;
  float3 direction;
  int32_t* exclude_primitives; // Optional for excluding primitives
  uint32_t exclude_count;           // Number of excluded primitives
};

struct dblRayInput 
{
  double3 origin;
  double3 direction;
  int32_t* exclude_primitives; // Optional for excluding primitives
  uint32_t exclude_count;           // Number of excluded primitives
};

struct RayOutput 
{
  float distance;
  uint surf_id;
  float3 normal;
};

struct dblRayOutput 
{
  double distance;
  uint surf_id;
  double3 normal;
};

/* variables for the single precision triangle mesh geometry */
struct TrianglesGeomData {
  float3 *vertex; // vertex buffer
  uint3 *index;   // index buffer
  float3 *normals;
  uint id;        // surface id
  int forward_vol; 
  int reverse_vol;
};

/* variables for double precision triangle mesh geometry */
struct DPTriangleGeomData {
  double3 *vertex; // vertex buffer
  uint3 *index;  // index buffer
  double3 *normals; // normals buffer
  double4 *dprays; // double precision rays
  uint id;       // surface id
  int forward_vol;
  int reverse_vol;
};

struct RayGenData {
  uint* frameBuffer;                     // Optional for debugging or visuals
  SurfaceAccelerationStructure world;    // The top-level accel structure
  RayInput *ray;
  RayOutput *out;
};

struct RayFireData {
  uint* frameBuffer;                     // Optional for debugging or visuals
  SurfaceAccelerationStructure world;    // The top-level accel structure
  RayInput ray;
  RayOutput out;
};

struct dblRayFireData {
  uint* frameBuffer;                     // Optional for debugging or visuals
  SurfaceAccelerationStructure world;    // The top-level accel structure
  dblRayInput ray;
  dblRayOutput out;
};

/* variables for the miss program */
struct MissProgData {
  float3 color0;
  float3 color1;
};

/* A small structure of constants that can change every frame without rebuilding the
  shader binding table. (must be 128 bytes or less) */
struct PushConstants {
  float time;
  float3 scene_center;
  struct Camera {
    float radius;
    float3 pos;
    float3 dir_00;
    float3 dir_du;
    float3 dir_dv;
  } camera;
};

struct RayFirePushConstants {
  float dist_limit;
  int orientation;
};

struct dblRayFirePushConstants {
  double dist_limit;
  int orientation;
};