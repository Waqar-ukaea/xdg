#include "../../include/xdg/gprt/shared_structs.h"
#include "../../include/xdg/geometry/plucker.h"

/* 
For now we have to use relative paths for includes, which is not ideal. If https://github.com/gprt-org/GPRT/pull/82 gets 
merged into GPRT we will be able to more robustly include these headers in the manner below: 

#include "xdg/gprt/shared_structs.h"
#include "xdg/geometry/plucker.h"
*/

[[vk::push_constant]]
dblRayFirePushConstants PC;

struct RayFirePayload {
    double distance;    // Distance to intersection
    int surf_id;     // ID of the surface hit
    SurfaceAccelerationStructure tlas;
    int primitive_id; // ID of the primitive hit
    xdg::PointInVolume piv; // Point in volume check (0 for outside, 1 for inside) 
    int next_volume;
    xdg::SurfaceBoundaryCondition boundary_condition;
    double3 normal;
};


struct DPAttribute
{
  double f64t; // double precision hit distance
  int global_prim_id;
};


[shader("closesthit")]
void ray_fire_hit(uniform DPTriangleGeomData record, inout RayFirePayload payload, in DPAttribute attr) {
    // Distance from the ray origin to the hit point
    uint hit_kind = HitKind();
    uint rayID = DispatchRaysIndex().x;

    // There is some logic for handling next volumes inside the h5m-reader which I could make use of too
    // TODO : Should the dblHit struct return the next volume ID for the ray back to the host

    payload.piv = (hit_kind == HIT_KIND_TRIANGLE_FRONT_FACE) 
    ? xdg::PointInVolume::OUTSIDE
    : xdg::PointInVolume::INSIDE; 

    payload.distance = attr.f64t;
    payload.surf_id = record.surf_id;
    payload.primitive_id = attr.global_prim_id;
    payload.boundary_condition = record.boundary_condition;
    payload.normal = record.normals[PrimitiveIndex()];

    int input_volume;
    if (PC.batch_mode != 0) {
        input_volume = record.ray_hits[rayID].volume;
    } else if (record.ray[rayID].volume_tree == record.reverse_tree) {
        input_volume = record.reverse_vol;
    } else {
        input_volume = record.forward_vol;
    }
    payload.next_volume = input_volume == record.forward_vol
        ? record.reverse_vol
        : record.forward_vol;
}

[shader("miss")]
void ray_fire_miss(inout RayFirePayload payload) {
    // Set the miss payload to default values
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.next_volume = -1;
    payload.boundary_condition = xdg::SurfaceBoundaryCondition::UNSET;
    payload.normal = double3(0.0, 0.0, 0.0);
}

// This ray generation program will kick off the ray tracing process,
// generating rays and tracing them into the world.
[shader("raygeneration")]
void ray_fire(uniform dblRayGenData record, uniform DPTriangleGeomData mesh) {
    RayFirePayload payload;
    uint rayID = DispatchRaysIndex().x;

    // Trace the ray into the scene
    RayDesc rayDesc;
    rayDesc.Origin = float3(record.ray[rayID].origin);
    rayDesc.Direction = normalize(float3(record.ray[rayID].direction));
    rayDesc.TMin = float(record.ray[rayID].tMin); 
    rayDesc.TMax = float(record.ray[rayID].tMax); 

    SurfaceAccelerationStructure world = record.ray[rayID].volume_accel;

    // Pass the ray's origin and direction to the payload
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.tlas = world;

    TraceRay(world, RAY_FLAG_NONE, 0xff, 0, 1, rayDesc, payload);

    // Store the distance to the hit point and the surface ID in buffers for CPU
    record.hit[rayID].distance = payload.distance;
    record.hit[rayID].surf_id = payload.surf_id;
    record.hit[rayID].primitive_id = payload.primitive_id;
}

[shader("raygeneration")]
void ray_fire_batch(uniform XDGRayHitRayGenData record,
                    uniform DPTriangleGeomData mesh) {
    uint rayID = DispatchRaysIndex().x;
    xdg::XDGRayHit ray_hit = record.ray_hits[rayID];

    RayFirePayload payload;
    payload.distance = ray_hit.t_max;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.piv = xdg::PointInVolume::OUTSIDE;
    payload.next_volume = -1;
    payload.boundary_condition = xdg::SurfaceBoundaryCondition::UNSET;
    payload.normal = double3(0.0, 0.0, 0.0);

    const int volume = ray_hit.volume;
    if (volume >= 0 && volume < record.volume_accel_count) {
        SurfaceAccelerationStructure world = record.volume_accels[volume];

        RayDesc rayDesc;
        rayDesc.Origin = float3(float(ray_hit.origin[0]),
                                float(ray_hit.origin[1]),
                                float(ray_hit.origin[2]));
        rayDesc.Direction = normalize(float3(float(ray_hit.direction[0]),
                                             float(ray_hit.direction[1]),
                                             float(ray_hit.direction[2])));
        rayDesc.TMin = float(ray_hit.t_min);
        rayDesc.TMax = float(ray_hit.t_max);

        payload.tlas = world;
        TraceRay(world, RAY_FLAG_NONE, 0xff, 0, 1, rayDesc, payload);
    }

    record.ray_hits[rayID].distance = payload.surf_id == -1
        ? ray_hit.t_max
        : payload.distance;
    record.ray_hits[rayID].surface = payload.surf_id;
    record.ray_hits[rayID].primitive = payload.primitive_id;
    record.ray_hits[rayID].point_in_volume = int(payload.piv);
    record.ray_hits[rayID].next_volume = payload.next_volume;
    record.ray_hits[rayID].boundary_condition = int(payload.boundary_condition);
    record.ray_hits[rayID].normal[0] = payload.normal.x;
    record.ray_hits[rayID].normal[1] = payload.normal.y;
    record.ray_hits[rayID].normal[2] = payload.normal.z;
}

[shader("raygeneration")]
void point_in_volume(uniform dblRayGenData record, uniform DPTriangleGeomData mesh) {
    RayFirePayload payload;
    uint rayID = DispatchRaysIndex().x;

    // Trace the ray into the scene
    RayDesc rayDesc;
    rayDesc.Origin = float3(record.ray[rayID].origin);
    rayDesc.Direction = float3(normalize(record.ray[rayID].direction));
    rayDesc.TMin = float(record.ray[rayID].tMin); 
    rayDesc.TMax = float(record.ray[rayID].tMax); 

    SurfaceAccelerationStructure world = record.ray[rayID].volume_accel;

    // Pass the ray's origin and direction to the payload
    payload.surf_id = -1;
    payload.tlas = world;
    payload.piv = xdg::PointInVolume::OUTSIDE; // Initialize point in volume check result to outside (0)
    
    TraceRay(world, RAY_FLAG_NONE, 0xff, 0, 1, rayDesc, payload);

    record.hit[rayID].surf_id = payload.surf_id;
    record.hit[rayID].piv = payload.piv; // Point in volume check result
}

// ------------------------------------------------- Compute Shaders -------------------------------------------------
/* A shader to compute and store AABB min/maxes in single precision using double precision coords*/
[shader("compute")]
[numthreads(256, 1, 1)]
void
populate_aabbs(uint3 DispatchThreadID: SV_DispatchThreadID, uniform DPTriangleGeomData record) {
    int primID = DispatchThreadID.x;
    if (primID >= record.num_faces) return;

    int3 indices = record.index[primID];
    dp::vec3 A = record.vertex[indices[0]];
    dp::vec3 B = record.vertex[indices[1]];
    dp::vec3 C = record.vertex[indices[2]];
    dp::vec3 dpaabbmin = min(min(A, B), C);
    dp::vec3 dpaabbmax = max(max(A, B), C);
    dp::vec3 bump = dp::vec3(record.bounding_box_bump, record.bounding_box_bump, record.bounding_box_bump);
    float3 fpaabbmin = float3(dpaabbmin - bump);
    float3 fpaabbmax = float3(dpaabbmax + bump);

    record.aabbs[2 * primID] = fpaabbmin;
    record.aabbs[2 * primID + 1] = fpaabbmax;
}

// ------------------------------------------------ CUSTOM INTERSECTION SHADERS ------------------------------------------------


/* 1D ray generation intersection with a double precision triangle using the Plucker intersection algorithm*/
[shader("intersection")]
void DPTrianglePluckerIntersection(uniform DPTriangleGeomData record)
{
    int primID = PrimitiveIndex();
    int global_prim_id = record.primitive_refs[primID].id;

    uint rayID = DispatchRaysIndex().x;
    uint nRays = DispatchRaysDimensions().x;

    if (rayID >= nRays) return;

    // Load vertices
    int3 indices = record.index[primID];
    dp::vec3 v0 = record.vertex[indices[0]];
    dp::vec3 v1 = record.vertex[indices[1]];
    dp::vec3 v2 = record.vertex[indices[2]];
    dp::vec3 vertices[3] = { v0, v1, v2 };

    // Load either the legacy internal ray or the shared XDG batch record.
    dp::vec3 origin;
    dp::vec3 direction;
    double tMin;
    double tMax;
    xdg::HitOrientation hitOrientation;
    int last_hit_primitive = -1;

    if (PC.batch_mode != 0) {
        origin = dp::vec3(record.ray_hits[rayID].origin[0],
                          record.ray_hits[rayID].origin[1],
                          record.ray_hits[rayID].origin[2]);
        direction = dp::vec3(record.ray_hits[rayID].direction[0],
                             record.ray_hits[rayID].direction[1],
                             record.ray_hits[rayID].direction[2]);
        tMin = record.ray_hits[rayID].t_min;
        tMax = record.ray_hits[rayID].t_max;
        hitOrientation = PC.hitOrientation;
        last_hit_primitive = record.ray_hits[rayID].last_hit_primitive;
    } else {
        origin = record.ray[rayID].origin;
        direction = record.ray[rayID].direction;
        tMin = record.ray[rayID].tMin;
        tMax = record.ray[rayID].tMax;
        hitOrientation = record.ray[rayID].hitOrientation;
    }

    bool useOrientation = false;
    int orientation = 0;

    xdg::PluckerIntersectionResult result = xdg::plucker_ray_tri_intersect(vertices, 
                                                                           origin, 
                                                                           direction, 
                                                                           tMax, 
                                                                           tMin, 
                                                                           useOrientation, 
                                                                           orientation);
    if (result.hit == false) return; // No intersection
    double t = result.t;
    
    DPAttribute attr;
    attr.f64t = t;

    float f32t = float(t);
    if (double(f32t) < t) f32t = next_after(f32t);

    dp::vec3 norm = record.normals[primID]; // recover double precision normal. TODO - Should we calculate from vertices instead?

    // sense adjustment of normal
    const bool reverse_sense = PC.batch_mode != 0
        ? record.ray_hits[rayID].volume == record.reverse_vol
        : record.ray[rayID].volume_tree == record.reverse_tree;
    if (reverse_sense)
    {
        norm = -norm;
    }

    double norm_dot_dir = dp::dot(norm, direction);
    uint hit_kind = norm_dot_dir < 0 ? HIT_KIND_TRIANGLE_FRONT_FACE
                                     : HIT_KIND_TRIANGLE_BACK_FACE;

    if (orientation_cull(direction, norm, hitOrientation))
    {
        return;
    }

    if (PC.batch_mode != 0) {
        if (last_hit_primitive == global_prim_id) return;
    } else {
        for (int i = 0; i < record.ray[rayID].exclude_count; ++i)
        {
            if (record.ray[rayID].exclude_primitives[i] == global_prim_id) {
                return;
            }
        }
    }
    attr.global_prim_id = global_prim_id;  
    ReportHit(f32t, hit_kind, attr);
}


// ------------------------------------------------- Helper functions -------------------------------------------------

bool orientation_cull(in dp::vec3 ray, in dp::vec3 normal, in xdg::HitOrientation orientation) {
    if (orientation == xdg::HitOrientation::ANY) return false; // No culling
    if (orientation == xdg::HitOrientation::EXITING) return dp::dot(ray, normal) < 0.0; // Cull exiting rays
    if (orientation == xdg::HitOrientation::ENTERING) return dp::dot(ray, normal) > 0.0; // Cull entering rays
    return false; // Default case, no culling
}

// Plucker coordinate
double plucker_edge_test(in dp::vec3 vertexa, in dp::vec3 vertexb, in dp::vec3 ray, in dp::vec3 ray_normal)
{
    double pip;
    const double near_zero = 10 * DBL_EPSILON;

    if (first(vertexa, vertexb))
    {
        dp::vec3 edge = vertexb - vertexa;
        dp::vec3 edge_normal = dp::cross(edge, vertexa);
        pip = dp::dot(ray, edge_normal) + dp::dot(ray_normal, edge);
    }
    else
    {
        dp::vec3 edge = vertexa - vertexb;
        dp::vec3 edge_normal = dp::cross(edge, vertexb);
        pip = dp::dot(ray, edge_normal) + dp::dot(ray_normal, edge);
        pip = -pip;
    }

    if (near_zero > abs(pip)) pip = 0.0;
    return pip;
}

/* Function to return the vertex with the lowest coordinates. To force the same
    ray-edge computation, the Plücker test needs to use consistent edge
    representation. This would be more simple with MOAB handles instead of
    coordinates... */
inline bool first(in dp::vec3 a, in dp::vec3 b)
{
    if (a[0] < b[0]) return true;

    if (a[0] == b[0] && a[1] < b[1]) return true;

    if (a[1] == b[1] && a[2] < b[2]) return true;

    return false;
}

float next_after(float a) {
  uint a_ = asuint(a);
  if (a < 0) {
    a_--;
  } else {
    a_++;
  }
  return asfloat(a_);
}
