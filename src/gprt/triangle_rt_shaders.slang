#ifndef XDG_GPRT_TRIANGLE_RT_SHADERS_SLANG
#define XDG_GPRT_TRIANGLE_RT_SHADERS_SLANG

#include "../../include/xdg/gprt/rt_common.slangh"

struct SurfaceRayFirePayload {
    double distance; // Distance to intersection
    int surf_id;     // ID of the surface hit
    SurfaceAccelerationStructure tlas;
    int primitive_id;       // ID of the primitive hit
    xdg::PointInVolume piv; // Point in volume check (0 for outside, 1 for inside)
};

[shader("closesthit")]
void ray_fire_hit(uniform DPTriangleGeomData record, inout SurfaceRayFirePayload payload, in DPAttribute attr) {
    // Distance from the ray origin to the hit point
    uint hit_kind = HitKind();
    uint rayID = DispatchRaysIndex().x;

    // There is some logic for handling next volumes inside the h5m-reader which I could make use of too
    // TODO : Should the dblHit struct return the next volume ID for the ray back to the host

    payload.piv = (hit_kind == HIT_KIND_TRIANGLE_FRONT_FACE) 
    ? xdg::PointInVolume::OUTSIDE
    : xdg::PointInVolume::INSIDE; 

    int instanceID = InstanceID();
    
    payload.distance = attr.f64t;
    payload.surf_id = record.surf_id;
    payload.primitive_id = attr.global_prim_id;
}

[shader("miss")]
void ray_fire_miss(inout SurfaceRayFirePayload payload) {
    // Set the miss payload to default values
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.piv = xdg::PointInVolume::UNSET;
}

// This ray generation program will kick off the ray tracing process,
// generating rays and tracing them into the world.
[shader("raygeneration")]
void ray_fire(uniform dblRayGenData record) {
    SurfaceRayFirePayload payload;
    uint rayID = DispatchRaysIndex().x;

    // Trace the ray into the scene
    RayDesc rayDesc;
    rayDesc.Origin = float3(record.ray[rayID].origin);
    rayDesc.Direction = normalize(float3(record.ray[rayID].direction));
    rayDesc.TMin = float(record.ray[rayID].tMin); 
    rayDesc.TMax = float(record.ray[rayID].tMax); 

    SurfaceAccelerationStructure world = record.ray[rayID].volume_accel_surf;

    // Pass the ray's origin and direction to the payload
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.tlas = world;

    TraceRay(world, RAY_FLAG_NONE, 0xff, xdg::RT_SURFACE_RAY_INDEX, xdg::RT_SURFACE_MISS_INDEX, rayDesc, payload);

    // Store the distance to the hit point and the surface ID in buffers for CPU
    record.hit[rayID].distance = payload.distance;
    record.hit[rayID].surf_id = payload.surf_id;
    record.hit[rayID].primitive_id = payload.primitive_id;
}

[shader("raygeneration")]
void point_in_volume(uniform dblRayGenData record) {
    SurfaceRayFirePayload payload;
    uint rayID = DispatchRaysIndex().x;

    // Trace the ray into the scene
    RayDesc rayDesc;
    rayDesc.Origin = float3(record.ray[rayID].origin);
    rayDesc.Direction = float3(normalize(record.ray[rayID].direction));
    rayDesc.TMin = float(record.ray[rayID].tMin); 
    rayDesc.TMax = float(record.ray[rayID].tMax); 

    SurfaceAccelerationStructure world = record.ray[rayID].volume_accel_surf;

    // Pass the ray's origin and direction to the payload
    payload.surf_id = -1;
    payload.tlas = world;
    payload.piv = xdg::PointInVolume::OUTSIDE; // Initialize point in volume check result to outside (0)
    
    TraceRay(world, RAY_FLAG_NONE, 0xff, xdg::RT_SURFACE_RAY_INDEX, xdg::RT_SURFACE_MISS_INDEX, rayDesc, payload);

    record.hit[rayID].surf_id = payload.surf_id;
    record.hit[rayID].piv = payload.piv; // Point in volume check result
}

// ------------------------------------------------- Compute Shaders -------------------------------------------------
/* A shader to compute and store AABB min/maxes in single precision using double precision coords for Triangles*/
[shader("compute")]
[numthreads(1, 1, 1)]
void
populate_tri_aabbs(uint3 DispatchThreadID: SV_DispatchThreadID, uniform DPTriangleGeomData record) {
    int primID = DispatchThreadID.x;
    int3 indices = record.index[primID];
    double3 A = record.vertex[indices[0]];
    double3 B = record.vertex[indices[1]];
    double3 C = record.vertex[indices[2]];
    double3 dpaabbmin = min(min(A, B), C);
    double3 dpaabbmax = max(max(A, B), C);
    float3 fpaabbmin = float3(dpaabbmin - float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON));
    float3 fpaabbmax = float3(dpaabbmax + float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON));

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

    // Load rays
    dp::vec3 origin = record.ray[rayID].origin;
    dp::vec3 direction = record.ray[rayID].direction;
    double tMin = record.ray[rayID].tMin;
    double tMax = record.ray[rayID].tMax;

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
    if (record.ray[rayID].volume_tree == record.reverse_tree)
    {
        norm = -norm;
    }

    double norm_dot_dir = dp::dot(norm, direction);
    uint hit_kind = norm_dot_dir < 0 ? HIT_KIND_TRIANGLE_FRONT_FACE
                                     : HIT_KIND_TRIANGLE_BACK_FACE;

    xdg::HitOrientation hitOrientation = record.ray[rayID].hitOrientation;

    if (orientation_cull(direction, norm, hitOrientation))
    {
        return;
    }

    for (int i = 0; i < record.ray[rayID].exclude_count; ++i)
    {
        if (record.ray[rayID].exclude_primitives[i] == global_prim_id) {
            return;
        }
    }
    attr.global_prim_id = global_prim_id;
    ReportHit(f32t, hit_kind, attr);
}

#endif // XDG_GPRT_TRIANGLE_RT_SHADERS_SLANG
