#include "../../include/xdg/gprt/shared_structs.h"

[[vk::push_constant]]
RayFirePushConstants PC;

struct RayFirePayload {
    float distance;    // Distance to intersection
    int surf_id;     // ID of the surface hit
    SurfaceAccelerationStructure tlas;
    int primitive_id; // ID of the primitive hit
    xdg::PointInVolume piv; // Point in volume check (0 for outside, 1 for inside) 
};

[shader("closesthit")]
void ray_fire_closesthit(uniform TriangleGeomData record, inout RayFirePayload payload) {
    // Distance from the ray origin to the hit point
    uint hit_kind = HitKind();
    uint rayID = DispatchRaysIndex().x;

    payload.piv = (hit_kind == HIT_KIND_TRIANGLE_FRONT_FACE) 
    ? xdg::PointInVolume::OUTSIDE
    : xdg::PointInVolume::INSIDE; 

    int instanceID = InstanceID();

    payload.distance = RayTCurrent();
    payload.surf_id = record.surf_id;
}

[shader("anyhit")]
void ray_fire_anyhit(uniform TriangleGeomData record, inout RayFirePayload payload) {
    // For point-in-volume checks, we can terminate the ray on the first hit
    uint rayID = DispatchRaysIndex().x;
    uint primID = PrimitiveIndex();

    float3 direction = record.ray[rayID].direction;

    float3 norm = record.normals[primID];
    // sense adjustment of normal
    if (PC.volume_tree == record.reverse_tree)
    {
        norm = -norm;
    }

    xdg::HitOrientation hitOrientation = PC.hitOrientation;

    if (orientation_cull(direction, norm, hitOrientation))
    {
        IgnoreHit();
        return;
    }

    int global_prim_id = record.primitive_refs[primID].id;

    for (int i = 0; i < record.ray[rayID].exclude_count; ++i)
    {
        if (record.ray[rayID].exclude_primitives[i] == global_prim_id) {
            IgnoreHit();
            return;
        }
    }
}

[shader("miss")]
void ray_fire_miss(inout RayFirePayload payload) {
    // Set the miss payload to default values
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
}

// This ray generation program will kick off the ray tracing process,
// generating rays and tracing them into the world.
[shader("raygeneration")]
void ray_fire(uniform RayGenData record, uniform TriangleGeomData mesh) {
    RayFirePayload payload;
    uint rayID = DispatchRaysIndex().x;

    // Trace the ray into the scene
    RayDesc rayDesc;
    rayDesc.Origin = float3(record.ray[rayID].origin);
    rayDesc.Direction = normalize(float3(record.ray[rayID].direction));
    rayDesc.TMin = float(PC.tMin); 
    rayDesc.TMax = float(PC.tMax); 

    SurfaceAccelerationStructure world = PC.volume_accel;

    // Pass the ray's origin and direction to the payload
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.tlas = world;

    TraceRay(world, RAY_FLAG_NONE, 0xff, 0, 1, rayDesc, payload);

    // Store the distance to the hit point and the surface ID in buffers for CPU
    record.hit[rayID].distance = payload.distance;
    record.hit[rayID].surf_id = payload.surf_id;
    record.hit[rayID].primitive_id = payload.primitive_id;
}

[shader("raygeneration")]
void point_in_volume(uniform RayGenData record, uniform TriangleGeomData mesh) {
    RayFirePayload payload;
    uint rayID = DispatchRaysIndex().x;

    // Trace the ray into the scene
    RayDesc rayDesc;
    rayDesc.Origin = float3(record.ray[rayID].origin);
    rayDesc.Direction = float3(normalize(record.ray[rayID].direction));
    rayDesc.TMin = float(PC.tMin); 
    rayDesc.TMax = float(PC.tMax); 

    SurfaceAccelerationStructure world = PC.volume_accel;

    // Pass the ray's origin and direction to the payload
    payload.surf_id = -1;
    payload.tlas = world;
    payload.piv = xdg::PointInVolume::OUTSIDE; // Initialize point in volume check result to outside (0)
    
    TraceRay(world, RAY_FLAG_NONE, 0xff, 0, 1, rayDesc, payload);

    record.hit[rayID].surf_id = payload.surf_id;
    record.hit[rayID].piv = payload.piv; // Point in volume check result
}

bool orientation_cull(in float3 ray, in float3 normal, in xdg::HitOrientation orientation) {
    if (orientation == xdg::HitOrientation::ANY) return false; // No culling
    if (orientation == xdg::HitOrientation::EXITING) return dot(ray, normal) < 0.0; // Cull exiting rays
    if (orientation == xdg::HitOrientation::ENTERING) return dot(ray, normal) > 0.0; // Cull entering rays
    return false; // Default case, no culling
}