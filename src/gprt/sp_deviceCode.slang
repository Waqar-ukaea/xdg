#include "../../include/xdg/gprt/shared_structs.h"

[[vk::push_constant]]
RayFirePushConstants PC;

struct RayFirePayload {
    float distance;
    int surf_id;
    int primitive_id;
    xdg::PointInVolume piv;
    int next_volume;
    xdg::SurfaceBoundaryCondition boundary_condition;
    float3 normal;
};

bool orientation_cull(in float3 direction,
                      in float3 normal,
                      in xdg::HitOrientation orientation) {
    if (orientation == xdg::HitOrientation::ANY) return false;
    if (orientation == xdg::HitOrientation::EXITING) {
        return dot(direction, normal) < 0.0f;
    }
    if (orientation == xdg::HitOrientation::ENTERING) {
        return dot(direction, normal) > 0.0f;
    }
    return false;
}

[shader("anyhit")]
void ray_fire_any_hit(uniform TriangleGeomData record,
                      inout RayFirePayload payload) {
    const uint ray_id = DispatchRaysIndex().x;
    const uint primitive_index = PrimitiveIndex();
    const int global_primitive = record.primitive_refs[primitive_index].id;

    float3 direction;
    int input_volume;
    int last_hit_primitive = -1;
    xdg::HitOrientation orientation;

    if (PC.batch_mode != 0) {
        direction = float3(float(record.ray_hits[ray_id].direction[0]),
                           float(record.ray_hits[ray_id].direction[1]),
                           float(record.ray_hits[ray_id].direction[2]));
        input_volume = record.ray_hits[ray_id].volume;
        last_hit_primitive = record.ray_hits[ray_id].last_hit_primitive;
        orientation = PC.hitOrientation;
    } else {
        direction = float3(record.ray[ray_id].direction);
        input_volume = record.ray[ray_id].volume_tree == record.reverse_tree
            ? record.reverse_vol
            : record.forward_vol;
        orientation = record.ray[ray_id].hitOrientation;
    }

    float3 oriented_normal = record.normals[primitive_index];
    if (input_volume == record.reverse_vol) {
        oriented_normal = -oriented_normal;
    }

    if (orientation_cull(direction, oriented_normal, orientation) ||
        last_hit_primitive == global_primitive) {
        IgnoreHit();
        return;
    }

    if (PC.batch_mode == 0) {
        for (int i = 0; i < record.ray[ray_id].exclude_count; ++i) {
            if (record.ray[ray_id].exclude_primitives[i] == global_primitive) {
                IgnoreHit();
                return;
            }
        }
    }
}

[shader("closesthit")]
void ray_fire_hit(uniform TriangleGeomData record,
                  inout RayFirePayload payload) {
    const uint ray_id = DispatchRaysIndex().x;
    const uint primitive_index = PrimitiveIndex();

    float3 direction;
    int input_volume;
    if (PC.batch_mode != 0) {
        direction = float3(float(record.ray_hits[ray_id].direction[0]),
                           float(record.ray_hits[ray_id].direction[1]),
                           float(record.ray_hits[ray_id].direction[2]));
        input_volume = record.ray_hits[ray_id].volume;
    } else {
        direction = float3(record.ray[ray_id].direction);
        input_volume = record.ray[ray_id].volume_tree == record.reverse_tree
            ? record.reverse_vol
            : record.forward_vol;
    }

    const float3 stored_normal = record.normals[primitive_index];
    const float3 oriented_normal = input_volume == record.reverse_vol
        ? -stored_normal
        : stored_normal;

    payload.distance = RayTCurrent();
    payload.surf_id = record.surf_id;
    payload.primitive_id = record.primitive_refs[primitive_index].id;
    payload.piv = dot(direction, oriented_normal) > 0.0f
        ? xdg::PointInVolume::INSIDE
        : xdg::PointInVolume::OUTSIDE;
    payload.next_volume = input_volume == record.forward_vol
        ? record.reverse_vol
        : record.forward_vol;
    payload.boundary_condition = record.boundary_condition;
    payload.normal = stored_normal;
}

[shader("miss")]
void ray_fire_miss(inout RayFirePayload payload) {
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.piv = xdg::PointInVolume::OUTSIDE;
    payload.next_volume = -1;
    payload.boundary_condition = xdg::SurfaceBoundaryCondition::UNSET;
    payload.normal = float3(0.0f, 0.0f, 0.0f);
}

[shader("raygeneration")]
void ray_fire(uniform dblRayGenData record,
              uniform TriangleGeomData mesh) {
    const uint ray_id = DispatchRaysIndex().x;

    RayFirePayload payload;
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.piv = xdg::PointInVolume::OUTSIDE;
    payload.next_volume = -1;
    payload.boundary_condition = xdg::SurfaceBoundaryCondition::UNSET;
    payload.normal = float3(0.0f, 0.0f, 0.0f);

    RayDesc ray_desc;
    ray_desc.Origin = float3(record.ray[ray_id].origin);
    ray_desc.Direction = normalize(float3(record.ray[ray_id].direction));
    ray_desc.TMin = float(record.ray[ray_id].tMin);
    ray_desc.TMax = float(record.ray[ray_id].tMax);

    TraceRay(record.ray[ray_id].volume_accel,
             RAY_FLAG_NONE, 0xff, 0, 1, ray_desc, payload);

    record.hit[ray_id].distance = double(payload.distance);
    record.hit[ray_id].surf_id = payload.surf_id;
    record.hit[ray_id].primitive_id = payload.primitive_id;
    record.hit[ray_id].piv = payload.piv;
}

[shader("raygeneration")]
void ray_fire_batch(uniform XDGRayHitRayGenData record,
                    uniform TriangleGeomData mesh) {
    const uint ray_id = DispatchRaysIndex().x;
    const xdg::XDGRayHit ray_hit = record.ray_hits[ray_id];

    RayFirePayload payload;
    payload.distance = float(ray_hit.t_max);
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.piv = xdg::PointInVolume::OUTSIDE;
    payload.next_volume = -1;
    payload.boundary_condition = xdg::SurfaceBoundaryCondition::UNSET;
    payload.normal = float3(0.0f, 0.0f, 0.0f);

    const int volume = ray_hit.volume;
    if (volume >= 0 && volume < record.volume_accel_count) {
        RayDesc ray_desc;
        ray_desc.Origin = float3(float(ray_hit.origin[0]),
                                 float(ray_hit.origin[1]),
                                 float(ray_hit.origin[2]));
        ray_desc.Direction = normalize(float3(float(ray_hit.direction[0]),
                                              float(ray_hit.direction[1]),
                                              float(ray_hit.direction[2])));
        ray_desc.TMin = float(ray_hit.t_min);
        ray_desc.TMax = float(ray_hit.t_max);

        TraceRay(record.volume_accels[volume],
                 RAY_FLAG_NONE, 0xff, 0, 1, ray_desc, payload);
    }

    record.ray_hits[ray_id].distance = payload.surf_id == -1
        ? ray_hit.t_max
        : double(payload.distance);
    record.ray_hits[ray_id].surface = payload.surf_id;
    record.ray_hits[ray_id].primitive = payload.primitive_id;
    record.ray_hits[ray_id].point_in_volume = int(payload.piv);
    record.ray_hits[ray_id].next_volume = payload.next_volume;
    record.ray_hits[ray_id].boundary_condition = int(payload.boundary_condition);
    record.ray_hits[ray_id].normal[0] = double(payload.normal.x);
    record.ray_hits[ray_id].normal[1] = double(payload.normal.y);
    record.ray_hits[ray_id].normal[2] = double(payload.normal.z);
}

[shader("raygeneration")]
void point_in_volume(uniform dblRayGenData record,
                     uniform TriangleGeomData mesh) {
    const uint ray_id = DispatchRaysIndex().x;

    RayFirePayload payload;
    payload.distance = -1.0f;
    payload.surf_id = -1;
    payload.primitive_id = -1;
    payload.piv = xdg::PointInVolume::OUTSIDE;
    payload.next_volume = -1;
    payload.boundary_condition = xdg::SurfaceBoundaryCondition::UNSET;
    payload.normal = float3(0.0f, 0.0f, 0.0f);

    RayDesc ray_desc;
    ray_desc.Origin = float3(record.ray[ray_id].origin);
    ray_desc.Direction = normalize(float3(record.ray[ray_id].direction));
    ray_desc.TMin = float(record.ray[ray_id].tMin);
    ray_desc.TMax = float(record.ray[ray_id].tMax);

    TraceRay(record.ray[ray_id].volume_accel,
             RAY_FLAG_NONE, 0xff, 0, 1, ray_desc, payload);

    record.hit[ray_id].surf_id = payload.surf_id;
    record.hit[ray_id].piv = payload.piv;
}
