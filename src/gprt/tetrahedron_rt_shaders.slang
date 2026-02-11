#ifndef XDG_GPRT_TETRAHEDRON_RT_SHADERS_SLANG
#define XDG_GPRT_TETRAHEDRON_RT_SHADERS_SLANG

#include "../../include/xdg/gprt/rt_common.slangh"

struct SolidRayFirePayload {
    double distance; // Distance to intersection
    int surf_id;     // ID of the surface hit
    SolidAccelerationStructure tlas;
    int primitive_id;       // ID of the primitive hit
};


// ------------------------------------------------- Compute Shaders -------------------------------------------------
/* A shader to compute and store AABB min/maxes in single precision using double precision coords for Tetrahedra*/
[shader("compute")]
[numthreads(1, 1, 1)]
void
populate_tet_aabbs(uint3 DispatchThreadID: SV_DispatchThreadID, uniform DPTetrahedronGeomData record) {
    int primID = DispatchThreadID.x;
    int4 indices = record.index[primID];
    double3 A = record.vertex[indices[0]];
    double3 B = record.vertex[indices[1]];
    double3 C = record.vertex[indices[2]];
    double3 D = record.vertex[indices[3]];
    double3 dpaabbmin = min(min(A, B), min(C, D));
    double3 dpaabbmax = max(max(A, B), max(C, D));
    float3 fpaabbmin = float3(dpaabbmin - float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON));
    float3 fpaabbmax = float3(dpaabbmax + float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON));
    
    record.aabbs[2 * primID] = fpaabbmin;
    record.aabbs[2 * primID + 1] = fpaabbmax;
}

#endif // XDG_GPRT_TETRAHEDRON_RT_SHADERS_SLANG
