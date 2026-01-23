#include "gprt.h"

#include "../include/xdg/gprt/ray.h"

struct GenerateRandomRayParams {
    xdg::dblRay* rays;
    uint numRays;
    double3 origin;
    uint seed;
    uint total_threads;
    double source_radius;
    int volume_mesh_id;
    uint enabled;
};

struct DebugReadTLASParams {
    SurfaceAccelerationStructure* table;
    uint index;
    SurfaceAccelerationStructure* out;
};
