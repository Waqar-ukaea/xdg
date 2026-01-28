#ifndef _XDG_RAY_BENCHMARK_H
#define _XDG_RAY_BENCHMARK_H

#include <algorithm>
#include <cmath>
#include <utility>

#include "gprt/gprt.h"
#include "xdg/gprt/ray.h"
#include "xdg/gprt/ray_tracer.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "ray_benchmark_shared.h"

extern GPRTProgram ray_benchmark_deviceCode;

namespace xdg::tools::benchmark {

inline double rand01(uint32_t &state)
{
  state = state * 1664525u + 1013904223u;
  return double(state) * (1.0 / 4294967296.0);
}

inline Direction random_unit_dir_lcg(uint32_t &state)
{
  double x1, x2, s;
  do {
    x1 = rand01(state) * 2.0 - 1.0;
    x2 = rand01(state) * 2.0 - 1.0;
    s  = x1 * x1 + x2 * x2;
  } while (s <= 0.0 || s >= 1.0);

  double t = 2.0 * std::sqrt(1.0 - s);
  return { x1 * t, x2 * t, 1.0 - 2.0 * s };
}

// Generates a random point cloud with radius (--source-radius)
inline std::pair<Position, Direction> random_spherical_source(const Position& origin,
                                                              std::uint32_t state,
                                                              double source_radius)
{
  // Always generate random direction
  Direction dir = random_unit_dir_lcg(state);
  Position pos = origin;
  if (source_radius > 0.0) {
    // random origins (spherical source)
    double r = source_radius * std::cbrt(rand01(state)); // uniform in ball
    pos += dir * r;
  }
  return {pos, dir};
}

//  - User creates their own GPU compute API method to populate rays and passes that to XDG
//  - In this miniapp we are using GPRT as a demonstration
//  - This callback runs inside populate_rays_external and receives XDG's device buffers
inline RayPopulationCallback make_generate_rays_callback(GPRTContext gprt_context,
                                                         Position origin,
                                                         double source_radius,
                                                         uint32_t seed,
                                                         MeshID volume)
{
  return [gprt_context, origin, source_radius, seed, volume](const DeviceRayHitBuffers& buffer, size_t numRays) {
    GPRTContext context = gprt_context;
    GPRTModule module = gprtModuleCreate(context, ray_benchmark_deviceCode);
    auto genRandomRays = gprtComputeCreate<GenerateRandomRayParams>(
                         context, module, "generate_random_rays");

    constexpr int threadsPerGroup = 64;
    const int neededGroups = static_cast<int>((numRays + threadsPerGroup - 1) / threadsPerGroup);
    const int groups = std::min(neededGroups, WORKGROUP_LIMIT);

    GenerateRandomRayParams randomRayParams = {};
    randomRayParams.rays = static_cast<dblRay*>(buffer.rayDevPtr); // Cast opaque pointer to typed dblRay*
    randomRayParams.numRays = static_cast<uint32_t>(numRays);
    randomRayParams.source_radius = source_radius;
    randomRayParams.origin = { origin.x, origin.y, origin.z };
    randomRayParams.seed = seed;
    randomRayParams.total_threads = static_cast<uint32_t>(groups * threadsPerGroup);
    randomRayParams.volume_mesh_id = volume;
    randomRayParams.enabled = 1u;

    gprtComputeLaunch(genRandomRays,
                      { static_cast<uint32_t>(groups), 1, 1 },
                      { static_cast<uint32_t>(threadsPerGroup), 1, 1 },
                      randomRayParams);
    gprtComputeSynchronize(context);

    gprtComputeDestroy(genRandomRays);
    gprtModuleDestroy(module);
  };
}

} // namespace xdg::tools::benchmark

#endif // _XDG_RAY_BENCHMARK_H
