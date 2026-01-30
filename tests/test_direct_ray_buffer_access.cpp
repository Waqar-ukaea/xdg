// for testing
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>

// xdg includes
#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/gprt/ray_tracer.h"
#include "xdg/xdg.h"
#include "mesh_mock.h"
#include "test_direct_ray_buffer_access_shared.h"
#include "util.h"
#include "gprt.h"

#include <chrono>

using namespace xdg;
using namespace xdg::test;

extern GPRTProgram test_direct_ray_buffer_access_deviceCode;

// This is a GPU only test - skip if no GPU ray tracing backends are enabled
TEMPLATE_TEST_CASE("Ray Fire with external populated rays", "[rayfire][mock]",
                   GPRT_Raytracer)
{
  constexpr auto rt_backend = TestType::value;

  DYNAMIC_SECTION(fmt::format("Backend = {}", rt_backend)) {
    check_ray_tracer_supported(rt_backend);

    auto rti = create_raytracer(rt_backend);
    REQUIRE(rti);

    auto mm = std::make_shared<MeshMock>(false);
    mm->init();
    REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

    auto xdg = std::make_shared<XDG>();
    xdg->set_mesh_manager_interface(mm);
    xdg->set_ray_tracing_interface(rti);
    xdg->prepare_raytracer();

    std::vector<Position> origins;
    std::vector<Direction> directions;
    size_t N = 64;
    make_rays(N, origins, directions);

    auto gprt_rt = std::dynamic_pointer_cast<GPRTRayTracer>(rti);
    REQUIRE(gprt_rt);

    const MeshID volume_id = mm->volumes()[0];
    GPRTContext context = gprt_rt->context();
    GPRTModule module = gprtModuleCreate(context, test_direct_ray_buffer_access_deviceCode);
    auto packRays = gprtComputeCreate<ExternalRayParams>(context, module, "pack_external_rays");

    std::vector<double> expected_distances(N, INFTY);
    std::vector<MeshID> expected_surfaces(N, ID_NONE);
    xdg->ray_fire(volume_id,
                  origins.data(),
                  directions.data(),
                  N,
                  expected_distances.data(),
                  expected_surfaces.data());

    // Create callback to populate rays on device
    RayPopulationCallback populate_callback = [&volume_id, &origins, &directions, &context, &packRays]
      (const DeviceRayHitBuffers& buffer, size_t numRays) {
      REQUIRE(origins.size() >= numRays);
      REQUIRE(directions.size() >= numRays);

      // Convert to double3 for use on GPU
      std::vector<double3> origins_device(numRays);
      std::vector<double3> directions_device(numRays);
      for (size_t i = 0; i < numRays; ++i) {
        origins_device[i] = {origins[i].x, origins[i].y, origins[i].z};
        directions_device[i] = {directions[i].x, directions[i].y, directions[i].z};
      }

      auto origins_buffer = gprtDeviceBufferCreate<double3>(context, numRays, origins_device.data());
      auto directions_buffer = gprtDeviceBufferCreate<double3>(context, numRays, directions_device.data());

      constexpr uint32_t threads_per_group = 256;
      const uint32_t groups = static_cast<uint32_t>((numRays + threads_per_group - 1) / threads_per_group);

      ExternalRayParams params = {};
      params.xdgRays = static_cast<dblRay*>(buffer.rayDevPtr);
      params.origins = gprtBufferGetDevicePointer(origins_buffer);
      params.directions = gprtBufferGetDevicePointer(directions_buffer);
      params.num_rays = static_cast<uint32_t>(numRays);
      params.total_threads = groups * threads_per_group;
      params.volume_mesh_id = volume_id;
      params.enabled = 1u;

      gprtComputeLaunch(packRays,
                        { groups, 1, 1 },
                        { threads_per_group, 1, 1 },
                        params);
      gprtComputeSynchronize(context);

      gprtBufferDestroy(origins_buffer);
      gprtBufferDestroy(directions_buffer);
    };
      
    // Populate rays via external API
    xdg->populate_rays_external(N, populate_callback);

    xdg->ray_fire_prepared(volume_id, N);
    std::vector<dblHit> hits;
    xdg->transfer_hits_buffer_to_host(N, hits);

    REQUIRE(hits.size() == N);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(hits[i].surf_id == expected_surfaces[i]);
      if (expected_surfaces[i] != ID_NONE) {
        REQUIRE_THAT(hits[i].distance, Catch::Matchers::WithinAbs(expected_distances[i], 1e-6));
      }
    }

    gprtComputeDestroy(packRays);
    gprtModuleDestroy(module);
  }
}
