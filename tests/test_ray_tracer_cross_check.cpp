// stl includes
#include <array>
#include <memory>
#include <vector>

// testing includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// xdg includes
#include "xdg/constants.h"
#include "xdg/ray_tracing_interface.h"
#include "xdg/xdg.h"

#include "util.h"

using namespace xdg;
using namespace xdg::test;

TEST_CASE("Test MOAB Pincell RT libraries Cross-Check ray_fire queries", "[moab][rayfire][cross-check]")
{

  check_ray_tracer_supported(RTLibrary::EMBREE);
  check_ray_tracer_supported(RTLibrary::GPRT);

  auto make_moab_xdg = [](RTLibrary rt_library) {
    auto xdg = XDG::create(MeshLibrary::MOAB, rt_library);
    xdg->mesh_manager()->load_file("pincell.h5m");
    xdg->mesh_manager()->init();
    xdg->mesh_manager()->parse_metadata();
    xdg->prepare_raytracer();
    return xdg;
  };

  const auto xdg_embree = make_moab_xdg(RTLibrary::EMBREE);
  const auto xdg_gprt = make_moab_xdg(RTLibrary::GPRT);

  std::vector<Direction> directions {
    {0.23, 0.45, 0.86},
    {-0.52, 0.37, 0.77},
    {0.61, -0.71, 0.35},
    {-0.31, -0.58, 0.75},
    {0.83, 0.18, -0.53},
    {-0.44, 0.82, -0.36}
  };

  const Position origin {0.0, 0.0, 0.0};

  for (const auto& volume : xdg_embree->mesh_manager()->volumes()) {
    for (const auto& direction : directions) {
      const auto embree_hit = xdg_embree->ray_fire(volume, origin, direction.normalized());
      const auto gprt_hit = xdg_gprt->ray_fire(volume, origin, direction.normalized());

      // Capture information of results for the two xdg instances to print when test fails
      CAPTURE(volume, direction, embree_hit.first, embree_hit.second, gprt_hit.first, gprt_hit.second);
      REQUIRE(gprt_hit.second == embree_hit.second);
      if (embree_hit.second != ID_NONE) {
        REQUIRE_THAT(gprt_hit.first, Catch::Matchers::WithinAbs(embree_hit.first, 1e-6));
      }
    }
  }
}
