// for testing
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// xdg includes
#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/ray_tracers.h"
#include "mesh_mock.h"

using namespace xdg;

// Factory method to create ray tracer based on which library selected
static std::shared_ptr<RayTracer> make_raytracer(RTLibrary rt) {
  switch (rt) {
    case RTLibrary::EMBREE:
    #ifdef XDG_ENABLE_EMBREE
      return std::make_shared<EmbreeRayTracer>();
    #else
      SUCCEED("Embree backend not built; skipping.");
      return {};
    #endif

    case RTLibrary::GPRT:
    #ifdef XDG_ENABLE_GPRT
      return std::make_shared<GPRTRayTracer>();
    #else
      SUCCEED("GPRT backend not built; skipping.");
      return {};
    #endif
  }
  FAIL("Unknown RT backend enum value");
  return {};
}

// Actual code for test suite
static void run_ray_fire_suite(const std::shared_ptr<RayTracer>& rti) {
  REQUIRE(rti);

  auto mm = std::make_shared<MeshMock>(false);
  mm->init();
  REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

  auto [volume_tree, element_tree] = rti->register_volume(mm, mm->volumes()[0]);
  REQUIRE(volume_tree != ID_NONE);
  REQUIRE(element_tree == ID_NONE);

  Position origin {0.0, 0.0, 0.0};
  Direction direction {1.0, 0.0, 0.0};
  std::pair<double, MeshID> intersection;

  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));

  direction *= -1;
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(2.0, 1e-6));

  direction = {0.0, 1.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(6.0, 1e-6));
  direction *= -1;
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(3.0, 1e-6));

  direction = {0.0, 0.0, 1.0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(7.0, 1e-6));
  direction *= -1;
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(4.0, 1e-6));

  // fire from the outside of the cube toward each face, ensuring that the intersection distances are correct
  // rays should skip entering intersections and intersect with the far side of the cube
  origin = {-10.0, 0.0, 0.0};
  direction = {1.0, 0.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(15.0, 1e-6));

  origin = {10.0, 0.0, 0.0};
  direction = {-1.0, 0.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(12.0, 1e-6));

  // fire from the outside of the cube toward each face, ensuring that the intersection distances are correct
  // in this case rays are fired with a HitOrientation::ENTERING. Rays should hit the first surface intersected
  origin = {-10.0, 0.0, 0.0};
  direction = {1.0, 0.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::ENTERING);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(8.0, 1e-6));

  origin = {10.0, 0.0, 0.0};
  direction = {-1.0, 0.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::ENTERING);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));

  // if the distance is just enough, we should still get a hit
  // limit distance of the ray, shouldn't get a hit
  origin = {0.0, 0.0, 0.0};
  direction = {1.0, 0.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction, 4.5);
  REQUIRE(intersection.second == ID_NONE);

  intersection = rti->ray_fire(volume_tree, origin, direction, 5.1);
  REQUIRE(intersection.second != ID_NONE);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));

  // Test excluding primitives, fire a ray from the origin and log the hit face
  // By providing the hit face as an excluded primitive in a subsequent ray fire,
  // there should be no intersection returned
  std::vector<MeshID> exclude_primitives;
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::EXITING, &exclude_primitives);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));
  REQUIRE(exclude_primitives.size() == 1);

  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::EXITING, &exclude_primitives);
  REQUIRE(intersection.second == ID_NONE);
}

// ------- single test, multiple sections (one per built backend) --------------

TEST_CASE("Ray Fire on MeshMock (per-backend sections)", "[rayfire][mock]") {
  const RTLibrary candidates[] = { RTLibrary::EMBREE, RTLibrary::GPRT };

  for (RTLibrary rt : candidates) {
    auto rti = make_raytracer(rt);
    if (!rti) continue; // backend not built → gracefully skip

    DYNAMIC_SECTION(std::string("Backend = ") + RT_LIB_TO_STR.at(rt)) {
      run_ray_fire_suite(rti);
    }
  }
}
