// for testing
#include <catch2/catch_test_macros.hpp>

// xdg includes
#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "mesh_mock.h"

// Backend headers only if built
#ifdef XDG_ENABLE_EMBREE
  #include "xdg/embree/ray_tracer.h"
#endif
#ifdef XDG_ENABLE_GPRT
  #include "xdg/gprt/ray_tracer.h"
#endif

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
static void run_point_in_volume_suite(const std::shared_ptr<RayTracer>& rti) {
  REQUIRE(rti);

  // Keep MeshMock usage consistent across backends
  auto mm = std::make_shared<MeshMock>(false);
  mm->init();
  REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

  auto [volume_tree, element_tree] = rti->register_volume(mm, mm->volumes()[0]);
  REQUIRE(volume_tree != ID_NONE);
  REQUIRE(element_tree == ID_NONE);

  Position point {0.0, 0.0, 0.0};
  bool result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == true);

  point = {0.0, 0.0, 1000.0};
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == false);

  // test a point just inside the positive x boundary
  point = {4.0 - 1e-6, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == true);

  // test a point just outside on the positive x boundary
  // no direction
  point = {5.001, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == false);

  // test a point on the positive x boundary
  // and provide a direction
  point = {5.0, 0.0, 0.0};
  Direction dir{1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point, &dir);
  REQUIRE(result == true);

  // test a point just outside the positive x boundary
  // and provide a direction
  point = {5.1, 0.0, 0.0};
  dir = {1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point, &dir);
  REQUIRE(result == false);

  // test a point just outside the positive x boundary,
  // flip the direction
  point = {5.1, 0.0, 0.0};
  dir = {-1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point, &dir);
  REQUIRE(result == false);
}

// ---------- single test, sections per backend --------------------------------

TEST_CASE("Point-in-volume on MeshMock (per-backend sections)", "[piv][mock]") {
  const RTLibrary candidates[] = { RTLibrary::EMBREE, RTLibrary::GPRT };

  for (RTLibrary rt : candidates) {
    auto rti = make_raytracer(rt);
    if (!rti) continue; // backend not built → skip
    DYNAMIC_SECTION(std::string("Backend = ") + RT_LIB_TO_STR.at(rt)) {
      run_point_in_volume_suite(rti);
    }
  }
}
