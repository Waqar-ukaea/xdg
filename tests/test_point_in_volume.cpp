// for testing
#include <catch2/catch_test_macros.hpp>

// xdg includes
#include "xdg/mesh_manager_interface.h"
#include "xdg/ray_tracing_interface.h"
#include "xdg/embree/ray_tracer.h"
#include "xdg/gprt/ray_tracer.h"
#include "mesh_mock.h"

using namespace xdg;

TEST_CASE("Test Point in Volume Embree-MeshMock")
{
  std::shared_ptr<MeshManager> mm = std::make_shared<MeshMock>(false);
  mm->init(); // this should do nothing, just good practice to call it
  REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

  std::shared_ptr<RayTracer> rti = std::make_shared<EmbreeRayTracer>();
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
  point = {4.0 - 1e-06, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == true);

  // test a point just outside on the positive x boundary
  // no direction
  point = {5.001, 0.0, 0,0};
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == false);

  // test a point on the positive x boundary
  // and provide a direction
  point = {5.0, 0.0, 0.0};
  Direction dir = {1.0, 0.0, 0.0};
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

#include <cstdint>
#include <vector>

TEST_CASE("Test Point in Volume GPRT-MeshMock")
{
  std::shared_ptr<MeshManager> mm = std::make_shared<MeshMock>();
  mm->init();
  REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

  std::shared_ptr<RayTracer> rti = std::make_shared<GPRTRayTracer>();
  auto [volume_tree, element_tree] = rti->register_volume(mm, mm->volumes()[0]);
  REQUIRE(volume_tree != ID_NONE);
  REQUIRE(element_tree == ID_NONE);
  rti->init();

  // --- Single-ray checks ---
  Position point {0.0, 0.0, 0.0};
  bool result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == true);

  point = {0.0, 0.0, 1000.0};
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == false);

  point = {4.0 - 1e-06, 0.0, 0.0}; // inside near +x
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == true);

  point = {5.001, 0.0, 0.0}; // outside +x
  result = rti->point_in_volume(volume_tree, point);
  REQUIRE(result == false);

  // boundary + direction provided
  point = {5.0, 0.0, 0.0};
  Direction dir_posx = {1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point, &dir_posx);
  REQUIRE(result == true); // NOTE: your TODO mentions this currently fails

  // outside +x with +x direction
  point = {5.1, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point, &dir_posx);
  REQUIRE(result == false);

  // outside +x with -x direction
  Direction dir_negx = {-1.0, 0.0, 0.0};
  point = {5.1, 0.0, 0.0};
  result = rti->point_in_volume(volume_tree, point, &dir_negx);
  REQUIRE(result == false);

  // --- Final batch test over the same 7 cases ---
  std::vector<Position> points {
    {0.0, 0.0, 0.0},         // inside
    {0.0, 0.0, 1000.0},      // outside
    {4.0 - 1e-06, 0.0, 0.0}, // inside
    {5.001, 0.0, 0.0},       // outside
    {5.0, 0.0, 0.0},         // boundary (dir provided)
    {5.1, 0.0, 0.0},         // outside (+x dir)
    {5.1, 0.0, 0.0}          // outside (-x dir)
  };

  // Per-ray optional directions: nullptr means "use default"
  const Direction* d0 = nullptr;
  const Direction* d1 = nullptr;
  const Direction* d2 = nullptr;
  const Direction* d3 = nullptr;
  const Direction* d4 = &dir_posx; // boundary with +x dir
  const Direction* d5 = &dir_posx; // outside with +x dir
  const Direction* d6 = &dir_negx; // outside with -x dir

  std::vector<const Direction*> dir_ptrs { d0, d1, d2, d3, d4, d5, d6 };

  std::vector<uint8_t> results(points.size(), 0);

  rti->batch_point_in_volume(
    volume_tree,
    points.data(),
    dir_ptrs.data(), 
    points.size(),
    results.data(),
    /*exclude_primitives=*/nullptr  // not implemented yet 
  );

  // Expected to mirror the single-ray assertions above
  std::vector<uint8_t> expected { 1, 0, 1, 0, 1, 0, 0 };

  for (size_t i = 0; i < expected.size(); ++i) {
    INFO("batch PIV case i=" << i);
    REQUIRE(results[i] == expected[i]);
  }
}
