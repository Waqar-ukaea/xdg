// for testing
#include <catch2/catch_test_macros.hpp>

// xdg includes
#include "xdg/mesh_manager_interface.h"
#include "xdg/ray_tracing_interface.h"

#include "mesh_mock.h"

using namespace xdg;

TEST_CASE("Test Point in Volume")
{
  std::shared_ptr<MeshManager> mm = std::make_shared<MeshMock>();
  mm->init(); // this should do nothing, just good practice to call it
  REQUIRE(mm->mesh_library() == MeshLibrary::INTERNAL);
  std::shared_ptr<RayTracer> rti = std::make_shared<RayTracer>();

  std::unordered_map<MeshID, TreeID> volume_to_scene_map;
  for (auto volume: mm->volumes()) {
    volume_to_scene_map[volume]= rti->register_volume(mm, volume);
  }
  MeshID volume = mm->volumes()[0];

  Position point {0.0, 0.0, 0.0};
  bool result = rti->point_in_volume(volume, point);
  REQUIRE(result == true);

  point = {0.0, 0.0, 1000.0};
  result = rti->point_in_volume(volume, point);
  REQUIRE(result == false);

  // test a point just inside the positive x boundary
  point = {5.0 - 1e-06, 0.0, 0.0};
  result = rti->point_in_volume(volume, point);
  REQUIRE(result == true);

  // test a point on the positive x boundary
  point = {5.0, 0.0, 0,0};
  result = rti->point_in_volume(volume, point);
  REQUIRE(result == false);

  // test a point just outside the positive x boundary
  // and provide a direction
  point = {5.0, 0.0, 0.0};
  Direction dir = {1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume, point, &dir);
  REQUIRE(result == true);

  // test a point just outside the positive x boundary
  // and provide a direction
  point = {5.1, 0.0, 0.0};
  dir = {1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume, point, &dir);
  REQUIRE(result == false);

  // test a point just outside the positive x boundary,
  // flip the direction
  point = {5.1, 0.0, 0.0};
  dir = {-1.0, 0.0, 0.0};
  result = rti->point_in_volume(volume, point, &dir);
  REQUIRE(result == false);
}