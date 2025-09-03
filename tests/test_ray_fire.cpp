// for testing
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// xdg includes
#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/embree/ray_tracer.h"
#include "xdg/gprt/ray_tracer.h"

#include "mesh_mock.h"

using namespace xdg;

TEST_CASE("Test Ray Fire Embree-MeshMock")
{
  std::shared_ptr<MeshManager> mm = std::make_shared<MeshMock>(false);
  mm->init(); // this should do nothing, just good practice to call it
  REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

  std::shared_ptr<RayTracer> rti = std::make_shared<EmbreeRayTracer>();
  auto [volume_tree, element_tree] = rti->register_volume(mm, mm->volumes()[0]);
  REQUIRE(volume_tree != ID_NONE);
  REQUIRE(element_tree == ID_NONE);

  Position origin {0.0, 0.0, 0.0};
  Direction direction {1.0, 0.0, 0.0};
  std::pair<double, MeshID> intersection;

  // fire from the origin toward each face, ensuring that the intersection distances are correct
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

  // limit distance of the ray, shouldn't get a hit
  origin = {0.0, 0.0, 0.0};
  direction = {1.0, 0.0, 0.0};
  intersection = rti->ray_fire(volume_tree, origin, direction, 4.5);
  REQUIRE(intersection.second == ID_NONE);

  // if the distance is just enough, we should still get a hit
  // limit distance of the ray, shouldn't get a hit
  origin = {0.0, 0.0, 0.0};
  direction = {1.0, 0.0, 0.0};
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

TEST_CASE("Test Ray Fire GPRT-MeshMock")
{
  std::shared_ptr<MeshManager> mm = std::make_shared<MeshMock>();
  mm->init();
  REQUIRE(mm->mesh_library() == MeshLibrary::MOCK);

  std::shared_ptr<RayTracer> rti = std::make_shared<GPRTRayTracer>();
  auto [volume_tree, element_tree] = rti->register_volume(mm, mm->volumes()[0]);
  REQUIRE(volume_tree != ID_NONE);
  REQUIRE(element_tree == ID_NONE);
  rti->init();

  struct RayCase {
    Position origin;
    Direction direction;
    double expectedDistance;
  };

  // Buckets for batch (INFTY limit only)
  std::vector<RayCase> exitingCases;
  std::vector<RayCase> enteringCases;

  Position origin;
  Direction direction;
  std::pair<double, MeshID> intersection;

  // ---- Inside cube (EXITING)
  origin = {0,0,0}; direction = {1,0,0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));
  exitingCases.push_back({origin, direction, 5.0});

  direction *= -1;
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(2.0, 1e-6));
  exitingCases.push_back({origin, direction, 2.0});

  direction = {0,1,0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(6.0, 1e-6));
  exitingCases.push_back({origin, direction, 6.0});

  direction *= -1;
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(3.0, 1e-6));
  exitingCases.push_back({origin, direction, 3.0});

  direction = {0,0,1};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(7.0, 1e-6));
  exitingCases.push_back({origin, direction, 7.0});

  direction *= -1;
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(4.0, 1e-6));
  exitingCases.push_back({origin, direction, 4.0});

  // ---- Outside cube (EXITING)
  origin = {-10,0,0}; direction = {1,0,0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(15.0, 1e-6));
  exitingCases.push_back({origin, direction, 15.0});

  origin = {10,0,0}; direction = {-1,0,0};
  intersection = rti->ray_fire(volume_tree, origin, direction);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(12.0, 1e-6));
  exitingCases.push_back({origin, direction, 12.0});

  // ---- Outside cube (ENTERING)
  origin = {-10,0,0}; direction = {1,0,0};
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::ENTERING);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(8.0, 1e-6));
  enteringCases.push_back({origin, direction, 8.0});

  origin = {10,0,0}; direction = {-1,0,0};
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::ENTERING);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));
  enteringCases.push_back({origin, direction, 5.0});

  // ---- Distance-limited (scalar-only)
  origin = {0,0,0}; direction = {1,0,0};
  intersection = rti->ray_fire(volume_tree, origin, direction, 4.5);
  REQUIRE(intersection.second == ID_NONE);

  intersection = rti->ray_fire(volume_tree, origin, direction, 5.1);
  REQUIRE(intersection.second != ID_NONE);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));

  // ---- Exclude primitives (scalar-only)
  std::vector<MeshID> exclude_primitives;
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::EXITING, &exclude_primitives);
  REQUIRE_THAT(intersection.first, Catch::Matchers::WithinAbs(5.0, 1e-6));
  REQUIRE(exclude_primitives.size() == 1);
  intersection = rti->ray_fire(volume_tree, origin, direction, INFTY, HitOrientation::EXITING, &exclude_primitives);
  REQUIRE(intersection.second == ID_NONE);

  // -------------------------------
  // Batch helpers (simple, one per orientation)
  // -------------------------------
  auto run_batch_simple = [&](const std::vector<RayCase>& cases,
                              HitOrientation orientation)
  {
    if (cases.empty()) return;

    std::vector<Position> origins;
    std::vector<Direction> directions;
    std::vector<double> expected;

    origins.reserve(cases.size());
    directions.reserve(cases.size());
    expected.reserve(cases.size());

    for (const auto& rc : cases) {
      origins.push_back(rc.origin);
      directions.push_back(rc.direction);
      expected.push_back(rc.expectedDistance);
    }

    std::vector<double> hitDistances(cases.size(), INFTY);
    std::vector<MeshID> surfaceIDs(cases.size(), ID_NONE);

    rti->batch_ray_fire(volume_tree,
                        origins.data(),
                        directions.data(),
                        cases.size(),
                        hitDistances.data(),
                        surfaceIDs.data(),
                        /*dist_limit*/ INFTY,
                        orientation,
                        /*exclude_primitives*/ nullptr);

    for (size_t i = 0; i < cases.size(); ++i) {
      REQUIRE_THAT(hitDistances[i], Catch::Matchers::WithinAbs(expected[i], 1e-6));
    }
  };

  // -------------------------------
  // Two quick batch calls
  // -------------------------------
  run_batch_simple(exitingCases,  HitOrientation::EXITING);
  run_batch_simple(enteringCases, HitOrientation::ENTERING);
}
