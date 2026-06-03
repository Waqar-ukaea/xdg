#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef XDG_OPENMP
#include <omp.h>
#endif

#include "argparse/argparse.hpp"

#include "xdg/constants.h"
#include "xdg/error.h"
#include "xdg/timer.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "ray_benchmark.h"

using namespace xdg;

int main(int argc, char** argv)
{
  argparse::ArgumentParser args("XDG Ray Tracing throughput benchmarking tool",
                                "1.0",
                                argparse::default_arguments::help);

  args.add_argument("filename")
    .help("Path to the input file");

  args.add_argument("volume")
    .help("Volume ID to query")
    .scan<'i', int>();

  args.add_argument("-n", "--num-rays")
    .default_value<std::uint32_t>(10'000'000)
    .help("Number of rays to cast for the benchmark")
    .scan<'u', std::uint32_t>();

  args.add_argument("-s", "--seed")
    .default_value<std::uint32_t>(12345)
    .help("Seed for random ray generation")
    .scan<'u', std::uint32_t>();

  args.add_argument("-o", "-p", "--origin", "--position")
    .default_value(std::vector<double>{0.0, 0.0, 0.0})
    .help("Ray origin/position")
    .scan<'g', double>()
    .nargs(3);

  args.add_argument("-m", "--mesh-library")
    .help("Mesh library to use. One of (MOAB, LIBMESH)")
    .default_value("MOAB");

  args.add_argument("-rt", "--rt-library")
    .help("Ray tracing library to use. Currently implemented: EMBREE")
    .default_value("EMBREE");

  args.add_argument("-l", "--list")
    .default_value(false)
    .implicit_value(true)
    .help("List all volumes in the file and exit");

  args.add_argument("-sr", "--source-radius")
    .default_value(0.0)
    .help("Radius of a scattered source around the origin")
    .scan<'g', double>();

  args.add_description(
    "Benchmarks ray-fire throughput for a selected mesh volume. A source "
    "position is provided and ray directions are randomly generated from it.");

  try {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(0);
  }

  std::string mesh_str = args.get<std::string>("--mesh-library");
  std::string rt_str = args.get<std::string>("--rt-library");
  std::string rt_label = rt_str;

  RTLibrary rt_lib;
  if (rt_str == "EMBREE") {
    rt_lib = RTLibrary::EMBREE;
  } else {
    fatal_error("Ray tracing library '{}' is not implemented in this benchmark tool yet", rt_str);
  }

  MeshLibrary mesh_lib;
  if (mesh_str == "MOAB") {
    mesh_lib = MeshLibrary::MOAB;
  } else if (mesh_str == "LIBMESH") {
    mesh_lib = MeshLibrary::LIBMESH;
  } else {
    fatal_error("Invalid mesh library '{}' specified", mesh_str);
  }

  const MeshID volume = args.get<int>("volume");
  const std::size_t num_rays = args.get<std::uint32_t>("--num-rays");
  const std::uint32_t seed = args.get<std::uint32_t>("--seed");
  const Position origin = args.get<std::vector<double>>("--origin");
  const double source_radius = args.get<double>("--source-radius");

  Timer wall_timer;
  Timer setup_timer;
  Timer generation_timer;
  Timer trace_timer;

  wall_timer.start();

  // XDG setup and ray tracer initialisation
  setup_timer.start();
  std::shared_ptr<XDG> xdg = XDG::create(mesh_lib, rt_lib);
  const auto& mesh_manager = xdg->mesh_manager();
  mesh_manager->load_file(args.get<std::string>("filename"));
  mesh_manager->init();

  if (args.get<bool>("--list")) {
    for (auto mesh_volume : mesh_manager->volumes()) {
      std::cout << mesh_volume << std::endl;
    }
    return 0;
  }

  xdg->prepare_volume_for_raytracing(volume);
  xdg->ray_tracing_interface()->init();
  setup_timer.stop();

  if (rt_lib == RTLibrary::EMBREE) {
    #ifdef XDG_OPENMP
    rt_label += " (" + std::to_string(omp_get_max_threads()) + " CPU threads)";
    #endif
  }

  std::cout << "Volume ID: " << volume
            << " with: " << mesh_manager->num_volume_faces(volume)
            << " faces" << std::endl;
  std::cout << "Starting ray fire benchmark with " << num_rays
            << " rays using " << rt_label << "\n" << std::endl;
  std::cout << "XDG initialisation time       = "
            << setup_timer.elapsed() << "s" << std::endl;

  if (rt_lib == RTLibrary::EMBREE) {
    // Generate random rays from source
    generation_timer.start();
    std::vector<Position> origins(num_rays);
    std::vector<Direction> directions(num_rays);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < num_rays; ++i) {
      std::uint32_t state = seed ^ static_cast<std::uint32_t>(i);
      auto sample = tools::benchmark::random_spherical_source(origin.x,
                                                              origin.y,
                                                              origin.z,
                                                              state,
                                                              source_radius);
      origins[i] = Position(sample.position[0],
                            sample.position[1],
                            sample.position[2]);
      directions[i] = Direction(sample.direction[0],
                                sample.direction[1],
                                sample.direction[2]);
    }
    generation_timer.stop();

    // Trace rays
    trace_timer.start();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < num_rays; ++i) {
      (void) xdg->ray_fire(volume, origins[i], directions[i]);
    }

    trace_timer.stop();
  }

  const double generation_time = generation_timer.elapsed();
  const double trace_time = trace_timer.elapsed();
  const double end_to_end_time = generation_time + trace_time;
  const double trace_only_rps = trace_time > 0.0
    ? static_cast<double>(num_rays) / trace_time
    : 0.0;
  const double end_to_end_rps = end_to_end_time > 0.0
    ? static_cast<double>(num_rays) / end_to_end_time
    : 0.0;

  wall_timer.stop();

  std::cout << "Random ray generation time   = "
            << generation_time << "s" << std::endl;
  std::cout << "Generation + tracing time    = "
            << end_to_end_time << "s" << std::endl;
  std::cout << "End-to-end throughput        = "
            << end_to_end_rps << " rays/s" << std::endl;
  std::cout << "Full wall-clock time         = "
            << wall_timer.elapsed() << "s (post-argparse)" << std::endl;

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Ray tracing time (trace-only)= "
            << trace_time << "s for " << num_rays << " rays" << std::endl;
  std::cout << "Trace-only throughput        = "
            << trace_only_rps << " rays/s" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  return 0;
}
