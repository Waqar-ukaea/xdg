#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"

#include "xdg/DPRT/ray_tracer.h"
#include "xdg/error.h"
#include "xdg/mesh_managers.h"
#include "xdg/ray_tracers.h"
#include "xdg/timer.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace xdg;

namespace {

struct BenchmarkResult {
  std::string label;
  double trace_time {0.0};
  size_t num_hits {0};
};

inline double rand01(uint32_t& state)
{
  state = state * 1664525u + 1013904223u;
  return double(state) * (1.0 / 4294967296.0);
}

Direction random_unit_dir_lcg(uint32_t& state)
{
  double x1, x2, s;
  do {
    x1 = rand01(state) * 2.0 - 1.0;
    x2 = rand01(state) * 2.0 - 1.0;
    s = x1 * x1 + x2 * x2;
  } while (s <= 0.0 || s >= 1.0);

  const double t = 2.0 * std::sqrt(1.0 - s);
  return {x1 * t, x2 * t, 1.0 - 2.0 * s};
}

std::pair<Position, Direction> random_spherical_source(const Position& origin,
                                                       uint32_t state,
                                                       double source_radius)
{
  Direction dir = random_unit_dir_lcg(state);
  Position pos = origin;
  if (source_radius > 0.0) {
    const double r = source_radius * std::cbrt(rand01(state));
    pos += dir * r;
  }
  return {pos, dir};
}

void populate_seeded_rays(std::vector<DPRTRay>& rays,
                          const Position& origin,
                          double source_radius,
                          uint32_t seed,
                          double t_max)
{
#pragma omp parallel for schedule(static) if(rays.size() > 4096)
  for (size_t i = 0; i < rays.size(); ++i) {
    auto [pos, dir] = random_spherical_source(origin, seed ^ static_cast<uint32_t>(i), source_radius);
    DPRTRay ray {};
    ray.origin = {pos.x, pos.y, pos.z};
    ray.direction = {dir.x, dir.y, dir.z};
    ray.tMin = 0.0;
    ray.tMax = t_max;
    rays[i] = ray;
  }
}

void initialize_hits(std::vector<DPRTHit>& hits)
{
#pragma omp parallel for schedule(static) if(hits.size() > 4096)
  for (size_t i = 0; i < hits.size(); ++i) {
    hits[i].primID = -1;
    hits[i].instID = -1;
    hits[i].geomUserData = 0;
    hits[i].t = INFTY;
    hits[i].u = 0.0;
    hits[i].v = 0.0;
  }
}

BenchmarkResult run_cpu_trace(const std::shared_ptr<XDG>& xdg,
                              MeshID volume,
                              const std::vector<DPRTRay>& rays)
{
  Timer timer;
  timer.start();

  size_t num_hits = 0;
#pragma omp parallel for reduction(+:num_hits) schedule(static)
  for (size_t i = 0; i < rays.size(); ++i) {
    const auto& ray = rays[i];
    Position origin(ray.origin.x, ray.origin.y, ray.origin.z);
    Direction direction(ray.direction.x, ray.direction.y, ray.direction.z);
    auto [distance, _surface] = xdg->ray_fire(volume, origin, direction, ray.tMax);
    if (distance < INFTY) ++num_hits;
  }

  timer.stop();
  return BenchmarkResult{"CPU host-threaded", timer.elapsed(), num_hits};
}

BenchmarkResult run_target_trace(const std::shared_ptr<DPRTRayTracer>& rti,
                                 TreeID tree,
                                 const std::vector<DPRTRay>& rays)
{
#ifndef _OPENMP
  (void)rti;
  (void)tree;
  (void)rays;
  fatal_error("OpenMP target offload path requested, but this build does not have OpenMP enabled.");
#else
  const int host_device = omp_get_initial_device();
  const int gpu_device = 0;
  const size_t ray_bytes = rays.size() * sizeof(DPRTRay);
  const size_t hit_bytes = rays.size() * sizeof(DPRTHit);

  std::vector<DPRTHit> hits(rays.size());
  initialize_hits(hits);

  auto* d_rays = static_cast<DPRTRay*>(omp_target_alloc(ray_bytes, gpu_device));
  auto* d_hits = static_cast<DPRTHit*>(omp_target_alloc(hit_bytes, gpu_device));
  if (!d_rays || !d_hits) {
    if (d_rays) omp_target_free(d_rays, gpu_device);
    if (d_hits) omp_target_free(d_hits, gpu_device);
    fatal_error("Failed to allocate OpenMP target memory for {} rays.", rays.size());
  }

  omp_target_memcpy(d_rays, rays.data(), ray_bytes, 0, 0, gpu_device, host_device);
  omp_target_memcpy(d_hits, hits.data(), hit_bytes, 0, 0, gpu_device, host_device);

  Timer timer;
  timer.start();
  rti->batch_ray_fire(tree, d_rays, d_hits, rays.size());
  timer.stop();

  omp_target_memcpy(hits.data(), d_hits, hit_bytes, 0, 0, host_device, gpu_device);
  omp_target_free(d_rays, gpu_device);
  omp_target_free(d_hits, gpu_device);

  size_t num_hits = 0;
  for (const auto& hit : hits) {
    if (hit.primID >= 0) ++num_hits;
  }

  return BenchmarkResult{"OpenMP target offload", timer.elapsed(), num_hits};
#endif
}

void print_result(const BenchmarkResult& result, size_t num_rays)
{
  const double throughput = result.trace_time > 0.0
    ? static_cast<double>(num_rays) / result.trace_time
    : 0.0;

  std::cout << result.label << '\n';
  std::cout << "  Trace time      = " << result.trace_time << " s\n";
  std::cout << "  Throughput      = " << throughput << " rays/s\n";
  std::cout << "  Hits / misses   = " << result.num_hits
            << " / " << (num_rays - result.num_hits) << '\n';
}

} // namespace

int main(int argc, char** argv)
{
  argparse::ArgumentParser args("XDG DPRT Benchmark Tool", "1.0", argparse::default_arguments::help);

  args.add_argument("filename").help("Path to the input file");

  args.add_argument("volume").help("Volume ID to query").scan<'i', int>();

  args.add_argument("-l", "--list")
    .default_value(false)
    .implicit_value(true)
    .help("List all volumes in the file and exit");

  args.add_argument("-n", "--num-rays")
    .default_value<uint32_t>(10'000'000)
    .help("Number of rays to generate for the benchmark")
    .scan<'u', uint32_t>();

  args.add_argument("-s", "--seed")
    .default_value<uint32_t>(12345)
    .help("Seed for deterministic ray generation")
    .scan<'u', uint32_t>();

  args.add_argument("-o", "-p", "--origin", "--position")
    .default_value(std::vector<double>{0.0, 0.0, 0.0})
    .help("Ray origin or source center")
    .scan<'g', double>().nargs(3);

  args.add_argument("-sr", "--source-radius")
    .default_value(0.0)
    .help("Radius of a uniformly sampled source blob around the origin")
    .scan<'g', double>();

  args.add_argument("-rt", "--rt-library")
    .default_value(std::string("DPRT"))
    .help("Ray tracing library to use. One of (DPRT, EMBREE)");

  args.add_description(
    "Benchmark DPRT ray tracing throughput for a given volume using a deterministic random spherical source. "
    "The same generated rays are reused across CPU and OpenMP target-offload backends.");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    return 1;
  }

  Timer wall_timer;
  Timer setup_timer;
  Timer gen_timer;
  wall_timer.start();
  setup_timer.start();

  const std::string rt_str = args.get<std::string>("--rt-library");
  RTLibrary rt_lib;
  if (rt_str == "DPRT") {
    rt_lib = RTLibrary::DPRT;
  } else if (rt_str == "EMBREE") {
    rt_lib = RTLibrary::EMBREE;
  } else {
    fatal_error("Invalid ray tracing library '{}' specified", rt_str);
  }

  std::shared_ptr<XDG> xdg = XDG::create(MeshLibrary::MOAB, rt_lib);
  const auto& mm = xdg->mesh_manager();
  mm->load_file(args.get<std::string>("filename"));
  mm->init();
  mm->parse_metadata();

  if (args.get<bool>("--list")) {
    std::cout << "Volumes:" << std::endl;
    for (auto volume : mm->volumes()) {
      std::cout << volume << std::endl;
    }
    return 0;
  }

  const MeshID volume = args.get<int>("volume");
  std::shared_ptr<DPRTRayTracer> dprt_rti;
  TreeID surface_tree = TREE_NONE;
  if (rt_lib == RTLibrary::DPRT) {
    dprt_rti = std::dynamic_pointer_cast<DPRTRayTracer>(xdg->ray_tracing_interface());
    if (!dprt_rti) fatal_error("Ray tracing interface is not DPRT.");
    const auto [registered_surface_tree, _unused_element_tree] = dprt_rti->register_volume(mm, volume);
    surface_tree = registered_surface_tree;
    dprt_rti->init();
  } else {
    xdg->prepare_volume_for_raytracing(volume);
  }

  setup_timer.stop();

  const size_t num_rays = args.get<uint32_t>("--num-rays");
  const uint32_t seed = args.get<uint32_t>("--seed");
  const Position origin = args.get<std::vector<double>>("--origin");
  const double source_radius = args.get<double>("--source-radius");

  const BoundingBox bbox = mm->volume_bounding_box(volume);
  const double t_max = std::max(1.0, 2.0 * bbox.max_chord_length());

  gen_timer.start();
  std::vector<DPRTRay> rays(num_rays);
  populate_seeded_rays(rays, origin, source_radius, seed, t_max);
  gen_timer.stop();

  std::cout << "Volume ID: " << volume << " with "
            << mm->num_volume_faces(volume) << " faces\n";
  std::cout << "Ray source origin: " << origin << '\n';
  std::cout << "Source radius: " << source_radius << '\n';
  std::cout << "Num rays: " << num_rays << '\n';
  std::cout << "Seed: " << seed << '\n';
  std::cout << "RT backend: " << rt_str << '\n';
#ifdef _OPENMP
  std::cout << "OpenMP max threads: " << omp_get_max_threads() << '\n';
#endif
  std::cout << "Setup time: " << setup_timer.elapsed() << " s\n";
  std::cout << "Ray generation time: " << gen_timer.elapsed() << " s\n";
  std::cout << "Ray tMax: " << t_max << "\n\n";

  BenchmarkResult result;
  if (rt_lib == RTLibrary::EMBREE) {
    result = run_cpu_trace(xdg, volume, rays);
  } else {
    result = run_target_trace(dprt_rti, surface_tree, rays);
  }

  wall_timer.stop();

  print_result(result, num_rays);

  std::cout << "\nFull wall-clock time = " << wall_timer.elapsed() << " s\n";

  return 0;
}
