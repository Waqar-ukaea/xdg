#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"

#include "xdg/DPRT/ray_tracer.h"
#include "xdg/error.h"
#include "xdg/mesh_managers.h"
#include "xdg/xdg.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace xdg;

namespace {

DPRTvec3 random_unit_vector(std::mt19937_64& rng)
{
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  while (true) {
    const double x = dist(rng);
    const double y = dist(rng);
    const double z = dist(rng);
    const double norm = std::sqrt(x * x + y * y + z * z);
    if (norm > 1e-12) {
      return DPRTvec3{x / norm, y / norm, z / norm};
    }
  }
}

void populate_random_rays(std::vector<DPRTRay>& rays,
                          const BoundingBox& bbox,
                          uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> x_dist(bbox.min_x, bbox.max_x);
  std::uniform_real_distribution<double> y_dist(bbox.min_y, bbox.max_y);
  std::uniform_real_distribution<double> z_dist(bbox.min_z, bbox.max_z);

  const double t_min = 0.0;
  const double t_max = std::max(1.0, 2.0 * bbox.max_chord_length());

  for (auto& ray : rays) {
    ray.origin = DPRTvec3{x_dist(rng), y_dist(rng), z_dist(rng)};
    ray.direction = random_unit_vector(rng);
    ray.tMin = t_min;
    ray.tMax = t_max;
  }
}

} // namespace

int main(int argc, char** argv)
{
  argparse::ArgumentParser args("XDG DPRT Test Tool", "1.0", argparse::default_arguments::help);

  args.add_argument("filename").help("Path to the input file");

  args.add_argument("volume").help("Volume ID to query").scan<'i', int>();

  args.add_argument("-l", "--list")
    .default_value(false)
    .implicit_value(true)
    .help("List all volumes in the file and exit");

  args.add_argument("-n", "--num-rays")
    .default_value(1024)
    .help("Number of random rays to generate")
    .scan<'i', int>();

  args.add_argument("--seed")
    .default_value(1337)
    .help("PRNG seed for ray generation")
    .scan<'i', int>();

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    return 1;
  }

  std::shared_ptr<XDG> xdg = XDG::create(MeshLibrary::MOAB, RTLibrary::DPRT);
  const auto& mm = xdg->mesh_manager();
  mm->load_file(args.get<std::string>("filename"));
  mm->init();
  mm->parse_metadata();

  auto rti = std::dynamic_pointer_cast<DPRTRayTracer>(xdg->ray_tracing_interface());
  if (!rti) {
    fatal_error("Ray tracing interface is not DPRT.");
  }

  if (args.get<bool>("--list")) {
    std::cout << "Volumes:" << std::endl;
    for (auto volume : mm->volumes()) {
      std::cout << volume << std::endl;
    }
    return 0;
  }

  const MeshID volume = args.get<int>("volume");
  const int num_rays_input = args.get<int>("--num-rays");
  if (num_rays_input <= 0) {
    fatal_error("--num-rays must be > 0 (got {})", num_rays_input);
  }
  const size_t num_rays = static_cast<size_t>(num_rays_input);
  const uint64_t seed = static_cast<uint64_t>(args.get<int>("--seed"));

  const auto [surface_tree, _unused_element_tree] = rti->register_volume(mm, volume);
  rti->init();

  const BoundingBox bbox = mm->volume_bounding_box(volume);
  std::vector<DPRTRay> rays(num_rays);
  populate_random_rays(rays, bbox, seed);

  std::vector<DPRTHit> hits(num_rays);
  for (auto& hit : hits) {
    hit.primID = -1;
    hit.instID = -1;
    hit.geomUserData = 0;
    hit.t = INFTY;
    hit.u = 0.0;
    hit.v = 0.0;
  }

#ifdef _OPENMP
  const int host_device = omp_get_initial_device();
  const int gpu_device = 0;
  const size_t ray_bytes = num_rays * sizeof(DPRTRay);
  const size_t hit_bytes = num_rays * sizeof(DPRTHit);

  auto* d_rays = static_cast<DPRTRay*>(omp_target_alloc(ray_bytes, gpu_device));
  auto* d_hits = static_cast<DPRTHit*>(omp_target_alloc(hit_bytes, gpu_device));
  if (!d_rays || !d_hits) {
    if (d_rays) omp_target_free(d_rays, gpu_device);
    if (d_hits) omp_target_free(d_hits, gpu_device);
    fatal_error("Failed to allocate OpenMP target memory for DPRT rays/hits.");
  }

  omp_target_memcpy(d_rays, rays.data(), ray_bytes, 0, 0, gpu_device, host_device);
  omp_target_memcpy(d_hits, hits.data(), hit_bytes, 0, 0, gpu_device, host_device);

  rti->dpr_trace(surface_tree, d_rays, d_hits, num_rays);

  omp_target_memcpy(hits.data(), d_hits, hit_bytes, 0, 0, host_device, gpu_device);
  omp_target_free(d_rays, gpu_device);
  omp_target_free(d_hits, gpu_device);
#else
  fatal_error("dpr-test requires OpenMP target offload support in this DPRT build.");
#endif

  size_t num_hits = 0;
  for (const auto& hit : hits) {
    if (hit.primID >= 0) ++num_hits;
  }

  std::cout << "Generated rays: " << num_rays << '\n';
  std::cout << "Hits: " << num_hits << '\n';
  std::cout << "Misses: " << (num_rays - num_hits) << '\n';

  const size_t preview_count = std::min<size_t>(5, hits.size());
  std::cout << "First " << preview_count << " hits:\n";
  for (size_t i = 0; i < preview_count; ++i) {
    const auto& hit = hits[i];
    std::cout << "  [" << i << "] primID=" << hit.primID
              << ", instID=" << hit.instID
              << ", geomUserData=" << hit.geomUserData
              << ", t=" << std::setprecision(17) << hit.t
              << ", (u,v)=(" << hit.u << ", " << hit.v << ")\n";
  }

  return 0;
}
