#include <array>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>

#include "xdg/error.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "argparse/argparse.hpp"

using namespace xdg;

int main(int argc, char** argv) {

  argparse::ArgumentParser args("XDG Ray Fire Tool", "1.0", argparse::default_arguments::help);

  args.add_argument("filename")
    .help("Path to the input file");

  args.add_argument("volume")
    .help("Volume ID to query").scan<'i', int>();

  args.add_argument("-l", "--list")
    .default_value(false)
    .implicit_value(true)
    .help("List all volumes in the file and exit");

  args.add_argument("-o", "-p", "--origin", "--position")
    .default_value(std::vector<double>{0.0, 0.0, 0.0})
    .help("Ray origin/position").scan<'g', double>().nargs(3);

  args.add_argument("-d", "--direction")
    .default_value(std::vector<double>{0.0, 0.0, 1.0})
    .help("Ray direction").scan<'g', double>().nargs(3);

  args.add_argument("-m", "--mesh-library")
      .help("Mesh library to use. One of (MOAB, LIBMESH)")
      .default_value("MOAB");

  args.add_argument("-r", "--rt-library")
      .help("Ray tracing library to use. One of (EMBREE, GPRT, CUBQL)")
      .default_value("EMBREE");

  auto& bvh_build_scope = args.add_mutually_exclusive_group();
  bvh_build_scope.add_argument("--single-volume-bvh")
      .default_value(false)
      .implicit_value(true)
      .help("Build a BVH only for the queried volume");
  bvh_build_scope.add_argument("--full-model-bvh")
      .default_value(false)
      .implicit_value(true)
      .help("Build BVHs for the full model (default)");

  args.add_argument("--batch")
      .default_value(false)
      .implicit_value(true)
      .help("Fire the ray through the backend batch ray-fire path using a one-ray batch");

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

RTLibrary rt_lib;
if (rt_str == "EMBREE")
  rt_lib = RTLibrary::EMBREE;
else if (rt_str == "GPRT")
  rt_lib = RTLibrary::GPRT;
else if (rt_str == "CUBQL")
  rt_lib = RTLibrary::CUBQL;
else
  fatal_error("Invalid ray tracing library '{}' specified", rt_str);

MeshLibrary mesh_lib;
if (mesh_str == "MOAB")
  mesh_lib = MeshLibrary::MOAB;
else if (mesh_str == "LIBMESH") {
  mesh_lib = MeshLibrary::LIBMESH;
  if (rt_lib == RTLibrary::GPRT || rt_lib == RTLibrary::CUBQL)
    fatal_error("LibMesh is not currently supported with GPRT");
}
else
  fatal_error("Invalid mesh library '{}' specified", mesh_str);

  // create an XDG instance with the specified mesh and ray tracing library
  std::shared_ptr<XDG> xdg = XDG::create(mesh_lib, rt_lib);
  const auto& mm = xdg->mesh_manager();
  mm->load_file(args.get<std::string>("filename"));
  mm->init();
  mm->parse_metadata();

  const MeshID volume = args.get<int>("volume");
  const bool single_volume_bvh = args.get<bool>("--single-volume-bvh");

  if (args.get<bool>("--list")) {
    std::cout << "Volumes: " << std::endl;
    for (auto volume : mm->volumes()) {
      std::cout << volume << std::endl;
    }
    exit(0);
  }

  if (single_volume_bvh) {
    xdg->prepare_volume_for_raytracing(volume);
    xdg->ray_tracing_interface()->init();
    std::cout << "BVH construction: single volume " << volume << std::endl;
  } else {
    xdg->prepare_raytracer();
    std::cout << "BVH construction: full model" << std::endl;
  }

  Position origin = args.get<std::vector<double>>("--origin");
  Direction direction = args.get<std::vector<double>>("--direction");
  direction.normalize();
  const bool use_batch = args.get<bool>("--batch");

  std::cout << "Origin: " << origin[0] << ", " << origin[1] << ", " << origin[2] << std::endl;
  std::cout << "Direction: " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;

  std::pair<double, MeshID> result;

  if (use_batch) {
    XDGRayHit host_ray_hit {};
    host_ray_hit.origin[0] = origin.x;
    host_ray_hit.origin[1] = origin.y;
    host_ray_hit.origin[2] = origin.z;
    host_ray_hit.direction[0] = direction.x;
    host_ray_hit.direction[1] = direction.y;
    host_ray_hit.direction[2] = direction.z;
    host_ray_hit.t_min = TINY_BIT;
    host_ray_hit.t_max = INFTY;
    host_ray_hit.volume = volume;
    host_ray_hit.last_hit_primitive = ID_NONE;
    host_ray_hit.distance = INFTY;
    host_ray_hit.surface = ID_NONE;
    host_ray_hit.primitive = ID_NONE;
    host_ray_hit.point_in_volume = ID_NONE;
    host_ray_hit.next_volume = ID_NONE;
    host_ray_hit.boundary_condition = UNSET;
    host_ray_hit.normal[0] = 0.0;
    host_ray_hit.normal[1] = 0.0;
    host_ray_hit.normal[2] = 0.0;

    const XDGRayHit input_ray_hit = host_ray_hit;

    XDGRayHitBuffer ray_hits = xdg->allocate_ray_hits(1);

    const auto fire_batch_once = [&](const Direction& query_direction) {
      host_ray_hit = input_ray_hit;
      host_ray_hit.direction[0] = query_direction.x;
      host_ray_hit.direction[1] = query_direction.y;
      host_ray_hit.direction[2] = query_direction.z;

      const int upload_status = omp_target_memcpy(ray_hits.data,
                                                  &host_ray_hit,
                                                  sizeof(XDGRayHit),
                                                  0,
                                                  0,
                                                  ray_hits.device_id,
                                                  omp_get_initial_device());
      if (upload_status != 0) {
        xdg->free_ray_hits(ray_hits);
        fatal_error("Failed to upload one-ray batch record to the target device.");
      }

      xdg->ray_fire_batch(ray_hits);

      const int download_status = omp_target_memcpy(&host_ray_hit,
                                                    ray_hits.data,
                                                    sizeof(XDGRayHit),
                                                    0,
                                                    0,
                                                    omp_get_initial_device(),
                                                    ray_hits.device_id);
      if (download_status != 0) {
        xdg->free_ray_hits(ray_hits);
        fatal_error("Failed to download one-ray batch record from the target device.");
      }
    };

    result = xdg->ray_fire(volume, origin, direction);

    std::cout << "Scalar Ray Fire Result after confirmed miss: distance=" << result.first
              << " surface=" << result.second << std::endl;

    fire_batch_once(direction);


    result = {host_ray_hit.distance, host_ray_hit.surface};
    std::cout << "Primitive: " << host_ray_hit.primitive << std::endl;
    std::cout << "Next volume: " << host_ray_hit.next_volume << std::endl;
    std::cout << "Boundary condition: " << host_ray_hit.boundary_condition << std::endl;

    if (host_ray_hit.surface == ID_NONE) {
      struct DirectionProbe {
        const char* name;
        Direction direction;
      };

      const double corner_component = 1.0 / std::sqrt(3.0);
      const std::array<DirectionProbe, 14> probes {{
        {"+X", { 1.0,  0.0,  0.0}},
        {"-X", {-1.0,  0.0,  0.0}},
        {"+Y", { 0.0,  1.0,  0.0}},
        {"-Y", { 0.0, -1.0,  0.0}},
        {"+Z", { 0.0,  0.0,  1.0}},
        {"-Z", { 0.0,  0.0, -1.0}},
        {"+++", { corner_component,  corner_component,  corner_component}},
        {"++-", { corner_component,  corner_component, -corner_component}},
        {"+-+", { corner_component, -corner_component,  corner_component}},
        {"+--", { corner_component, -corner_component, -corner_component}},
        {"-++", {-corner_component,  corner_component,  corner_component}},
        {"-+-", {-corner_component,  corner_component, -corner_component}},
        {"--+", {-corner_component, -corner_component,  corner_component}},
        {"---", {-corner_component, -corner_component, -corner_component}}
      }};

      int probe_misses = 0;
      int probe_hits = 0;

      std::cout << "Miss direction diagnostic: " << probes.size()
                << " cube face-normal and corner directions against the same XDG setup"
                << std::endl;

      for (const auto& probe : probes) {
        fire_batch_once(probe.direction);

        const bool probe_missed = host_ray_hit.surface == ID_NONE;
        probe_misses += probe_missed;
        probe_hits += !probe_missed;

        std::cout << "Direction probe " << probe.name
                  << ": direction="
                  << std::setprecision(17)
                  << probe.direction.x << ','
                  << probe.direction.y << ','
                  << probe.direction.z
                  << " distance=" << host_ray_hit.distance
                  << " surface=" << host_ray_hit.surface
                  << " primitive=" << host_ray_hit.primitive
                  << " missed=" << std::boolalpha << probe_missed
                  << std::noboolalpha << std::endl;
      }

      std::cout << "Miss direction summary: probes=" << probes.size()
                << " misses=" << probe_misses
                << " hits=" << probe_hits << std::endl;
    }

    xdg->free_ray_hits(ray_hits);
  } else {
    result = xdg->ray_fire(volume, origin, direction);
  }

  std::cout << std::setprecision(17) << "Distance: " << result.first << std::endl;
  std::cout << "Surface: " << result.second << std::endl;

  return 0;
}
