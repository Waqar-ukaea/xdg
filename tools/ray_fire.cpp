#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>

#include "xdg/error.h"
#include "xdg/DPRT/ray_tracer.h"
#include "xdg/mesh_managers.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "argparse/argparse.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

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
      .help("Ray tracing library to use. One of (EMBREE, GPRT, DPRT)")
      .default_value("EMBREE");

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
else if (rt_str == "DPRT")
  rt_lib = RTLibrary::DPRT;
else
  fatal_error("Invalid ray tracing library '{}' specified", rt_str);

MeshLibrary mesh_lib;
if (mesh_str == "MOAB")
  mesh_lib = MeshLibrary::MOAB;
else if (mesh_str == "LIBMESH") 
  mesh_lib = MeshLibrary::LIBMESH;
else
  fatal_error("Invalid mesh library '{}' specified", mesh_str);

  // create a mesh manager
  std::shared_ptr<XDG> xdg = XDG::create(mesh_lib, rt_lib);
  const auto& mm = xdg->mesh_manager();
  mm->load_file(args.get<std::string>("filename"));
  mm->init();
  mm->parse_metadata();

  auto rti = xdg->ray_tracing_interface();

  if (args.get<bool>("--list")) {
    std::cout << "Volumes: " << std::endl;
    for (auto volume : mm->volumes()) {
      std::cout << volume << std::endl;
    }
    exit(0);
  }

  MeshID volume = args.get<int>("volume");

  Position origin = args.get<std::vector<double>>("--origin");
  Direction direction = args.get<std::vector<double>>("--direction");
  direction.normalize();

  std::cout << "Origin: " << origin[0] << ", " << origin[1] << ", " << origin[2] << std::endl;
  std::cout << "Direction: " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;

  std::pair<double, MeshID> result;

  xdg->prepare_volume_for_raytracing(volume);
  rti->init(); // Typically called during XDG::prepare_raytracer(). Required to build SBT after volume registration.

  if (rt_lib == RTLibrary::DPRT) {

    DPRTRay ray;
    ray.origin = {origin[0], origin[1], origin[2]};
    ray.direction = {direction[0], direction[1], direction[2]};
    ray.tMin = 0.0;
    ray.tMax = INFTY;

    DPRTHit hit;
    hit.primID = -1;
    hit.instID = -1;
    hit.geomUserData = 0;
    hit.t = INFTY;
    hit.u = 0.0;
    hit.v = 0.0;

    const int host_device = omp_get_initial_device();
    const int gpu_device = 0;
    auto* d_ray = static_cast<DPRTRay*>(omp_target_alloc(sizeof(DPRTRay), gpu_device));
    auto* d_hit = static_cast<DPRTHit*>(omp_target_alloc(sizeof(DPRTHit), gpu_device));

    omp_target_memcpy(d_ray, &ray, sizeof(DPRTRay), 0, 0, gpu_device, host_device);
    omp_target_memcpy(d_hit, &hit, sizeof(DPRTHit), 0, 0, gpu_device, host_device);

    xdg->batch_ray_fire(volume, d_ray, d_hit, 1);

    omp_target_memcpy(&hit, d_hit, sizeof(DPRTHit), 0, 0, host_device, gpu_device);
    omp_target_free(d_ray, gpu_device);
    omp_target_free(d_hit, gpu_device);

    result = {hit.t, static_cast<MeshID>(hit.geomUserData)};
  } 
  else {
    result = xdg->ray_fire(volume, origin, direction);
  }

  std::cout << std::setprecision(17) << "Distance: " << result.first << std::endl;
  std::cout << "Surface: " << result.second << std::endl;

  return 0;
}
