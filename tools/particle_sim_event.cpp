#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/ranges.h>

#include "xdg/error.h"
#include "xdg/timer.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "argparse/argparse.hpp"

#include "particle_sim_event.h"

using namespace xdg;

int main(int argc, char** argv) {

// argument parsing
argparse::ArgumentParser args("XDG Particle Pseudo-Simulation", "1.0", argparse::default_arguments::help);

args.add_argument("filename")
    .help("Path to the input file");

args.add_argument("-v", "--verbose")
    .default_value(false)
    .implicit_value(true)
    .help("Enable verbose output of particle events");

args.add_argument("-m", "--mfp")
    .default_value(1.0)
    .help("Mean free path of the particles").scan<'g', double>();

args.add_argument("-n", "--n-particles")
    .default_value(1000000u)
    .help("Number of particles to simulate").scan<'u', uint32_t>();

args.add_argument("-e", "--max-events")
    .default_value(1000u)
    .help("Maximum number of events per particle").scan<'u', uint32_t>();

args.add_argument("-g", "--ipc-graveyard")
    .default_value(false)
    .implicit_value(true)
    .help("Treat the implicit complement as a graveyard (i.e. particles that enter it are killed)");

args.add_argument("-m", "--mesh-library")
    .help("Mesh library to use. One of (MOAB, LIBMESH)")
    .default_value("MOAB");

args.add_argument("-r", "--rt-library")
    .help("Ray tracing library to use. Event transport currently requires CUBQL")
    .default_value("CUBQL");

args.add_argument("--format")
  .default_value("human")
  .choices("human", "csv")
  .help("stdout format. Human readable (default) or csv");

try {
  args.parse_args(argc, argv);
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << args;
  exit(0);
}

Timer wall_timer;
wall_timer.start();

EventSimulationData sim_data;

// create a mesh manager
std::string mesh_str = args.get<std::string>("--mesh-library");
std::string rt_str = args.get<std::string>("--rt-library");
const std::string model_filename = args.get<std::string>("filename");
const std::string model_name = std::filesystem::path(model_filename).filename().string();
const std::string output_format = args.get<std::string>("--format");

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
  if (rt_lib == RTLibrary::GPRT)
    fatal_error("LibMesh is not currently supported with GPRT");
}
else
  fatal_error("Invalid mesh library '{}' specified", mesh_str);

// create an XDG instance with the specified mesh and ray tracing library
Timer xdg_setup_timer;
xdg_setup_timer.start();
std::shared_ptr<XDG> xdg = XDG::create(mesh_lib, rt_lib);
const auto& mm = xdg->mesh_manager();
mm->load_file(model_filename);
mm->init();
mm->parse_metadata();
xdg->prepare_raytracer();
xdg_setup_timer.stop();
sim_data.profiling.xdg_setup_s += xdg_setup_timer.elapsed();

sim_data.xdg_ = xdg;

const int model_num_volumes = mm->num_volumes();
const int model_num_surfaces = mm->num_surfaces();
const int model_num_volume_elements = mm->num_volume_elements();
const int model_num_vertices = mm->num_vertices();
std::uint64_t model_num_surface_primitives = 0;
for (MeshID surface : mm->surfaces()) {
  model_num_surface_primitives += static_cast<std::uint64_t>(mm->num_surface_faces(surface));
}

std::uint64_t model_num_volume_surface_refs = 0;
for (MeshID volume : mm->volumes()) {
  model_num_volume_surface_refs += static_cast<std::uint64_t>(mm->num_volume_faces(volume));
}

// update the mean free path
sim_data.mfp_ = args.get<double>("--mfp");

sim_data.verbose_particles_ = args.get<bool>("--verbose");
sim_data.implicit_complement_is_graveyard_ = args.get<bool>("--ipc-graveyard");
sim_data.n_particles_ = args.get<uint32_t>("--n-particles");
sim_data.max_events_ = args.get<uint32_t>("--max-events");

transport_particle_event_based(sim_data);

wall_timer.stop();
const double wall_time = wall_timer.elapsed();
const auto& profiling = sim_data.profiling;
const double ray_throughput =
  profiling.advance_ray_trace_s > 0.0
    ? static_cast<double>(profiling.rays_traced) / profiling.advance_ray_trace_s
    : 0.0;

const std::vector<std::string> csv_columns {
  "model",
  "mesh_library",
  "rt_library",
  "model_num_volumes",
  "model_num_surfaces",
  "model_num_volume_elements",
  "model_num_vertices",
  "model_num_surface_primitives",
  "model_num_volume_surface_refs",
  "n_particles",
  "max_events",
  "mean_free_path",
  "wall_time_s",
  "profile_xdg_setup_s",
  "profile_transport_s",
  "profile_advance_total_s",
  "profile_advance_pack_rays_s",
  "profile_advance_ray_trace_s",
  "profile_advance_update_particles_s",
  "profile_collision_s",
  "profile_surface_crossing_s",
  "profile_ray_batches",
  "profile_collision_calls",
  "profile_surface_crossing_calls",
  "profile_rays_traced",
  "profile_ray_throughput_rays_per_s"
};

const std::vector<std::string> csv_values {
  model_name,
  mesh_str,
  rt_str,
  fmt::format("{}", model_num_volumes),
  fmt::format("{}", model_num_surfaces),
  fmt::format("{}", model_num_volume_elements),
  fmt::format("{}", model_num_vertices),
  fmt::format("{}", model_num_surface_primitives),
  fmt::format("{}", model_num_volume_surface_refs),
  fmt::format("{}", sim_data.n_particles_),
  fmt::format("{}", sim_data.max_events_),
  fmt::format("{}", sim_data.mfp_),
  fmt::format("{}", wall_time),
  fmt::format("{}", profiling.xdg_setup_s),
  fmt::format("{}", profiling.transport_s),
  fmt::format("{}", profiling.advance_total_s),
  fmt::format("{}", profiling.advance_pack_rays_s),
  fmt::format("{}", profiling.advance_ray_trace_s),
  fmt::format("{}", profiling.advance_update_particles_s),
  fmt::format("{}", profiling.collision_s),
  fmt::format("{}", profiling.surface_crossing_s),
  fmt::format("{}", profiling.advance_calls),
  fmt::format("{}", profiling.collision_calls),
  fmt::format("{}", profiling.surface_crossing_calls),
  fmt::format("{}", profiling.rays_traced),
  fmt::format("{}", ray_throughput)
};

if (output_format == "csv") {
  std::cout << fmt::format("{}\n", fmt::join(csv_columns, ","));
  std::cout << fmt::format("{}\n", fmt::join(csv_values, ","));
} else {
  std::cout << "\nXDG event-based particle pseudo-simulation\n";
  std::cout << "----------------------------------------\n";
  std::cout << "Model                 : " << model_name << "\n";
  std::cout << "Mesh library          : " << mesh_str << "\n";
  std::cout << "Ray tracing library   : " << rt_str << "\n";
  std::cout << "Volumes               : " << model_num_volumes << "\n";
  std::cout << "Surfaces              : " << model_num_surfaces << "\n";
  std::cout << "Volume elements       : " << model_num_volume_elements << "\n";
  std::cout << "Vertices              : " << model_num_vertices << "\n";
  std::cout << "Surface primitives    : " << model_num_surface_primitives << "\n";
  std::cout << "Volume surface refs   : " << model_num_volume_surface_refs << "\n";
  std::cout << "Particles             : " << sim_data.n_particles_ << "\n";
  std::cout << "Max events/particle   : " << sim_data.max_events_ << "\n";
  std::cout << "Mean free path        : " << sim_data.mfp_ << "\n";
  std::cout << "----------------------------------------\n";
  std::cout << "Full wall-clock time  : " << wall_time << " s\n";
  std::cout << "XDG setup             : " << profiling.xdg_setup_s << " s\n";
  std::cout << "Transport time        : " << profiling.transport_s << " s\n";
  std::cout << "Advance total         : " << profiling.advance_total_s
            << " s (" << profiling.advance_calls << " calls)\n";
  std::cout << "  Ray batches         : " << profiling.advance_calls << "\n";
  std::cout << "  Pack rays           : " << profiling.advance_pack_rays_s << " s\n";
  std::cout << "  Ray trace           : " << profiling.advance_ray_trace_s << " s\n";
  std::cout << "  Rays traced         : "
            << fmt::format("{:.6e}", static_cast<double>(profiling.rays_traced)) << "\n";
  std::cout << "  Ray throughput      : " << ray_throughput << " rays/s\n";
  std::cout << "  Update particles    : " << profiling.advance_update_particles_s << " s\n";
  std::cout << "Collision events      : " << profiling.collision_s
            << " s (" << profiling.collision_calls << " calls)\n";
  std::cout << "Surface crossings     : " << profiling.surface_crossing_s
            << " s (" << profiling.surface_crossing_calls << " calls)\n";
  std::cout << "----------------------------------------\n";
  std::cout << "Cell track lengths\n";
  for (const auto& [cell, dist] : sim_data.cell_tracks) {
    std::cout << "Cell " << cell << "             : " << dist << "\n";
  }
}


return 0;
}
