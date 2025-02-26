#include <iostream>
#include <memory>
#include <string>

#include "xdg/error.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "argparse/argparse.hpp"

#include "particle_sim.h"

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

args.add_argument("-l", "--library")
    .help("Mesh library to use. One of (MOAB, LIBMESH)")
    .default_value("MOAB");

  try {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(0);
  }

// Problem Setup
srand48(42);

SimulationData sim_data;
sim_data.n_particles_ = 1;

// create a mesh manager
std::shared_ptr<XDG> xdg {nullptr};
if (args.get<std::string>("--library") == "MOAB")
  xdg = XDG::create(MeshLibrary::MOAB);
else if (args.get<std::string>("--library") == "LIBMESH")
  xdg = XDG::create(MeshLibrary::LIBMESH);
else
  fatal_error("Invalid mesh library {} specified", args.get<std::string>("--library"));

sim_data.xdg_ = xdg;

const auto& mm = xdg->mesh_manager();
mm->load_file(args.get<std::string>("filename"));
mm->init();
mm->parse_metadata();
xdg->prepare_raytracer();

// update the mean free path
sim_data.mfp_ = args.get<double>("--mfp");

sim_data.verbose_particles_ = args.get<bool>("--verbose");

transport_particles(sim_data);

// report distances in each cell in a table
write_message("Cell Track Lengths");
write_message("-----------");
for (const auto& [cell, dist] : sim_data.cell_tracks) {
  write_message("Cell {}: {}", cell, dist);
}
write_message("-----------");


return 0;
}
