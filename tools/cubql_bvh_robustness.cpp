#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

#include "argparse/argparse.hpp"

#include "xdg/error.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/moab/mesh_manager.h"
#include "xdg/vec3da.h"

// This diagnostic intentionally inspects the cuBQL BVH owned by the ray
// tracer. Keep this access local to the tool rather than exposing diagnostic
// internals through XDG's public API.
#define private public
#include "xdg/cuBQL/ray_tracer.h"
#undef private

using namespace xdg;

namespace {

using Node = cuBQL::BinaryBVH<float, 3>::Node;

struct BuildDiagnostics {
  std::pair<double, MeshID> hit;
  MeshID hit_primitive {ID_NONE};
  bool point_in_volume {false};
  bool primitive_permutation {false};
  std::uint64_t reachable_nodes {0};
  std::uint64_t reachable_primitives {0};
  std::uint32_t total_nodes {0};
  std::uint32_t total_primitives {0};
  std::uint32_t expected_leaf {std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t bad_expected_path_node {std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t invalid_parent_bounds {0};
};

bool bounds_contain_point(const cuBQL::box3f& bounds, const Position& point)
{
  return point.x >= bounds.lower.x && point.x <= bounds.upper.x &&
         point.y >= bounds.lower.y && point.y <= bounds.upper.y &&
         point.z >= bounds.lower.z && point.z <= bounds.upper.z;
}

bool bounds_contain_bounds(const cuBQL::box3f& parent,
                           const cuBQL::box3f& child)
{
  return child.lower.x >= parent.lower.x && child.upper.x <= parent.upper.x &&
         child.lower.y >= parent.lower.y && child.upper.y <= parent.upper.y &&
         child.lower.z >= parent.lower.z && child.upper.z <= parent.upper.z;
}

std::uint32_t find_expected_primitive_ref(
  const std::shared_ptr<MeshManager>& mesh,
  MeshID volume,
  MeshID expected_surface,
  MeshID expected_primitive)
{
  std::uint32_t primitive_ref = 0;
  for (const MeshID surface : mesh->get_volume_surfaces(volume)) {
    for (const MeshID primitive : mesh->get_surface_faces(surface)) {
      if (surface == expected_surface && primitive == expected_primitive) {
        return primitive_ref;
      }
      ++primitive_ref;
    }
  }

  fatal_error("Expected surface {} primitive {} is not in volume {}",
              expected_surface,
              expected_primitive,
              volume);
  return std::numeric_limits<std::uint32_t>::max();
}

BuildDiagnostics inspect_build(const std::shared_ptr<MeshManager>& mesh,
                               MeshID volume,
                               const Position& origin,
                               const Direction& direction,
                               std::uint32_t expected_primitive_ref,
                               double expected_distance)
{
  BuildDiagnostics diagnostics;

  // A fresh tracer creates a fresh BVH. The mesh and OpenMP target runtime
  // remain alive across calls to preserve the original in-process repro.
  auto ray_tracer = std::make_shared<CuBQLRayTracer>();
  const TreeID tree = ray_tracer->create_surface_tree(mesh, volume);
  const auto& group = ray_tracer->tree_to_volume_group_.at(tree);

  std::vector<MeshID> hit_primitives;
  diagnostics.hit = ray_tracer->ray_fire(tree,
                                         origin,
                                         direction,
                                         INFTY,
                                         HitOrientation::EXITING,
                                         &hit_primitives);
  if (!hit_primitives.empty()) {
    diagnostics.hit_primitive = hit_primitives.back();
  }
  diagnostics.point_in_volume =
    ray_tracer->point_in_volume(tree, origin, &direction);

  diagnostics.total_nodes = group.bvh.numNodes;
  diagnostics.total_primitives = group.bvh.numPrims;

  std::vector<std::uint32_t> raw_primitive_ids(group.bvh.numPrims);
  const int primitive_copy_status = omp_target_memcpy(
    raw_primitive_ids.data(),
    group.bvh.primIDs,
    raw_primitive_ids.size() * sizeof(std::uint32_t),
    0,
    0,
    omp_get_initial_device(),
    group.gpu_id);
  if (primitive_copy_status != 0) {
    fatal_error("Failed to copy cuBQL primitive IDs to the host");
  }

  auto sorted_primitive_ids = raw_primitive_ids;
  std::sort(sorted_primitive_ids.begin(), sorted_primitive_ids.end());
  diagnostics.primitive_permutation = true;
  for (std::uint32_t i = 0; i < sorted_primitive_ids.size(); ++i) {
    if (sorted_primitive_ids[i] != i) {
      diagnostics.primitive_permutation = false;
      break;
    }
  }

  std::vector<Node> nodes(group.bvh.numNodes);
  const int node_copy_status = omp_target_memcpy(
    nodes.data(),
    group.bvh.nodes,
    nodes.size() * sizeof(Node),
    0,
    0,
    omp_get_initial_device(),
    group.gpu_id);
  if (node_copy_status != 0) {
    fatal_error("Failed to copy cuBQL BVH nodes to the host");
  }

  constexpr std::uint32_t no_node = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> parents(nodes.size(), no_node);
  std::vector<std::uint32_t> stack {0};
  while (!stack.empty()) {
    const std::uint32_t node_id = stack.back();
    stack.pop_back();
    if (node_id >= nodes.size()) {
      continue;
    }

    ++diagnostics.reachable_nodes;
    const auto& node = nodes[node_id];
    if (node.admin.count != 0) {
      diagnostics.reachable_primitives += node.admin.count;
      continue;
    }

    const std::uint32_t left_id = node.admin.offset;
    const std::uint32_t right_id = left_id + 1;
    if (left_id >= nodes.size() || right_id >= nodes.size()) {
      continue;
    }

    parents[left_id] = node_id;
    parents[right_id] = node_id;
    if (!bounds_contain_bounds(node.bounds, nodes[left_id].bounds)) {
      ++diagnostics.invalid_parent_bounds;
    }
    if (!bounds_contain_bounds(node.bounds, nodes[right_id].bounds)) {
      ++diagnostics.invalid_parent_bounds;
    }
    stack.push_back(left_id);
    stack.push_back(right_id);
  }

  const auto primitive_slot_it =
    std::find(raw_primitive_ids.begin(),
              raw_primitive_ids.end(),
              expected_primitive_ref);
  if (primitive_slot_it != raw_primitive_ids.end()) {
    const std::uint32_t primitive_slot = static_cast<std::uint32_t>(
      primitive_slot_it - raw_primitive_ids.begin());
    for (std::uint32_t node_id = 0; node_id < nodes.size(); ++node_id) {
      const auto& node = nodes[node_id];
      if (node.admin.count != 0 &&
          primitive_slot >= node.admin.offset &&
          primitive_slot < node.admin.offset + node.admin.count) {
        diagnostics.expected_leaf = node_id;
        break;
      }
    }
  }

  const Position expected_hit = origin + expected_distance * direction;
  for (std::uint32_t node_id = diagnostics.expected_leaf;
       node_id != no_node;
       node_id = parents[node_id]) {
    if (!bounds_contain_point(nodes[node_id].bounds, expected_hit)) {
      diagnostics.bad_expected_path_node = node_id;
      break;
    }
  }

  return diagnostics;
}

int node_for_output(std::uint32_t node_id)
{
  return node_id == std::numeric_limits<std::uint32_t>::max()
           ? -1
           : static_cast<int>(node_id);
}

} // namespace

int main(int argc, char** argv)
{
  argparse::ArgumentParser args("cuBQL BVH robustness diagnostic",
                                "1.0",
                                argparse::default_arguments::help);

  args.add_argument("filename")
    .help("Path to the input mesh");
  args.add_argument("volume")
    .help("Volume whose cuBQL surface BVH is rebuilt")
    .scan<'i', int>();
  args.add_argument("-o", "--origin")
    .required()
    .help("Exact ray origin")
    .scan<'g', double>()
    .nargs(3);
  args.add_argument("-d", "--direction")
    .required()
    .help("Exact ray direction; the tool does not renormalize it")
    .scan<'g', double>()
    .nargs(3);
  args.add_argument("--expected-surface")
    .required()
    .help("Expected hit surface ID")
    .scan<'i', int>();
  args.add_argument("--expected-primitive")
    .required()
    .help("Expected hit primitive ID")
    .scan<'i', int>();
  args.add_argument("--expected-distance")
    .required()
    .help("Expected hit distance used for ancestor-bound validation")
    .scan<'g', double>();
  args.add_argument("-n", "--num-builds")
    .default_value<std::uint32_t>(20)
    .help("Number of fresh BVHs to build inside this process")
    .scan<'u', std::uint32_t>();
  args.add_argument("-v", "--verbose")
    .default_value(false)
    .implicit_value(true)
    .help("Print diagnostics for every build, including successful builds");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& error) {
    std::cerr << error.what() << '\n' << args;
    return 1;
  }

  const auto origin_values = args.get<std::vector<double>>("--origin");
  const auto direction_values = args.get<std::vector<double>>("--direction");
  const Position origin {origin_values[0], origin_values[1], origin_values[2]};
  const Direction direction {
    direction_values[0], direction_values[1], direction_values[2]};
  const double direction_norm = direction.length();
  if (!std::isfinite(direction_norm) || direction_norm == 0.0) {
    fatal_error("Ray direction must be finite and nonzero");
  }

  const MeshID volume = args.get<int>("volume");
  const MeshID expected_surface = args.get<int>("--expected-surface");
  const MeshID expected_primitive = args.get<int>("--expected-primitive");
  const double expected_distance = args.get<double>("--expected-distance");
  const std::uint32_t num_builds = args.get<std::uint32_t>("--num-builds");
  const bool verbose = args.get<bool>("--verbose");
  if (num_builds == 0) {
    fatal_error("Number of BVH builds must be greater than zero");
  }

  // The mesh is deliberately loaded once. All BVHs below are built in this
  // process against this same mesh and OpenMP target runtime.
  auto mesh = std::make_shared<MOABMeshManager>();
  mesh->load_file(args.get<std::string>("filename"));
  mesh->init();
  mesh->parse_metadata();

  const std::uint32_t expected_primitive_ref =
    find_expected_primitive_ref(mesh,
                                volume,
                                expected_surface,
                                expected_primitive);

  std::cout << std::setprecision(17);
  std::cout << "volume=" << volume
            << " builds=" << num_builds
            << " origin=" << origin.x << ',' << origin.y << ',' << origin.z
            << " direction=" << direction.x << ',' << direction.y << ',' << direction.z
            << " direction_norm=" << direction_norm
            << " expected_surface=" << expected_surface
            << " expected_primitive=" << expected_primitive
            << " expected_distance=" << expected_distance
            << " expected_primitive_ref=" << expected_primitive_ref
            << '\n';

  std::uint32_t misses = 0;
  std::uint32_t unexpected_hits = 0;
  std::uint32_t bad_expected_paths = 0;
  std::uint32_t structurally_invalid_builds = 0;

  for (std::uint32_t build = 1; build <= num_builds; ++build) {
    const BuildDiagnostics diagnostics =
      inspect_build(mesh,
                    volume,
                    origin,
                    direction,
                    expected_primitive_ref,
                    expected_distance);

    const bool missed = diagnostics.hit.second == ID_NONE;
    const bool unexpected_hit =
      diagnostics.hit.second != expected_surface ||
      diagnostics.hit_primitive != expected_primitive;
    const bool bad_expected_path =
      diagnostics.bad_expected_path_node !=
      std::numeric_limits<std::uint32_t>::max();
    const bool structurally_invalid =
      !diagnostics.primitive_permutation ||
      diagnostics.reachable_primitives != diagnostics.total_primitives ||
      diagnostics.reachable_nodes + 1 != diagnostics.total_nodes ||
      diagnostics.invalid_parent_bounds != 0;

    misses += missed;
    unexpected_hits += unexpected_hit;
    bad_expected_paths += bad_expected_path;
    structurally_invalid_builds += structurally_invalid;

    if (verbose || missed || unexpected_hit || bad_expected_path || structurally_invalid) {
      std::cout << "build=" << build
                << " inside=" << diagnostics.point_in_volume
                << " surface=" << diagnostics.hit.second
                << " distance=" << diagnostics.hit.first
                << " primitive=" << diagnostics.hit_primitive
                << " permutation=" << diagnostics.primitive_permutation
                << " reachable_nodes=" << diagnostics.reachable_nodes
                << '/' << diagnostics.total_nodes
                << " reachable_primitives=" << diagnostics.reachable_primitives
                << '/' << diagnostics.total_primitives
                << " expected_leaf=" << node_for_output(diagnostics.expected_leaf)
                << " bad_expected_path_node="
                << node_for_output(diagnostics.bad_expected_path_node)
                << " invalid_parent_bounds=" << diagnostics.invalid_parent_bounds
                << '\n';
    }
  }

  std::cout << "summary builds=" << num_builds
            << " misses=" << misses
            << " unexpected_hits=" << unexpected_hits
            << " bad_expected_paths=" << bad_expected_paths
            << " structurally_invalid_builds=" << structurally_invalid_builds
            << '\n';

  return misses == 0 && unexpected_hits == 0 &&
             bad_expected_paths == 0 && structurally_invalid_builds == 0
           ? 0
           : 1;
}
