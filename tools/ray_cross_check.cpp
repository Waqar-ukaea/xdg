#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "xdg/available_device_probe.h"
#include "xdg/constants.h"
#include "xdg/device_ray.h"
#include "xdg/xdg.h"

#ifdef XDG_ENABLE_GPRT
#include "xdg/gprt/ray_tracer.h"
#endif

using namespace xdg;

namespace
{

constexpr std::uint64_t DEFAULT_NUM_RAYS = 10'000'000'000ULL;
constexpr std::uint32_t DEFAULT_REPETITIONS = 10;
constexpr std::uint64_t DEFAULT_BATCH_SIZE = 1'000'000ULL;
constexpr std::uint64_t DEFAULT_SEED = 12345ULL;
constexpr double DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-6;
constexpr double DEFAULT_RELATIVE_TOLERANCE = 1.0e-12;
constexpr double TWO_PI = 6.283185307179586476925286766559;

struct PairStats
{
  std::string backend_a;
  std::string backend_b;
  std::uint64_t num_queries {0};
  std::uint64_t matched_hits {0};
  std::uint64_t matched_misses {0};
  std::uint64_t hit_miss_disagreements {0};
  std::uint64_t surface_disagreements_on_hits {0};
  std::uint64_t distance_disagreements_on_hits {0};
  double max_absolute_distance_difference {0.0};
  double max_relative_distance_difference {0.0};
};

struct BackendCase
{
  RTLibrary library;
  std::string name;
  std::shared_ptr<XDG> xdg;
  XDGRayHitBuffer device_buffer;
  std::size_t device_buffer_capacity {0};

  BackendCase(RTLibrary library_, std::string name_, std::shared_ptr<XDG> xdg_)
      : library(library_), name(std::move(name_)), xdg(std::move(xdg_))
  {
  }

  ~BackendCase()
  {
    if (device_buffer.data)
    {
      xdg->free_ray_hits(device_buffer);
    }
  }

  BackendCase(const BackendCase&) = delete;
  BackendCase& operator=(const BackendCase&) = delete;
};

std::string uppercase(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::vector<std::string> split_backends(const std::string& value)
{
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ','))
  {
    token.erase(
        std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c); }),
        token.end());
    if (!token.empty())
      result.push_back(uppercase(token));
  }
  return result;
}

bool backend_is_available(RTLibrary library)
{
  switch (library)
  {
  case RTLibrary::EMBREE:
#ifdef XDG_ENABLE_EMBREE
    return true;
#else
    return false;
#endif

  case RTLibrary::GPRT:
#ifdef XDG_ENABLE_GPRT
    return system_has_vk_device();
#else
    return false;
#endif

  case RTLibrary::CUBQL:
#ifdef XDG_ENABLE_CUBQL
    return system_has_omp_target_device();
#else
    return false;
#endif
  }
  return false;
}

RTLibrary parse_backend(const std::string& name)
{
  const std::string normalized = uppercase(name);
  if (normalized == "EMBREE")
    return RTLibrary::EMBREE;
  if (normalized == "GPRT")
    return RTLibrary::GPRT;
  if (normalized == "CUBQL")
    return RTLibrary::CUBQL;
  throw std::runtime_error("Unknown ray-tracing backend '" + name + "'");
}

std::vector<RTLibrary> selected_backends(const std::string& selection)
{
  std::vector<RTLibrary> result;
  const std::string normalized = uppercase(selection);

  if (normalized == "ALL")
  {
    for (const RTLibrary library : {RTLibrary::EMBREE, RTLibrary::GPRT, RTLibrary::CUBQL})
    {
      if (backend_is_available(library))
        result.push_back(library);
    }
    return result;
  }

  for (const auto& name : split_backends(selection))
  {
    const RTLibrary library = parse_backend(name);
    if (!backend_is_available(library))
    {
      throw std::runtime_error("Requested backend " + RT_LIB_TO_STR.at(library) +
                               " is not compiled in or has no usable device");
    }
    if (std::find(result.begin(), result.end(), library) == result.end())
    {
      result.push_back(library);
    }
  }
  return result;
}

std::uint64_t splitmix64(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

double unit_double(std::uint64_t value)
{
  return static_cast<double>(value >> 11U) * 0x1.0p-53;
}

std::array<double, 3> direction_for(std::uint64_t ray_index, std::uint64_t seed)
{
  const std::uint64_t key = seed + ray_index * 0x9e3779b97f4a7c15ULL;
  const double z = 2.0 * unit_double(splitmix64(key)) - 1.0;
  const double phi = TWO_PI * unit_double(splitmix64(key ^ 0xd1b54a32d192ed03ULL));
  const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
  return {radial * std::cos(phi), radial * std::sin(phi), z};
}

void generate_rays(std::vector<XDGRayHit>& rays, std::uint64_t first_ray, std::uint64_t seed,
                   MeshID volume, const Position& origin)
{
  for (std::size_t local_index = 0; local_index < rays.size(); ++local_index)
  {
    const auto direction = direction_for(first_ray + static_cast<std::uint64_t>(local_index), seed);

    XDGRayHit ray_hit {};
    ray_hit.origin[0] = origin.x;
    ray_hit.origin[1] = origin.y;
    ray_hit.origin[2] = origin.z;
    ray_hit.direction[0] = direction[0];
    ray_hit.direction[1] = direction[1];
    ray_hit.direction[2] = direction[2];
    ray_hit.t_min = 0.0;
    ray_hit.t_max = INFTY;
    ray_hit.volume = volume;
    ray_hit.last_hit_primitive = ID_NONE;
    ray_hit.distance = INFTY;
    ray_hit.surface = ID_NONE;
    ray_hit.primitive = ID_NONE;
    ray_hit.point_in_volume = OUTSIDE;
    ray_hit.next_volume = ID_NONE;
    ray_hit.boundary_condition = UNSET;
    rays[local_index] = ray_hit;
  }
}

std::unique_ptr<BackendCase> make_backend_case(const std::string& filename, MeshID volume,
                                               RTLibrary library)
{
  auto xdg = XDG::create(MeshLibrary::MOAB, library);
  const auto& mesh_manager = xdg->mesh_manager();
  mesh_manager->load_file(filename);
  mesh_manager->init();
  mesh_manager->parse_metadata();

  const auto& volumes = mesh_manager->volumes();
  if (std::find(volumes.begin(), volumes.end(), volume) == volumes.end())
  {
    throw std::runtime_error("Volume " + std::to_string(volume) + " is not present in " + filename);
  }

  xdg->prepare_volume_for_raytracing(volume);
  xdg->ray_tracing_interface()->init();
  return std::make_unique<BackendCase>(library, RT_LIB_TO_STR.at(library), std::move(xdg));
}

void ensure_cubql_buffer(BackendCase& backend, std::size_t count)
{
#ifdef XDG_ENABLE_CUBQL
  if (backend.device_buffer_capacity >= count)
    return;
  if (backend.device_buffer.data)
    backend.xdg->free_ray_hits(backend.device_buffer);
  backend.device_buffer = backend.xdg->allocate_ray_hits(count);
  backend.device_buffer_capacity = count;
#else
  (void)backend;
  (void)count;
  throw std::runtime_error("cuBQL support is not compiled in");
#endif
}

void run_backend_batch(BackendCase& backend, const std::vector<XDGRayHit>& input,
                       std::vector<XDGRayHit>& output)
{
  output = input;

  if (backend.library == RTLibrary::EMBREE)
  {
#ifdef XDG_ENABLE_EMBREE
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < output.size(); ++i)
    {
      const auto& ray = input[i];
      const auto result =
          backend.xdg->ray_fire(ray.volume, Position {ray.origin[0], ray.origin[1], ray.origin[2]},
                                Direction {ray.direction[0], ray.direction[1], ray.direction[2]},
                                ray.t_max, HitOrientation::EXITING);
      output[i].distance = result.first;
      output[i].surface = result.second;
      output[i].primitive = ID_NONE;
    }
    return;
#else
    throw std::runtime_error("Embree support is not compiled in");
#endif
  }

  if (backend.library == RTLibrary::GPRT)
  {
#ifdef XDG_ENABLE_GPRT
    auto gprt = std::dynamic_pointer_cast<GPRTRayTracer>(backend.xdg->ray_tracing_interface());
    if (!gprt)
      throw std::runtime_error("Failed to access GPRT ray tracer");
    gprt->ray_fire_host_batch(output, HitOrientation::EXITING);
    return;
#else
    throw std::runtime_error("GPRT support is not compiled in");
#endif
  }

  if (backend.library == RTLibrary::CUBQL)
  {
#if defined(XDG_ENABLE_CUBQL) && defined(_OPENMP)
    ensure_cubql_buffer(backend, output.size());
    const std::size_t bytes = output.size() * sizeof(XDGRayHit);
    const int host_device = omp_get_initial_device();
    const int gpu_device = backend.device_buffer.device_id;

    int status = omp_target_memcpy(backend.device_buffer.data, output.data(), bytes, 0, 0,
                                   gpu_device, host_device);
    if (status != 0)
    {
      throw std::runtime_error("Failed to copy cross-check rays to cuBQL device");
    }

    const XDGRayHitBuffer active_buffer {backend.device_buffer.data, output.size(), gpu_device};
    backend.xdg->ray_fire_batch(active_buffer, HitOrientation::EXITING);

    status = omp_target_memcpy(output.data(), backend.device_buffer.data, bytes, 0, 0, host_device,
                               gpu_device);
    if (status != 0)
    {
      throw std::runtime_error("Failed to copy cuBQL cross-check results to host");
    }
    return;
#else
    throw std::runtime_error("cuBQL cross-check execution requires OpenMP target support");
#endif
  }

  throw std::runtime_error("Unsupported ray-tracing backend");
}

void compare_batch(const std::vector<XDGRayHit>& a, const std::vector<XDGRayHit>& b,
                   double absolute_tolerance, double relative_tolerance, PairStats& stats)
{
  if (a.size() != b.size())
  {
    throw std::runtime_error("Backend result batches have different sizes");
  }

  for (std::size_t i = 0; i < a.size(); ++i)
  {
    ++stats.num_queries;
    const bool a_hit = a[i].surface != ID_NONE;
    const bool b_hit = b[i].surface != ID_NONE;

    if (a_hit != b_hit)
    {
      ++stats.hit_miss_disagreements;
      continue;
    }
    if (!a_hit)
    {
      ++stats.matched_misses;
      continue;
    }
    ++stats.matched_hits;

    if (a[i].surface != b[i].surface)
    {
      ++stats.surface_disagreements_on_hits;
    }

    const double distance_a = a[i].distance;
    const double distance_b = b[i].distance;
    if (!std::isfinite(distance_a) || !std::isfinite(distance_b))
    {
      ++stats.distance_disagreements_on_hits;
      stats.max_absolute_distance_difference = std::numeric_limits<double>::infinity();
      stats.max_relative_distance_difference = std::numeric_limits<double>::infinity();
      continue;
    }

    const double absolute_difference = std::abs(distance_a - distance_b);
    const double distance_scale = std::max(std::abs(distance_a), std::abs(distance_b));
    const double relative_difference =
        distance_scale > 0.0 ? absolute_difference / distance_scale : absolute_difference;

    stats.max_absolute_distance_difference =
        std::max(stats.max_absolute_distance_difference, absolute_difference);
    stats.max_relative_distance_difference =
        std::max(stats.max_relative_distance_difference, relative_difference);

    const double tolerance = absolute_tolerance + relative_tolerance * distance_scale;
    if (absolute_difference > tolerance)
    {
      ++stats.distance_disagreements_on_hits;
    }
  }
}

std::vector<PairStats> make_pair_stats(const std::vector<std::unique_ptr<BackendCase>>& backends)
{
  std::vector<PairStats> result;
  for (std::size_t a = 0; a < backends.size(); ++a)
  {
    for (std::size_t b = a + 1; b < backends.size(); ++b)
    {
      PairStats stats;
      stats.backend_a = backends[a]->name;
      stats.backend_b = backends[b]->name;
      result.push_back(std::move(stats));
    }
  }
  return result;
}

void merge_stats(PairStats& aggregate, const PairStats& run)
{
  aggregate.num_queries += run.num_queries;
  aggregate.matched_hits += run.matched_hits;
  aggregate.matched_misses += run.matched_misses;
  aggregate.hit_miss_disagreements += run.hit_miss_disagreements;
  aggregate.surface_disagreements_on_hits += run.surface_disagreements_on_hits;
  aggregate.distance_disagreements_on_hits += run.distance_disagreements_on_hits;
  aggregate.max_absolute_distance_difference =
      std::max(aggregate.max_absolute_distance_difference, run.max_absolute_distance_difference);
  aggregate.max_relative_distance_difference =
      std::max(aggregate.max_relative_distance_difference, run.max_relative_distance_difference);
}

std::string csv_escape(const std::string& value)
{
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char c : value)
  {
    if (c == '"')
      escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

void write_csv_header(std::ostream& output)
{
  output << "scope,model,volume,origin_x,origin_y,origin_z,backend_a,backend_b,"
            "repetition,seed,num_queries,matched_hits,matched_misses,"
            "hit_miss_disagreements,"
            "surface_disagreements_on_hits,distance_disagreements_on_hits,"
            "max_absolute_distance_difference_model_units,"
            "max_relative_distance_difference,absolute_tolerance_model_units,"
            "relative_tolerance,batch_size\n";
}

void write_csv_row(std::ostream& output, const std::string& scope, const std::string& model,
                   MeshID volume, const Position& origin, const PairStats& stats,
                   const std::string& repetition, const std::string& seed,
                   double absolute_tolerance, double relative_tolerance, std::size_t batch_size)
{
  output << csv_escape(scope) << ',' << csv_escape(model) << ',' << volume << ','
         << std::setprecision(17) << origin.x << ',' << origin.y << ',' << origin.z << ','
         << csv_escape(stats.backend_a) << ',' << csv_escape(stats.backend_b) << ',' << repetition
         << ',' << seed << ',' << stats.num_queries << ',' << stats.matched_hits << ','
         << stats.matched_misses << ',' << stats.hit_miss_disagreements << ','
         << stats.surface_disagreements_on_hits << ',' << stats.distance_disagreements_on_hits
         << ',' << stats.max_absolute_distance_difference << ','
         << stats.max_relative_distance_difference << ',' << absolute_tolerance << ','
         << relative_tolerance << ',' << batch_size << '\n';
}

void print_stats(const PairStats& stats)
{
  std::cout << "  " << stats.backend_a << "--" << stats.backend_b << '\n'
            << "    Queries                    : " << stats.num_queries << '\n'
            << "    Matched hits               : " << stats.matched_hits << '\n'
            << "    Matched misses             : " << stats.matched_misses << '\n'
            << "    Hit/miss disagreements     : " << stats.hit_miss_disagreements << '\n'
            << "    Surface disagreements/hits : " << stats.surface_disagreements_on_hits << '\n'
            << "    Distance disagreements/hits: " << stats.distance_disagreements_on_hits << '\n'
            << "    Maximum absolute difference: " << std::scientific << std::setprecision(6)
            << stats.max_absolute_distance_difference << " model units\n"
            << "    Maximum relative difference: " << stats.max_relative_distance_difference << '\n'
            << std::defaultfloat;
}

} // namespace

int main(int argc, char** argv)
{
  argparse::ArgumentParser args("XDG ray-tracer cross-check", "1.0",
                                argparse::default_arguments::help);

  args.add_argument("filename").help("Path to the input MOAB file");

  args.add_argument("volume").help("Mesh volume ID containing the ray origin").scan<'i', int>();

  args.add_argument("-n", "--num-rays")
      .default_value<std::uint64_t>(std::uint64_t {DEFAULT_NUM_RAYS})
      .help("Number of rays compared per repetition")
      .scan<'u', std::uint64_t>();

  args.add_argument("-r", "--repetitions")
      .default_value<std::uint32_t>(std::uint32_t {DEFAULT_REPETITIONS})
      .help("Number of distinct deterministic ray populations")
      .scan<'u', std::uint32_t>();

  args.add_argument("-b", "--batch-size")
      .default_value<std::uint64_t>(std::uint64_t {DEFAULT_BATCH_SIZE})
      .help("Number of rays retained and traced at once")
      .scan<'u', std::uint64_t>();

  args.add_argument("-s", "--seed")
      .default_value<std::uint64_t>(std::uint64_t {DEFAULT_SEED})
      .help("Base seed for deterministic counter-based ray generation")
      .scan<'u', std::uint64_t>();

  args.add_argument("--repeat-same-rays")
      .default_value(false)
      .implicit_value(true)
      .help("Reuse the same ray population instead of varying it by repetition");

  args.add_argument("--backends")
      .default_value(std::string("ALL"))
      .help("Comma-separated backends (EMBREE,GPRT,CUBQL) or ALL");

  args.add_argument("-o", "--origin")
      .help("Fixed ray origin; defaults to the selected volume bounding-box "
            "center")
      .scan<'g', double>()
      .nargs(3);

  args.add_argument("--absolute-tolerance")
      .default_value(DEFAULT_ABSOLUTE_TOLERANCE)
      .help("Absolute hit-distance tolerance in model units")
      .scan<'g', double>();

  args.add_argument("--relative-tolerance")
      .default_value(DEFAULT_RELATIVE_TOLERANCE)
      .help("Relative hit-distance tolerance")
      .scan<'g', double>();

  args.add_argument("--output")
      .default_value(std::string("ray-cross-check.csv"))
      .help("CSV output path");

  args.add_argument("--overwrite")
      .default_value(false)
      .implicit_value(true)
      .help("Allow replacement of an existing output CSV");

  args.add_description("Streams deterministic, identical rays through every selected backend "
                       "and "
                       "accumulates pairwise hit, surface, and distance agreement statistics. "
                       "Defaults to 10 repetitions of 10 billion rays without storing the full "
                       "ray population in memory.");

  try
  {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& error)
  {
    std::cerr << error.what() << "\n\n" << args;
    return 1;
  }

  try
  {
    const std::string filename = args.get<std::string>("filename");
    const MeshID volume = args.get<int>("volume");
    const std::uint64_t num_rays = args.get<std::uint64_t>("--num-rays");
    const std::uint32_t repetitions = args.get<std::uint32_t>("--repetitions");
    const std::uint64_t requested_batch_size = args.get<std::uint64_t>("--batch-size");
    const std::uint64_t base_seed = args.get<std::uint64_t>("--seed");
    const bool repeat_same_rays = args.get<bool>("--repeat-same-rays");
    const double absolute_tolerance = args.get<double>("--absolute-tolerance");
    const double relative_tolerance = args.get<double>("--relative-tolerance");
    const std::string output_filename = args.get<std::string>("--output");
    const bool overwrite = args.get<bool>("--overwrite");

    if (num_rays == 0)
      throw std::runtime_error("--num-rays must be positive");
    if (repetitions == 0)
    {
      throw std::runtime_error("--repetitions must be positive");
    }
    if (requested_batch_size == 0 || requested_batch_size > std::numeric_limits<std::size_t>::max())
    {
      throw std::runtime_error("--batch-size is outside the supported range");
    }
    if (absolute_tolerance < 0.0 || relative_tolerance < 0.0)
    {
      throw std::runtime_error("Distance tolerances cannot be negative");
    }
    if (num_rays > std::numeric_limits<std::uint64_t>::max() / repetitions)
    {
      throw std::runtime_error("Total ray count overflows a 64-bit counter");
    }
    if (std::filesystem::exists(output_filename) && !overwrite)
    {
      throw std::runtime_error("Output file already exists: " + output_filename +
                               " (use --overwrite to replace it)");
    }

    const std::size_t batch_size =
        static_cast<std::size_t>(std::min<std::uint64_t>(requested_batch_size, num_rays));
    const auto libraries = selected_backends(args.get<std::string>("--backends"));
    if (libraries.size() < 2)
    {
      throw std::runtime_error("At least two requested ray-tracing backends must be available");
    }
    if (std::find(libraries.begin(), libraries.end(), RTLibrary::GPRT) != libraries.end() &&
        batch_size > std::numeric_limits<std::uint32_t>::max())
    {
      throw std::runtime_error("GPRT batch size exceeds its 32-bit launch limit");
    }

    std::vector<std::unique_ptr<BackendCase>> backends;
    backends.reserve(libraries.size());
    for (const RTLibrary library : libraries)
    {
      std::cout << "Initialising " << RT_LIB_TO_STR.at(library) << "...\n";
      backends.push_back(make_backend_case(filename, volume, library));
    }

    Position origin = backends.front()->xdg->mesh_manager()->volume_bounding_box(volume).center();
    if (const auto origin_arg = args.present<std::vector<double>>("--origin"))
    {
      origin = Position(origin_arg.value());
    }

    const Direction origin_check_direction {0.371390676, 0.557086014, 0.742781353};
    for (const auto& backend : backends)
    {
      if (!backend->xdg->point_in_volume(volume, origin, &origin_check_direction))
      {
        throw std::runtime_error("Origin is not inside volume " + std::to_string(volume) +
                                 " according to " + backend->name +
                                 "; provide a valid point with --origin");
      }
    }

    std::ofstream csv(output_filename, std::ios::out | std::ios::trunc);
    if (!csv)
    {
      throw std::runtime_error("Could not open output file: " + output_filename);
    }
    write_csv_header(csv);

    const std::string model = std::filesystem::path(filename).filename().string();
    const std::uint64_t total_queries = num_rays * repetitions;

    std::cout << "\nXDG streaming ray-tracer cross-check\n"
              << "------------------------------------\n"
              << "Model                 : " << model << '\n'
              << "Volume                : " << volume << '\n'
              << "Origin                : " << origin.x << ", " << origin.y << ", " << origin.z
              << '\n'
              << "Backends              : ";
    for (std::size_t i = 0; i < backends.size(); ++i)
    {
      if (i != 0)
        std::cout << ", ";
      std::cout << backends[i]->name;
    }
    std::cout << '\n'
              << "Rays/repetition       : " << num_rays << '\n'
              << "Repetitions           : " << repetitions << '\n'
              << "Total rays/backend    : " << total_queries << '\n'
              << "Comparisons/pair      : " << total_queries << '\n'
              << "Batch size            : " << batch_size << '\n'
              << "Absolute tolerance    : " << absolute_tolerance << '\n'
              << "Relative tolerance    : " << relative_tolerance << '\n'
              << "Output                : " << output_filename << '\n'
              << "------------------------------------\n";

    auto aggregate_stats = make_pair_stats(backends);
    std::vector<XDGRayHit> input(batch_size);
    std::vector<std::vector<XDGRayHit>> results(backends.size());

    const auto full_start = std::chrono::steady_clock::now();
    for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition)
    {
      const std::uint64_t run_seed =
          repeat_same_rays
              ? base_seed
              : base_seed + static_cast<std::uint64_t>(repetition) * 0x9e3779b97f4a7c15ULL;
      auto run_stats = make_pair_stats(backends);
      std::uint64_t processed = 0;
      std::uint32_t last_percent = std::numeric_limits<std::uint32_t>::max();
      const auto repetition_start = std::chrono::steady_clock::now();

      while (processed < num_rays)
      {
        const std::size_t active_count =
            static_cast<std::size_t>(std::min<std::uint64_t>(batch_size, num_rays - processed));
        input.resize(active_count);
        generate_rays(input, processed, run_seed, volume, origin);

        for (std::size_t backend_index = 0; backend_index < backends.size(); ++backend_index)
        {
          run_backend_batch(*backends[backend_index], input, results[backend_index]);
        }

        std::size_t pair_index = 0;
        for (std::size_t a = 0; a < backends.size(); ++a)
        {
          for (std::size_t b = a + 1; b < backends.size(); ++b)
          {
            compare_batch(results[a], results[b], absolute_tolerance, relative_tolerance,
                          run_stats[pair_index]);
            ++pair_index;
          }
        }

        processed += active_count;
        const std::uint32_t percent = static_cast<std::uint32_t>(
            (static_cast<long double>(processed) * 100.0L) / static_cast<long double>(num_rays));
        if (percent != last_percent)
        {
          std::cerr << '\r' << "Repetition " << (repetition + 1) << '/' << repetitions << ": "
                    << std::setw(3) << percent << '%' << std::flush;
          last_percent = percent;
        }
      }
      std::cerr << '\n';

      const double repetition_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - repetition_start)
              .count();
      std::cout << "\nRepetition " << (repetition + 1) << " completed in " << repetition_seconds
                << " s\n";

      for (std::size_t pair = 0; pair < run_stats.size(); ++pair)
      {
        print_stats(run_stats[pair]);
        merge_stats(aggregate_stats[pair], run_stats[pair]);
        write_csv_row(csv, "repetition", model, volume, origin, run_stats[pair],
                      std::to_string(repetition + 1), std::to_string(run_seed), absolute_tolerance,
                      relative_tolerance, batch_size);
      }
      csv.flush();
    }

    for (const auto& stats : aggregate_stats)
    {
      write_csv_row(csv, "aggregate", model, volume, origin, stats, "", "", absolute_tolerance,
                    relative_tolerance, batch_size);
    }
    csv.flush();

    const double full_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - full_start).count();
    std::cout << "\nAggregate results\n"
              << "-----------------\n";
    for (const auto& stats : aggregate_stats)
      print_stats(stats);
    std::cout << "\nCross-check time      : " << full_seconds << " s\n"
              << "Results written to    : " << output_filename << '\n';
  }
  catch (const std::exception& error)
  {
    std::cerr << "ray-cross-check: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
