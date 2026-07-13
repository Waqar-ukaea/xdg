#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "xdg/error.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "random_lcg.h"

using namespace xdg;

// Lightweight append queue used by event queues.
// Making use of the same internal device data view pattern I borrowed from cuBQL/DPRT
template<typename T>
struct DeviceAppendQueue {
  struct DD {
    T* data {nullptr};
    int* size {nullptr}; // Number of current valid entries
    int capacity {0}; // Current allocated limit on number of entries before container overflow

    int thread_safe_append(const T& value)
    {
      int idx; 
      #pragma omp atomic capture 
      idx = (*size)++;

      if (idx >= capacity) {
        #pragma omp atomic write
        *size = capacity;
        return -1;
      }

      data[idx] = value;
      return idx;
    };
  };

  T* d_data {nullptr};
  int* d_size {nullptr};
  int h_size {0};
  int capacity {0};
  int gpu_id {0};
  int host_id {omp_get_initial_device()};

  void allocate(int capacity_, int gpu_id_)
  {
    release();

    capacity = capacity_;
    gpu_id = gpu_id_;
    host_id = omp_get_initial_device();
    h_size = 0;

    if (capacity == 0) return;

    d_data = static_cast<T*>(omp_target_alloc(capacity * sizeof(T), gpu_id));
    d_size = static_cast<int*>(omp_target_alloc(sizeof(int), gpu_id));

    if (!d_data || !d_size) {
      release();
      fatal_error("Failed to allocate event queue on OpenMP target device.");
    }

    resize(0);
  }

  void release()
  {
    if (d_data) {
      omp_target_free(d_data, gpu_id);
      d_data = nullptr;
    }
    if (d_size) {
      omp_target_free(d_size, gpu_id);
      d_size = nullptr;
    }

    h_size = 0;
    capacity = 0;
  }

  void resize(int size)
  {
    h_size = size;
    if (!d_size) return;

    omp_target_memcpy(d_size,
                      &h_size,
                      sizeof(int),
                      0,
                      0,
                      gpu_id,
                      host_id);
  }

  // Small wrapper to call resize with size 0
  void reset()
  {
    resize(0);
  }

  void sync_size_device_to_host()
  {
    if (!d_size) return;

    omp_target_memcpy(&h_size,
                      d_size,
                      sizeof(int),
                      0,
                      0,
                      host_id,
                      gpu_id);
  }

  int size() const { return h_size; }

  DD get_device_data() const
  {
    return {d_data, d_size, capacity};
  }
};
struct EventQueueItem {
  uint32_t idx; // particle index in event-based particle buffer
};

using ParticleEventQueue = DeviceAppendQueue<EventQueueItem>;

#ifdef _OPENMP
#pragma omp declare target
#endif

struct EventParticle {
  void initialize(uint32_t id,
                  std::uint32_t seed,
                  Position r,
                  Direction u,
                  MeshID volume)
  {
    id_ = id;
    r_.x = r.x;
    r_.y = r.y;
    r_.z = r.z;
    u_.x = u.x;
    u_.y = u.y;
    u_.z = u.z;
    volume_ = volume;
    surface_hit_ = ID_NONE;
    surface_hit_distance_ = INFTY;
    collision_distance_ = INFTY;
    last_surface_hit_ = ID_NONE;
    rng_state_ = seed ^ id;
    n_events_ = 0;
    alive_ = true;
    stuck_ = false;
  }

  void sample_collision_distance(double mfp)
  {
    collision_distance_ = -std::log(1.0 - xdg::tools::random::rand01(rng_state_)) * mfp;
  }

  double advance()
  {
    double distance;

    if (collision_distance_ < surface_hit_distance_) {
      distance = collision_distance_;
    } else {
      distance = surface_hit_distance_;
    }

    // explicit scalar to avoid compilation issues with vec3da operator overloads on device
    // TODO - Try via vec3da operator overload
    r_.x += distance * u_.x;
    r_.y += distance * u_.y;
    r_.z += distance * u_.z;

    return distance;
  }

  void collide()
  {
    n_events_++;

    double direction[3];
    xdg::tools::random::random_unit_dir_lcg(rng_state_, direction);
    u_.x = direction[0];
    u_.y = direction[1];
    u_.z = direction[2];

    // reset surface-hit-state explicitly
    surface_hit_ = ID_NONE;
    surface_hit_distance_ = INFTY;
    last_surface_hit_ = ID_NONE;
  }

  void store_surface_hit(MeshID surface, double distance)
  {
    surface_hit_ = surface;
    surface_hit_distance_ = distance;
    last_surface_hit_ = surface;
  }

  uint32_t id_ {0};
  Position r_ {};
  Direction u_ {};
  MeshID volume_ {ID_NONE};

  MeshID surface_hit_ {ID_NONE};
  double surface_hit_distance_ {INFTY};
  double collision_distance_ {INFTY};
  MeshID last_surface_hit_ {ID_NONE};

  std::uint32_t rng_state_ {0};
  int32_t n_events_ {0};
  bool alive_ {true};
  bool stuck_ {false};
};

#ifdef _OPENMP
#pragma omp end declare target
#endif

struct EventSimulationData {
  std::shared_ptr<XDG> xdg_;
  double mfp_ {1.0};
  std::uint32_t seed_ {42};
  uint32_t n_particles_ {100000};
  uint32_t max_events_ {1000};
  bool verbose_particles_ {false};
  bool implicit_complement_is_graveyard_ {false};
  std::unordered_map<MeshID, double> cell_tracks;

  uint32_t max_particles_in_flight_ {100000};
  EventParticle* device_particles {nullptr};
  int gpu_id {0};
  int host_id {omp_get_initial_device()};
  XDGRayHitBuffer ray_hits;
  ParticleEventQueue advance_particle_queue;
  ParticleEventQueue surface_crossing_queue;
  ParticleEventQueue collision_queue;
};

void process_init_events(EventSimulationData& sim_data);
void process_advance_particle_events(EventSimulationData& sim_data);
void process_surface_crossing_events(EventSimulationData& sim_data);
void process_collision_events(EventSimulationData& sim_data);

inline void transport_particle_event_based(EventSimulationData& sim_data) {
  // MPI will be needed for multi GPU 
  // #ifdef OPENMC_MPI
  // MPI_Barrier( mpi::intracomm );
  // #endif

  /*
    A couple of different concepts from the event based algorithm don't need to be considered here:
    - We don't need any fuel/xs lookup related events. 
    - Death events amount to essentially just setting the flag alive_ = false
    - We don't need to worry about secondary particles/a revival bank. This is a consequence of nuclear physics
    - We don't worry about in flight particles and batch all particles in one go
  
  */

  if (sim_data.n_particles_ == 0) {
    fatal_error("Number of event particles must be greater than 0");
  }

  if (sim_data.n_particles_ > sim_data.max_particles_in_flight_) {
    fatal_error("Event particle refill is not implemented; n_particles must be <= max_particles_in_flight");
  }

  sim_data.ray_hits = sim_data.xdg_->allocate_ray_hits(sim_data.n_particles_);
  sim_data.gpu_id = sim_data.ray_hits.device_id;
  sim_data.host_id = omp_get_initial_device();

  sim_data.device_particles = static_cast<EventParticle*>(
    omp_target_alloc(sim_data.n_particles_ * sizeof(EventParticle), sim_data.gpu_id));
  if (!sim_data.device_particles) {
    fatal_error("Failed to allocate event particles on OpenMP target device.");
  }

  sim_data.advance_particle_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);
  sim_data.surface_crossing_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);
  sim_data.collision_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);

  process_init_events(sim_data);

  // Event-based transport loop
  while (true) {
    // Determine which event kernel has the longest queue
    int64_t max = std::max({
      sim_data.advance_particle_queue.size(),
      sim_data.surface_crossing_queue.size(),
      sim_data.collision_queue.size()});


    // Execute event with the longest queue
    if (max == 0) {
      break;
    } else if (max == sim_data.advance_particle_queue.size()) {
      process_advance_particle_events(sim_data);
    } else if (max == sim_data.surface_crossing_queue.size()) {
      process_surface_crossing_events(sim_data);
    } else if (max == sim_data.collision_queue.size()) {
      process_collision_events(sim_data);
    }
  }


  // MPI will be needed for multi gpu
  // #ifdef OPENMC_MPI
  // MPI_Barrier( mpi::intracomm );
  // #endif

  sim_data.advance_particle_queue.release();
  sim_data.surface_crossing_queue.release();
  sim_data.collision_queue.release();

  omp_target_free(sim_data.device_particles, sim_data.gpu_id);
  sim_data.device_particles = nullptr;

  sim_data.xdg_->free_ray_hits(sim_data.ray_hits);
}

inline void process_init_events(EventSimulationData& sim_data)
{
  // Match the history-based miniapp source initialization.
  Position r {0.0, 0.0, 0.0};
  Direction u {1.0, 0.0, 0.0};
  MeshID volume = sim_data.xdg_->find_volume(r, u);

  EventParticle* device_particles = sim_data.device_particles;
  const int n_particles = static_cast<int>(sim_data.n_particles_);
  const std::uint32_t seed = sim_data.seed_;
  const int gpu_id = sim_data.gpu_id;
  auto advance_queue = sim_data.advance_particle_queue.get_device_data();

  if (!device_particles) {
    fatal_error("Error allocating event particle device storage.");
  }

  sim_data.advance_particle_queue.reset();
  sim_data.surface_crossing_queue.reset();
  sim_data.collision_queue.reset();

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    firstprivate(advance_queue, seed, r, u, volume)
  for (int i = 0; i < n_particles; ++i) {
    device_particles[i].initialize(static_cast<uint32_t>(i), seed, r, u, volume);
    advance_queue.thread_safe_append({static_cast<uint32_t>(i)}); // no need for xs lookup so we just append particle to queue
  }

  sim_data.advance_particle_queue.sync_size_device_to_host(); // ensure host side event scheduler knows the correct queue size
}

inline void process_advance_particle_events(EventSimulationData& sim_data) 
{
  const int n_advance = sim_data.advance_particle_queue.size();

  if (n_advance == 0) 
  {
    // TODO - Once things are definitely working we can probably remove this warning/check
    warning("Advance_particle_events launched with a queue size of 0. Early return called...");
    return;
  }

  EventParticle* device_particles = sim_data.device_particles;
  XDGRayHit* ray_hits = sim_data.ray_hits.data;
  auto advance_queue = sim_data.advance_particle_queue.get_device_data();
  auto surface_crossing_queue = sim_data.surface_crossing_queue.get_device_data();
  auto collision_queue = sim_data.collision_queue.get_device_data();
  const double mfp = sim_data.mfp_;
  const int gpu_id = sim_data.gpu_id;

  // rayhit packing kernel
  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles, ray_hits) \
    firstprivate(advance_queue)
  for (int i = 0; i < n_advance; ++i) {
    const uint32_t particle_idx = advance_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];

    ray_hits[i].origin[0] = p.r_.x;
    ray_hits[i].origin[1] = p.r_.y;
    ray_hits[i].origin[2] = p.r_.z;
    ray_hits[i].direction[0] = p.u_.x;
    ray_hits[i].direction[1] = p.u_.y;
    ray_hits[i].direction[2] = p.u_.z;
    ray_hits[i].t_min = 0.0;
    ray_hits[i].t_max = INFTY;
    ray_hits[i].volume = p.volume_;
    ray_hits[i].distance = INFTY;
    ray_hits[i].surface = ID_NONE;
    ray_hits[i].primitive = ID_NONE;
    ray_hits[i].point_in_volume = OUTSIDE;
  }


  /*
    XDG ray_fire_batch is host-orchestrated but operates on a device-resident
    XDGRayHitBuffer. The particle kernels pack rays into that device buffer,
    the host calls xdg->ray_fire_batch(), and a later particle kernel consumes
    the hit results from the same device buffer.

    This differs from OpenMC's GPU path, where Particle::event_advance() calls
    the device-callable distance-to-boundary logic directly inside one OpenMP
    target kernel. Here we instead call a ray packing kernel followed by xdg's 
    ray_fire kernel and then the particle advance kernel but doing so should
    allow us to benefit from GPU accelerated ray tracing against the CAD.
  */

  // Create a view over the actively packed portion of the preallocated ray-hit buffer
  XDGRayHitBuffer active_hits {sim_data.ray_hits.data,
                               static_cast<std::size_t>(n_advance),
                               sim_data.ray_hits.device_id};

  sim_data.xdg_->ray_fire_batch(active_hits); // Perform GPU-accelerated ray tracing via XDG backend

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles, ray_hits) \
    firstprivate(advance_queue, surface_crossing_queue, collision_queue, mfp)
  for (int i = 0; i < n_advance; i++) {
    const uint32_t particle_idx = advance_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];
    const XDGRayHit& hit = ray_hits[i];

    // Set particle state to stuck. Also killed for now but we could tally the number of stuck particles later
    if (hit.distance == 0.0) {
      p.alive_ = false;
      p.stuck_ = true;
      continue;
    }

    // Set particle state to killed
    if (hit.surface == ID_NONE) {
      p.alive_ = false;
      continue;
    }

    p.store_surface_hit(hit.surface, hit.distance);
    p.sample_collision_distance(mfp);
    p.advance();

    if (p.collision_distance_ < p.surface_hit_distance_) {
      collision_queue.thread_safe_append({particle_idx});
    } else {
      surface_crossing_queue.thread_safe_append({particle_idx});
    }
  }

  sim_data.surface_crossing_queue.sync_size_device_to_host();
  sim_data.collision_queue.sync_size_device_to_host();
  sim_data.advance_particle_queue.reset();
}

inline void process_collision_events(EventSimulationData& sim_data) 
{
  EventParticle* device_particles = sim_data.device_particles;
  const int n_collisions = sim_data.collision_queue.size();

  if (n_collisions == 0) {
    return;
  }

  auto advance_queue = sim_data.advance_particle_queue.get_device_data();
  auto collision_queue = sim_data.collision_queue.get_device_data();
  const int gpu_id = sim_data.gpu_id;
  const int max_events = sim_data.max_events_;

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    firstprivate(advance_queue, collision_queue, max_events)
  for (int i = 0; i < n_collisions; i++)
  {
    const uint32_t particle_idx = collision_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];

    p.collide();
    
    if (p.n_events_ >= max_events) {
      p.alive_ = false;
      continue;
    }

    advance_queue.thread_safe_append({particle_idx});
  }

  sim_data.advance_particle_queue.sync_size_device_to_host();
  sim_data.collision_queue.reset();
}

// void process_death_events();
/*
  Particle death in the pseudo transport app doesn't really mean all that much. 
  We are essentially just setting the particle.alive_ member to false. 
  I think a better approach is to actually just handle death as it happens. 
  Rather than waiting for everything a particle's death state should update when it 
  occurs. So we don't bother with a process_death_events() method.
*/
