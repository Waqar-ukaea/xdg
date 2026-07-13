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

struct Particle {

Particle(std::shared_ptr<XDG> xdg, uint32_t id, uint32_t max_events, bool verbose=true, bool ipc_graveyard=false)
: verbose_(verbose), xdg_(xdg), id_(id), max_events_(max_events), ipc_graveyard_(ipc_graveyard) {}

template<typename... Params>
void log (const std::string& msg, const Params&... fmt_args) {
  if (!verbose_) return;
  write_message(msg, fmt_args...);
}

void initialize() {
  // TODO: replace with sampling
  r_ = {0.0, 0.0, 0.0};
  u_ = {1.0, 0.0, 0.0};

  volume_ = xdg_->find_volume(r_, u_);
  log("Particle {} initialized in volume {}", id_, volume_);
}

void surf_dist() {
  surface_intersection_ = xdg_->ray_fire(volume_, r_, u_, INFTY, HitOrientation::EXITING, &history_);
  if (surface_intersection_.first == 0.0) {
    fatal_error("Particle {} stuck at position ({}, {}, {}) on surfacce {}", id_, r_.x, r_.y, r_.z, surface_intersection_.second);
    alive_ = false;
    return;
  }
  if (surface_intersection_.second == ID_NONE) {
    fatal_error("Particle {} lost in volume {}", id_, volume_);
    alive_ = false;
    return;
  }
  log("Intersected surface {} at distance {} ", surface_intersection_.second, surface_intersection_.first);
}

void sample_collision_distance(double mfp) {
  collision_distance_ = -std::log(1.0 - drand48()) * mfp;
}

void collide() {
  n_events_++;
  log("Event {} for particle {}", n_events_, id_);
  u_ = rand_dir();
  log("Particle {} collides with material at position ({}, {}, {}), new direction is ({}, {}, {})", id_, r_.x, r_.y, r_.z, u_.z, u_.y, u_.z);
  history_.clear();
}

void advance(std::unordered_map<MeshID, double>& cell_tracks)
{
  log("Comparing surface intersection distance {} to collision distance {}", surface_intersection_.first, collision_distance_);
  if (collision_distance_ < surface_intersection_.first) {
    r_ += collision_distance_ * u_;
    cell_tracks[volume_] += collision_distance_;
    log("Particle {} collides with material at position ({}, {}, {}) ", id_, r_.x, r_.y, r_.z);
  } else {
    r_ += surface_intersection_.first * u_;
    cell_tracks[volume_] += surface_intersection_.first;
    log("Particle {} advances to surface {} at position ({}, {}, {}) ", id_, surface_intersection_.second, r_.x, r_.y, r_.z);
  }
}

void cross_surface()
{
  n_events_++;
  log("Event {} for particle {}", n_events_, id_);
  auto boundary_condition = xdg_->mesh_manager()->get_surface_property(surface_intersection_.second, PropertyType::BOUNDARY_CONDITION);
  // check for the surface boundary condition
  if (boundary_condition.value == "reflecting" || boundary_condition.value == "reflective") {
    log("Particle {} reflects off surface {}", id_, surface_intersection_.second);
    log("Direction before reflection: ({}, {}, {})", u_.x, u_.y, u_.z);

    Direction normal = xdg_->surface_normal(surface_intersection_.second, r_, &history_);
    log("Normal to surface: ({}, {}, {})", normal.x, normal.y, normal.z);

    double proj = dot(normal, u_);
    double mag = normal.length();
    normal = normal * (2.0 * proj/mag);
    u_ = u_ - normal;
    u_ = u_.normalize();
    log("Direction after reflection: ({}, {}, {})", u_.x, u_.y, u_.z);
    // reset to last intersection
    if (history_.size() > 0) {
      log("Resetting particle history to last intersection");
      history_ = {history_.back()};
    }
  } else if (boundary_condition.value == "vacuum") {
    log("Particle {} encounters vacuum boundary at surface {}", id_, surface_intersection_.second);
    alive_ = false;
  } else {
    volume_ = xdg_->mesh_manager()->next_volume(volume_, surface_intersection_.second);
    log("Particle {} enters volume {}", id_, volume_);
    if (ipc_graveyard_ && volume_ == xdg_->mesh_manager()->implicit_complement()) volume_ = ID_NONE;
    if (volume_ == ID_NONE) {
      alive_ = false;
      return;
    }
  }
}

// Data Members
bool verbose_ {true};
std::shared_ptr<XDG> xdg_;
uint32_t id_ {0};
int32_t max_events_ {1000};
bool ipc_graveyard_ {false};

Position r_;
Direction u_;
MeshID volume_ {ID_NONE};
std::vector<MeshID> history_ {};
std::pair<double, MeshID> surface_intersection_ {INFTY, ID_NONE};
double collision_distance_ {INFTY};
int32_t n_events_ {0};
bool alive_ {true};
};

struct EventSimulationData {
  std::shared_ptr<XDG> xdg_;
  double mfp_ {1.0};
  uint32_t n_particles_ {100};
  uint32_t max_events_ {1000};
  bool verbose_particles_ {false};
  bool implicit_complement_is_graveyard_ {false};
  std::unordered_map<MeshID, double> cell_tracks;

  uint32_t max_particles_in_flight_ {100000};
  std::vector<Particle> particles;
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

  sim_data.particles.clear();
  sim_data.particles.reserve(sim_data.n_particles_);

  const int gpu_id = omp_get_default_device();
  sim_data.advance_particle_queue.allocate(sim_data.n_particles_, gpu_id);
  sim_data.surface_crossing_queue.allocate(sim_data.n_particles_, gpu_id);
  sim_data.collision_queue.allocate(sim_data.n_particles_, gpu_id);

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
}


/*
  initialize position/direction
  find starting volume (call this globally for all particles [in flight])
  set global particle id
  reset event counter
  reset alive flag
  reset ray history
  reset pending surface hit / collision distance
  initialize per-particle RNG seed if you stop using global drand48()
  enqueue into advance queue

  Essentially set up particle state (for all particles) ready to call advance_particle
*/

// void process_death_events();
/*
  Particle death in the pseudo transport app doesn't really mean all that much. 
  We are essentially just setting the particle.alive_ member to false. 
  I think a better approach is to actually just handle death as it happens. 
  Rather than waiting for everything a particle's death state should update when it 
  occurs. So we don't both with a process_death_events() method.
*/
