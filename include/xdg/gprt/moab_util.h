#include <vector>
#include <map>
#include <set>

#include "xdg/mesh_manager_interface.h"

#include "moab/Core.hpp"
#include "moab/Range.hpp"
#include "MBTagConventions.hpp"

#include "gprt.h"
#include "xdg/gprt/sharedCode.h"

using namespace moab;

namespace xdg {

int DEBUG_SURF = -4;

std::set<EntityHandle> visible_surfs;

float3 rnd_color() {
  return normalize(float3( std::rand(), std::rand(), std::rand()));
}

template<class T>
struct MBTriangleSurface {

  typedef T geom_data_type;

  int id;
  int n_tris;
  int frontface_vol;
  int backface_vol;
  std::vector<typename T::vertex_type> vertices;
  std::vector<uint3> connectivity;
  GPRTBufferOf<float3> aabb_buffer;
  GPRTGeomOf<T> triangle_geom_s;
  int2 parent_ids;
  bool aabbs_present {false};

  struct SurfaceData {
    std::vector<double> coords;
    std::vector<uint3> connectivity;
  };

MBTriangleSurface(const std::shared_ptr<MeshManager> mesh_manager, MeshID volume, MeshID surface) {

  // get the triangles for this surface
  auto surf_tris = mesh_manager->get_surface_faces(surface);

  n_tris = surf_tris.size();

  auto conn = mesh_manager->get_surface_connectivity(surface);

  auto coords = mesh_manager->get_surface_vertices(surface);
  vertices.resize(coords.size() / 3);

  for (int i=0; i<vertices.size(); i++)
  {
    vertices[i] = typename T::vertex_type(coords[3*i], coords[3*i+1], coords[3*i+2]); 
  }

  for (int i = 0; i < vertices.size(); i++) {
    vertices[i] = typename T::vertex_type(coords[3*i], coords[3*i+1], coords[3*i+2]);
  }

  connectivity.resize(surf_tris.size());

  for (int i = 0; i < surf_tris.size(); i++) {
    connectivity[i] = uint3(conn[3*i], conn[3*i+1], conn[3*i+2]);
  }

  auto parents = mesh_manager->get_parent_volumes(surface);
  parent_ids[0] = parents.first;
  parent_ids[1] = parents.second;
}


};

using SPTriangleSurface = MBTriangleSurface<SPTriangleData>;
using DPTriangleSurface = MBTriangleSurface<DPTriangleData>;

template<class T>
struct MBVolume {
  // Constructor
  MBVolume(MeshID id) : id_(id) {};

  // Methods
  void populate_surfaces(const std::shared_ptr<MeshManager> mesh_manager) {
    MeshID vol = mesh_manager->volumes()[id_];
    auto surfaces = mesh_manager->get_volume_surfaces(vol);

    for (auto surf : surfaces)
      surfaces_.emplace_back(std::move(T(mesh_manager, surf, vol)));
  }

  void create_geoms(GPRTContext context, GPRTGeomTypeOf<typename T::geom_data_type> g_type) {
    for (auto& surf : surfaces_) {
      vertex_buffers_.push_back(gprtDeviceBufferCreate<typename T::geom_data_type::vertex_type>(context, surf.vertices.size(), surf.vertices.data()));
      connectivity_buffers_.push_back(gprtDeviceBufferCreate<uint3>(context, surf.connectivity.size(), surf.connectivity.data()));
      gprt_geoms_.push_back(gprtGeomCreate<typename T::geom_data_type>(context, g_type));
      typename T::geom_data_type* geom_data = gprtGeomGetParameters(gprt_geoms_.back());
      geom_data->vertex = gprtBufferGetDevicePointer(vertex_buffers_.back());
      geom_data->index = gprtBufferGetDevicePointer(connectivity_buffers_.back());
      
      geom_data->id = surf.id;
      geom_data->vols = surf.parent_ids;
    }
  }

  void setup(GPRTContext context, GPRTModule module) {
    for (int i = 0; i < surfaces_.size(); i++) {
      auto& surf = surfaces_[i];
      gprtTrianglesSetVertices(gprt_geoms_[i], vertex_buffers_[i], surf.vertices.size());
      gprtTrianglesSetIndices(gprt_geoms_[i], connectivity_buffers_[i], surf.connectivity.size());
    }
  }

  // does nothing by default
  void dbl_setup(int2 fbSize, GPRTBufferOf<double> dpray_buff) {}

  void create_accel_structures(GPRTContext context) {
    blas_ = gprtTrianglesAccelCreate(context, gprt_geoms_.size(), gprt_geoms_.data());
    gprtAccelBuild(context, blas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
    
    gprt::Instance instance = gprtAccelGetInstance(blas_);
    GPRTBufferOf<gprt::Instance> instance_buffer = gprtDeviceBufferCreate<gprt::Instance>(context, 1, &instance);
    
    tlas_ = gprtInstanceAccelCreate(context, 1, instance_buffer);
    gprtAccelBuild(context, tlas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
  }

  void cleanup () {
    gprtAccelDestroy(tlas_);
    gprtAccelDestroy(blas_);
    for (auto& vert_buff : vertex_buffers_) gprtBufferDestroy(vert_buff);
    for (auto& conn_buff : connectivity_buffers_) gprtBufferDestroy(conn_buff);
    for (auto& geom : gprt_geoms_) gprtGeomDestroy(geom);
    // for (auto& surf : surfaces_) {
    //   if (surf.aabb_buffer) gprtBufferDestroy(surf.aabb_buffer);
    // }
  }

  // Data members
  int id_;
  std::vector<T> surfaces_;
  std::vector<GPRTBufferOf<typename T::geom_data_type::vertex_type>> vertex_buffers_;
  std::vector<GPRTBufferOf<uint3>> connectivity_buffers_;
  std::vector<GPRTGeomOf<typename T::geom_data_type>> gprt_geoms_;
  GPRTAccel blas_;
  GPRTAccel tlas_;
};

template<>
void MBVolume<DPTriangleSurface>::setup(GPRTContext context, GPRTModule module)
{
  // populate AABB buffer
  for (int i = 0; i < surfaces_.size(); i++) {
    auto& surf = surfaces_[i];
    auto& geom = gprt_geoms_[i];

    surf.aabb_buffer = gprtDeviceBufferCreate<float3>(context, 2*surf.n_tris, nullptr);
    gprtAABBsSetPositions(geom, surf.aabb_buffer, surf.n_tris, 2*sizeof(float3), 0);
    GPRTComputeOf<DPTriangleData> boundsProg = gprtComputeCreate<DPTriangleData>(context, module, "DPTriangle");
    auto boundsProgData = gprtComputeGetParameters(boundsProg);
    boundsProgData->vertex = gprtBufferGetHandle(vertex_buffers_[i]);
    boundsProgData->index = gprtBufferGetHandle(connectivity_buffers_[i]);
    boundsProgData->aabbs = gprtBufferGetHandle(surf.aabb_buffer);
    gprtBuildShaderBindingTable(context, GPRT_SBT_COMPUTE);
    gprtComputeLaunch1D(context, boundsProg, surf.n_tris);
    surf.aabbs_present = true;
  }
}

template<>
void MBVolume<DPTriangleSurface>::dbl_setup(int2 fbSize, GPRTBufferOf<double> dpray_buff)
{
  for (auto& geom : gprt_geoms_) {
    auto geom_data = gprtGeomGetParameters(geom);
    geom_data->fbSize = fbSize;
    geom_data->dpRays = gprtBufferGetHandle(dpray_buff);
  }
}

template<>
void MBVolume<DPTriangleSurface>::create_accel_structures(GPRTContext context)
{
  blas_ = gprtAABBAccelCreate(context, gprt_geoms_.size(), gprt_geoms_.data());
  gprtAccelBuild(context, blas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
  tlas_ = gprtInstanceAccelCreate(context, 1, &blas_);
  gprtAccelBuild(context, tlas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
}

template<class T>
struct MBVolumes {
  // Constructors
  MBVolumes(std::vector<int> ids) {
    for (auto id : ids) {
      volumes().push_back(MBVolume<T>(id));
    }
  }

  // Methods
  void populate_surfaces(const std::shared_ptr<MeshManager> mesh_manager) {
    for (auto& volume : volumes()) {
      volume.populate_surfaces(mesh_manager);
    }
  }

  void create_geoms(GPRTContext context, GPRTGeomTypeOf<typename T::geom_data_type> g_type) {
    for (auto& volume : volumes()) {
      volume.create_geoms(context, g_type);
    }
  }

  void setup(GPRTContext context, GPRTModule module) {
    for (auto& volume : volumes()) {
      volume.setup(context, module);
    }
  }

  void dbl_setup(int2 fbSize, GPRTBufferOf<double> dpray_buff) {
    for (auto& volume : volumes()) {
      volume.dbl_setup(fbSize, dpray_buff);
    }
  }

  void create_accel_structures(GPRTContext context) {
    // gather up all BLAS and join into a single TLAS
    std::vector<GPRTAccel> blass;
    for (auto& vol : volumes()) {
      vol.create_accel_structures(context);
      blass.push_back(vol.blas_);
    }
    world_tlas_ = gprtInstanceAccelCreate(context, blass.size(), blass.data());
    gprtAccelBuild(context, world_tlas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);

    std::vector<GPRTAccel> accel_ptrs;
    for (auto& vol : volumes()) accel_ptrs.push_back(gprtAccelGetHandle(vol.tlas_));
    // map acceleration pointers to a device buffer
    tlas_buffer_ = gprtDeviceBufferCreate<GPRTAccel>(context, accel_ptrs.size(), accel_ptrs.data());

    // create a map of volume ID to index
    std::map<int, int> vol_id_to_idx_map;
    for (int i = 0; i < volumes().size(); i++) {
      vol_id_to_idx_map[volumes()[i].id_] = i;
    }

    // set surface parent indices into the tlas buffer for each surface
    for (auto& vol : volumes()) {
      for (auto& geom : vol.gprt_geoms_) {
        auto geom_data = gprtGeomGetParameters(geom);
        geom_data->ff_vol = vol_id_to_idx_map[geom_data->vols[0]];
        geom_data->bf_vol = vol_id_to_idx_map[geom_data->vols[1]];
      }
    }
  }

  void cleanup() {
    gprtAccelDestroy(world_tlas_);
    gprtBufferDestroy(tlas_buffer_);
    for (auto& vol : volumes()) {
      vol.cleanup();
    }
  }

  // Accessors
  const auto& volumes() const { return volumes_; }
  auto& volumes() { return volumes_; }

  // Data members
  std::vector<MBVolume<T>> volumes_;
  GPRTAccel world_tlas_;
  GPRTBufferOf<GPRTAccel> tlas_buffer_;
};

template<class T>
struct MBTriangleSurfaces {

  // Data members
  std::vector<T> surfaces_;
  std::vector<GPRTAccel> blass_;
};



// Create an object that is a collection of SPTriangle surface objects and can
//  - call any necessary methods for final setup (set_buffers, aabbs, etc.)
//  - create it's own BLAS for all surfaces in the container

// TLAS creation and index mapping into the TLAS should be able to occur on each of these containers
template<class T, class G>
std::map<int, std::vector<T>> setup_surfaces(GPRTContext context, GPRTModule module, const std::shared_ptr<MeshManager> mesh_manager, GPRTGeomTypeOf<G> g_type, std::vector<int> visible_vol_ids = {}) {

    int n_surfs = mesh_manager->num_surfaces();

    if (n_surfs == 0) {
      std::cerr << "No surfaces were found in the model" << std::endl;
      std::exit(1);
    }

    if (visible_vol_ids.size() == 0) {
      for (int i = 0; i < mesh_manager->num_volumes(); i++) visible_vol_ids.push_back(mesh_manager->volumes(i+1));
      // add graveyard explicitly
      // EntityHandle gyg;
      // Range gys;
      // mesh_manager->
      // dag->get_graveyard_group(gyg);
      // dag->moab_instance()->get_entities_by_type(gyg, MBENTITYSET, gys);
    }

    std::map<int, std::vector<T>> out;
    for (auto vol_id : visible_vol_ids) {
      EntityHandle vol = dag->entity_by_id(3, vol_id);

      if (std::find(visible_vol_ids.begin(), visible_vol_ids.end(), vol_id) == visible_vol_ids.end()) continue;

      Range vol_surfs;
      rval = dag->moab_instance()->get_child_meshsets(vol, vol_surfs);

      std::vector<T> surf_geoms;
      for (auto surf : vol_surfs) {
        int surf_id = dag->id_by_index(2, dag->index_by_handle(surf));
        surf_geoms.emplace_back(std::move(T(context, dag->moab_instance(), g_type, surf_id, vol_id)));
      }
      out[vol_id] = surf_geoms;
    }
    return out;
}

std::pair<double3, double3> bounding_box(Interface* mbi) {
  ErrorCode rval;

  Range all_verts;
  rval = mbi->get_entities_by_dimension(0, 0, all_verts, true);
  MB_CHK_SET_ERR_CONT(rval, "Failed to retrieve all vertices");

  std::vector<double> coords(3*all_verts.size());
  rval = mbi->get_coords(all_verts, coords.data());
  MB_CHK_SET_ERR_CONT(rval, "Failed to get vertex coordinates");

  double3 aabbmin = double3(coords[0], coords[1], coords[2]);
  double3 aabbmax = aabbmin;
  for (uint32_t i = 1; i < all_verts.size(); ++i) {
    aabbmin = linalg::min(aabbmin, double3(coords[i * 3 + 0],
                                           coords[i * 3 + 1],
                                           coords[i * 3 + 2]));
    aabbmax = linalg::max(aabbmax, double3(coords[i * 3 + 0],
                                           coords[i * 3 + 1],
                                           coords[i * 3 + 2]));
  }

  return {aabbmin, aabbmax};
}

} // namespace xdg