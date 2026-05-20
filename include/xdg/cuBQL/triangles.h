#ifndef _XDG_CUBQL_TRIANGLES_H
#define _XDG_CUBQL_TRIANGLES_H

#include <cstdint>
#include <vector>

// Guards to prevent CUDA headers from being included in host code, which causes
// failed compilation with LLVM-clang.
#if defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#undef __CUDA_ARCH__
#endif

#include "cuBQL/bvh.h"
#include "cuBQL/math/vec.h"
#include "xdg/constants.h"
#include "xdg/cuBQL/cuBQL_backend.h"

namespace xdg {

/*
  Owns the triangle buffers for one topological surface. The nested DD type is
  the compact device-data view copied into OpenMP target regions instead of the
  full host-side owner.
*/
struct CuBQLSurfaceMesh {
  struct DD {
    // Topological metadata
    MeshID surface_id {ID_NONE};

    // Geometric data
    const cuBQL::vec3d* vertices {nullptr};
    const cuBQL::vec3i* indices {nullptr};
    const MeshID* primitive_refs {nullptr};
  };

  // Topological metadata
  MeshID surface_id {ID_NONE};

  // Device buffers for triangle data
  cuBQL::vec3d* d_vertices {nullptr};
  cuBQL::vec3i* d_indices {nullptr};
  MeshID* d_primitive_refs {nullptr};

  uint32_t num_vertices {0};
  uint32_t num_triangles {0};
  int gpu_id {0};

  // Accessor for Device Data struct, which is passed to cuBQL BVH traversal/intersection functions
  DD get_device_data() const
  {
    return {
      surface_id,
      d_vertices,
      d_indices,
      d_primitive_refs
    };
  }

  void release();
};

/*
  Owns a cuBQL bottom-level acceleration structure over surface triangles.
  The nested DD type is the compact device-data view used during traversal.

  TODO - DPRT also supports grouping multiple triangle meshes into one BLAS.
  xdg currently uses a simpler one topological surface mesh per BLAS layout
  but maybe it would be useful for us to also support multiple surfaces being
  tied to the same BLAS?
*/
struct CuBQLSurfaceBLAS {
  struct DD {
    CuBQLSurfaceMesh::DD mesh; // Mesh data device handle
    cuBQL::bvh3d bvh; // B:AS device handle
  };

  cuBQL::bvh3d bvh; // BLAS host handle
  CuBQLSurfaceMesh mesh; // Surface mesh host owner

  uint32_t num_prims {0};
  int gpu_id {0};

  DD get_device_data() const
  {
    return {mesh.get_device_data(), bvh};
  }

  void release();
};

/*
  Owns a cuBQL top-level acceleration structure for one topological volume.
  The TLAS groups the surface BLASes that bound that volume and stores
  per-volume relationship metadata for each surface instance.
*/
struct CuBQLVolumeTLAS {
  /*
    TLAS-local instance payload. The same surface BLAS can participate in
    different volume TLASes with different sense, so reverse_sense belongs on
    the volume-surface relationship rather than on the reusable surface mesh
    or BLAS geometry.
  */
  struct SurfaceInstanceDD {
    CuBQLSurfaceBLAS::DD surface_blas;
    bool reverse_sense {false};
  };

  struct DD {
    const SurfaceInstanceDD* surface_instances {nullptr};
    cuBQL::bvh3d bvh; // TLAS device handle 
  };

  cuBQL::bvh3d bvh; // TLAS host handle
  SurfaceInstanceDD* d_surface_instances {nullptr};

  uint32_t num_surface_instances {0};
  int gpu_id {0};

  DD get_device_data() const
  {
    return {d_surface_instances, bvh};
  }

  void release();
};

} // namespace xdg

#endif // include guard
