#ifndef _XDG_CUBQL_TRIANGLES_H
#define _XDG_CUBQL_TRIANGLES_H

#include <cstdint>

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

// Triangle mesh data structure for storing triangle vertex/index data.
// Including a device-data struct (DD) 
struct CuBQLSurfaceMesh {
  struct DD {
    // Topological metadata
    MeshID surface_id {ID_NONE};
    MeshID forward_parent {ID_NONE};
    MeshID reverse_parent {ID_NONE};

    // Geometric data
    const cuBQL::vec3d* vertices {nullptr};
    const cuBQL::vec3i* indices {nullptr};
    const MeshID* primitive_refs {nullptr};
  };

  DD dd;

  // Device buffers for triangle data
  cuBQL::vec3d* d_vertices {nullptr};
  cuBQL::vec3i* d_indices {nullptr};
  MeshID* d_primitive_refs {nullptr};

  // Metadata of the surface mesh TODO - do we really need these stored here?
  uint32_t num_vertices {0};
  uint32_t num_triangles {0};
  int gpu_id {0};

  // Accessor for Device Data struct, which is passed to cuBQL BVH traversal/intersection functions
  DD get_device_data() const
  {
    DD device_data;
    device_data.surface_id = dd.surface_id;
    device_data.forward_parent = dd.forward_parent;
    device_data.reverse_parent = dd.reverse_parent;
    device_data.vertices = d_vertices;
    device_data.indices = d_indices;
    device_data.primitive_refs = d_primitive_refs;
    return device_data;
  }
};

struct CuBQLSurfaceBLAS {
  struct PrimRef {
    int meshID {0};
    int primID {0};
  };

  struct DD {
    CuBQLSurfaceMesh::DD* meshes {nullptr};
    PrimRef* primRefs {nullptr};
    cuBQL::bvh3d bvh;
  };

  cuBQL::bvh3d bvh;
  CuBQLSurfaceMesh::DD* d_meshDDs {nullptr};
  PrimRef* d_primRefs {nullptr};

  uint32_t num_meshes {0};
  uint32_t num_prims {0};
  int gpu_id {0};

  DD get_device_data() const
  {
    DD device_data;
    device_data.meshes = d_meshDDs;
    device_data.primRefs = d_primRefs;
    device_data.bvh = bvh;
    return device_data;
  }
};

} // namespace xdg

#endif // include guard
