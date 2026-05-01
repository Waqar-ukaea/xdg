#ifndef _XDG_CUBQL_RAY_TRACING_INTERFACE_H
#define _XDG_CUBQL_RAY_TRACING_INTERFACE_H

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xdg/constants.h"
#include "xdg/geometry_data.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/ray.h"
#include "xdg/ray_tracing_interface.h"

// Guards to prevent CUDA headers from being included in host code, which causes failed compilation with LLVM-clang
#if defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#undef __CUDA_ARCH__
#endif

#include "cuBQL/bvh.h"



namespace xdg {

class CuBQLRayTracer : public RayTracer {
public:
  CuBQLRayTracer();
  ~CuBQLRayTracer() override;

  RTLibrary library() const override { return RTLibrary::CUBQL; }

  void init() override;

  std::pair<TreeID, TreeID>
  register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                  MeshID volume) override;

  TreeID create_surface_tree(const std::shared_ptr<MeshManager>& mesh_manager,
                             MeshID volume) override;

  TreeID create_element_tree(const std::shared_ptr<MeshManager>& mesh_manager,
                             MeshID volume) override;

  void create_global_surface_tree() override;

  void create_global_element_tree() override;

  MeshID find_element(const Position& point) const override;

  MeshID find_element(TreeID tree, const Position& point) const override;

  bool point_in_volume(TreeID tree,
                       const Position& point,
                       const Direction* direction = nullptr,
                       const std::vector<MeshID>* exclude_primitives = nullptr) const override;

  std::pair<double, MeshID> ray_fire(TreeID tree,
                                     const Position& origin,
                                     const Direction& direction,
                                     const double dist_limit = INFTY,
                                     HitOrientation orientation = HitOrientation::EXITING,
                                     std::vector<MeshID>* const exclude_primitives = nullptr) override;

  std::pair<double, MeshID> closest(TreeID tree,
                                    const Position& origin) override;

  bool occluded(TreeID tree,
                const Position& origin,
                const Direction& direction,
                double& dist) const override;

private:
  struct CuBQLSurfaceBVH {
    MeshID surface {ID_NONE};
    cuBQL::bvh3d bvh;

    cuBQL::vec3d* d_vertices {nullptr};
    cuBQL::vec3i* d_indices {nullptr};
    cuBQL::vec3d* d_normals {nullptr};
    MeshID* d_primitive_refs {nullptr};

    uint32_t num_vertices {0};
    uint32_t num_faces {0};
    int gpu_id {0};
  };

  struct CuBQLRayHit {
    double distance {INFTY};
    MeshID surface {ID_NONE};
    MeshID primitive {ID_NONE};
  };

  std::vector<CuBQLSurfaceBVH> surface_bvhs_;
  std::unordered_map<TreeID, std::vector<size_t>> tree_to_surface_bvh_indices_;
};

} // namespace xdg

#endif // include guard
