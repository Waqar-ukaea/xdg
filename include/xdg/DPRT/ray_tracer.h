#ifndef XDG_DPRT_RAY_TRACER_H
#define XDG_DPRT_RAY_TRACER_H

#include <unordered_map>


#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/ray_tracing_interface.h"
#include "xdg/error.h"
#include <dprt/dprt.h>

namespace xdg {

class DPRTRayTracer : public RayTracer {
public:
  DPRTRayTracer();
  ~DPRTRayTracer();

  RTLibrary library() const override { return RTLibrary::DPRT; }

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

  void batch_ray_fire(TreeID tree, DPRTRay* d_rays, DPRTHit* d_hits, size_t num_rays) override;

  MeshID find_element(const Position& point) const override;

  MeshID find_element(TreeID tree, const Position& point) const override;

  std::pair<double, MeshID> closest(TreeID tree,
                                    const Position& origin) override;

  bool occluded(TreeID tree,
                const Position& origin,
                const Direction& direction,
                double& dist) const override;

private:
  DPRTContext context_ {nullptr};
  std::unordered_map<TreeID, DPRTModel> surface_tree_to_model_;
};
} // namespace xdg

#endif // XDG_DPRT_RAY_TRACER_H
