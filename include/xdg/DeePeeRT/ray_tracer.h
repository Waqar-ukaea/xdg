#ifndef XDG_DEEPEE_RT_RAY_TRACER_H
#define XDG_DEEPEE_RT_RAY_TRACER_H

#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/ray_tracing_interface.h"
#include "xdg/error.h"


class DeePeeRTRayTracer : public RayTracerInterface {
public:
  DeePeeRTRayTracer();
  ~DeePeeRTRayTracer();

  RTLibrary library() const override { return RTLibrary::DEEPEE_RT; }

  void set_geom_data(const std::shared_ptr<MeshManager> mesh_manager) override;

  void init() override;

  std::pair<TreeID, TreeID>
  register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                  MeshID volume) override;
  
  TreeID create_surface_tree(const std::shared_ptr<MeshManager>& mesh_manager,
                             MeshID volume) override;
    
  TreeID create_element_tree(const std::shared_ptr<MeshManager>& mesh_manager,
                             MeshID volume) override;

    void create_global_surface_tree() override;

    void create_global_element_tree() override
    {
      warning("Global element trees not currently supported with DeePeeRT ray tracer");
      return;
    };

    void populate_rays_external(size_t numRays,
                                const RayPopulationCallback& callback) override;

    void download_hits(const size_t num_rays,
                       std::vector<dblHit>& hits);


  void ray_fire_prepared(const size_t num_rays,
                         const double dist_limit = INFTY,
                         HitOrientation orientation = HitOrientation::EXITING) override;


  
  
#endif // XDG_DEEPEE_RT_RAY_TRACER_H