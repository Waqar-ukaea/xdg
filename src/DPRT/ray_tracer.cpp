#include "xdg/DPRT/ray_tracer.h"

namespace xdg {

DPRTRayTracer::DPRTRayTracer() = default;
DPRTRayTracer::~DPRTRayTracer() = default;

void DPRTRayTracer::init()
{
  warning("DPRT ray tracer init() is currently a stub.");
}

std::pair<TreeID, TreeID>
DPRTRayTracer::register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                               MeshID volume)
{
  TreeID surface_tree = create_surface_tree(mesh_manager, volume);
  TreeID element_tree = create_element_tree(mesh_manager, volume);
  return {surface_tree, element_tree};
}

TreeID DPRTRayTracer::create_surface_tree(const std::shared_ptr<MeshManager>&, MeshID)
{
  warning("DPRT surface tree creation is not implemented.");
  return TREE_NONE;
}

TreeID DPRTRayTracer::create_element_tree(const std::shared_ptr<MeshManager>&, MeshID)
{
  warning("DPRT element tree creation is not implemented.");
  return TREE_NONE;
}

void DPRTRayTracer::create_global_surface_tree()
{
  warning("DPRT global surface tree creation is not implemented.");
}

void DPRTRayTracer::create_global_element_tree()
{
  warning("DPRT global element tree creation is not implemented.");
}

bool DPRTRayTracer::point_in_volume(TreeID,
                                    const Position&,
                                    const Direction*,
                                    const std::vector<MeshID>*) const
{
  fatal_error("DPRT point_in_volume() is not implemented.");
}

std::pair<double, MeshID> DPRTRayTracer::ray_fire(TreeID,
                                                  const Position&,
                                                  const Direction&,
                                                  const double,
                                                  HitOrientation,
                                                  std::vector<MeshID>* const)
{
  fatal_error("DPRT ray_fire() is not implemented.");
}

MeshID DPRTRayTracer::find_element(const Position&) const
{
  fatal_error("DPRT find_element() is not implemented.");
}

MeshID DPRTRayTracer::find_element(TreeID, const Position&) const
{
  fatal_error("DPRT find_element(tree, point) is not implemented.");
}

std::pair<double, MeshID> DPRTRayTracer::closest(TreeID, const Position&)
{
  fatal_error("DPRT closest() is not implemented.");
}

bool DPRTRayTracer::occluded(TreeID, const Position&, const Direction&, double&) const
{
  fatal_error("DPRT occluded() is not implemented.");
}

void DPRTRayTracer::ray_fire_prepared(const size_t, const double, HitOrientation)
{
  fatal_error("DPRT ray_fire_prepared() is not implemented.");
}

void DPRTRayTracer::populate_rays_external(size_t, const RayPopulationCallback&)
{
  fatal_error("DPRT populate_rays_external() is not implemented.");
}

} // namespace xdg

