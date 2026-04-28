#include "xdg/cuBQL/ray_tracer.h"
#include "xdg/error.h"

// Guards to prevent CUDA headers from being included in host code, which causes failed compilation with LLVM-clang
#if defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#undef __CUDA_ARCH__
#endif

#include "cuBQL/bvh.h"
#include "cuBQL/builder/omp.h"

namespace xdg {

CuBQLRayTracer::CuBQLRayTracer() = default;

CuBQLRayTracer::~CuBQLRayTracer() = default;

void CuBQLRayTracer::init()
{
}

std::pair<TreeID, TreeID>
CuBQLRayTracer::register_volume(const std::shared_ptr<MeshManager>& mesh_manager,
                                MeshID volume)
{
  TreeID surface_tree = create_surface_tree(mesh_manager, volume);
  TreeID element_tree = create_element_tree(mesh_manager, volume);
  return {surface_tree, element_tree};
}

TreeID
CuBQLRayTracer::create_surface_tree(const std::shared_ptr<MeshManager>&,
                                    MeshID)
{
  // Testing cubql compilation
  cuBQL::bvh_t<double, 3> bvh;


  return TREE_NONE;
}

TreeID
CuBQLRayTracer::create_element_tree(const std::shared_ptr<MeshManager>&,
                                    MeshID)
{
  fatal_error("Element trees not currently supported with cuBQL ray tracer");
  return TREE_NONE;
}

void CuBQLRayTracer::create_global_surface_tree()
{
  fatal_error("Global surface trees not currently supported with cuBQL ray tracer");
}

void CuBQLRayTracer::create_global_element_tree()
{
  fatal_error("Global element trees not currently supported with cuBQL ray tracer");
}

MeshID CuBQLRayTracer::find_element(const Position&) const
{
  fatal_error("Element queries not currently supported with cuBQL ray tracer");
  return ID_NONE;
}

MeshID CuBQLRayTracer::find_element(TreeID, const Position&) const
{
  fatal_error("Element queries not currently supported with cuBQL ray tracer");
  return ID_NONE;
}

bool CuBQLRayTracer::point_in_volume(TreeID,
                                     const Position&,
                                     const Direction*,
                                     const std::vector<MeshID>*) const
{
  fatal_error("Point-in-volume queries not currently supported with cuBQL ray tracer");
  return false;
}

std::pair<double, MeshID>
CuBQLRayTracer::ray_fire(TreeID,
                         const Position&,
                         const Direction&,
                         const double,
                         HitOrientation,
                         std::vector<MeshID>* const)
{
  fatal_error("Ray-fire queries not currently supported with cuBQL ray tracer");
  return {INFTY, ID_NONE};
}

std::pair<double, MeshID>
CuBQLRayTracer::closest(TreeID, const Position&)
{
  fatal_error("Closest queries not currently supported with cuBQL ray tracer");
  return {INFTY, ID_NONE};
}

bool CuBQLRayTracer::occluded(TreeID,
                              const Position&,
                              const Direction&,
                              double&) const
{
  fatal_error("Occlusion queries not currently supported with cuBQL ray tracer");
  return false;
}

} // namespace xdg
