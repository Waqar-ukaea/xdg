#ifndef _XDG_LIBMESH_MESH_MANAGER
#define _XDG_LIBMESH_MESH_MANAGER

#include <memory>

#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"

#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
namespace xdg {


class LibMeshMeshManager : public MeshManager {

public:
  LibMeshMeshManager(void* ptr);

  LibMeshMeshManager();

  ~LibMeshMeshManager();

  MeshLibrary mesh_library() const override { return MeshLibrary::LIBMESH; }

  void load_file(const std::string& filepath) override;

  void initialize_libmesh();

  void init() override;

  void parse_metadata() override;

  int num_volumes() const override { return volumes_.size(); }

  int num_surfaces() const override { return surfaces_.size(); }

  int num_ents_of_dimension(int dim) const override {
    switch (dim) {
      case 3: return num_volumes();
      case 2: return num_surfaces();
      default: return 0;
    }
  }

  int num_volume_elements(MeshID volume) const override {
    return mesh()->n_elem();
  }

  int num_surface_elements(MeshID surface) const override {
    return mesh()->n_elem();
  }

  void discover_surface_elements();

  std::vector<MeshID> get_volume_elements(MeshID volume) const override;

  std::vector<MeshID> get_surface_elements(MeshID surface) const override;

  std::vector<Vertex> element_vertices(MeshID element) const override;

  std::array<Vertex, 3> triangle_vertices(MeshID triangle) const override;

  std::vector<MeshID> get_volume_surfaces(MeshID volume) const override;

  MeshID create_volume() override;

  void add_surface_to_volume(MeshID volume, MeshID surface, Sense sense, bool overwrite=false) override {
    throw std::runtime_error("Add surface to volume not implemented for libMesh");
  }

  std::pair<MeshID, MeshID> surface_senses(MeshID surface) const override;

  Sense surface_sense(MeshID surface, MeshID volume) const override;

  // Accessors
  const libMesh::Mesh* mesh() const { return mesh_.get(); }

  private:
    std::unique_ptr<libMesh::Mesh> mesh_ {nullptr};
    // TODO: make this global so it isn't owned by a single mesh manager
    std::unique_ptr<libMesh::LibMeshInit> libmesh_init {nullptr};

    // sideset face mapping, stores the element and the side number
    // for each face in the mesh that lies on a boundary
    std::unordered_map<MeshID, std::vector<std::pair<const libMesh::Elem*, MeshID>>> sideset_element_map_;

  struct MeshIDPairHash {
    std::size_t operator()(const std::pair<MeshID, MeshID>& p) const
    {
      return 4096 * p.first + p.second;
    }
  };

  std::unordered_map<std::pair<MeshID, MeshID>, std::vector<std::pair<const libMesh::Elem*, int>>, MeshIDPairHash>
  subdomain_interface_map_;

  // TODO: store proper data types here
  std::unordered_map<MeshID, std::vector<std::pair<const libMesh::Elem*, int>>> surface_map_;

  std::unordered_map<MeshID, std::pair<MeshID, MeshID>> surface_senses_;

};

} // namespace xdg

#endif // include guard