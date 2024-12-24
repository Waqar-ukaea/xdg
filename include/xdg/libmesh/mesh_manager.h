#ifndef _XDG_LIBMESH_MESH_MANAGER
#define _XDG_LIBMESH_MESH_MANAGER

#include <memory>

#include "xdg/constants.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/error.h"

#include "libmesh/libmesh.h"
#include "libmesh/elem.h"
#include "libmesh/mesh.h"
namespace xdg {


class LibMeshMeshManager : public MeshManager {

public:
  LibMeshMeshManager(void* ptr);

  LibMeshMeshManager();

  ~LibMeshMeshManager();

  // Backend methods (specific to libMesh)
  //! In the case that boundary conditions are not explicitly defined, this method will
  //! discover element faces on sudbomain interfaces, create surfaces and topology accordingly,
  //! and assign transmission boundary conditions for these interfacdes
  void discover_surface_elements();

  //! Initialize libMesh library
  void initialize_libmesh();

  // Interface methods
  MeshLibrary mesh_library() const override { return MeshLibrary::LIBMESH; }

  void load_file(const std::string& filepath) override;

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

  std::vector<MeshID> get_volume_elements(MeshID volume) const override;

  std::vector<MeshID> get_surface_elements(MeshID surface) const override;

  std::vector<Vertex> element_vertices(MeshID element) const override;

  std::array<Vertex, 3> triangle_vertices(MeshID triangle) const override;

  std::vector<MeshID> get_volume_surfaces(MeshID volume) const override;

  MeshID create_volume() override;

  void add_surface_to_volume(MeshID volume, MeshID surface, Sense sense, bool overwrite=false) override;

  std::pair<MeshID, MeshID> surface_senses(MeshID surface) const override;

  Sense surface_sense(MeshID surface, MeshID volume) const override;

  // Accessors
  const libMesh::Mesh* mesh() const { return mesh_.get(); }
  libMesh::Mesh* mesh() { return mesh_.get(); }

  private:

  //! Helper struct for unique identification of an element face
  struct SidePair {

    SidePair() = default;

    SidePair(std::pair<const libMesh::Elem*, int> old_side) {
      side.first = old_side.first;
      side_num_ = old_side.second;
      side.second = old_side.first->neighbor_ptr(side_num_);
      if (side.second != nullptr) {
        set_order();
        set_side_num();
      }
    }

    SidePair(const libMesh::Elem* elem, int side_num) {
      side.first = elem;
      side_num_ = side_num;
      side.second = elem->neighbor_ptr(side_num);
      if (side.second != nullptr) {
        set_order();
        set_side_num();
      }
    }

    SidePair(const libMesh::Elem* elem1, const libMesh::Elem* elem2) {
      side.first = elem1;
      side.second = elem2;
      if (elem1 == nullptr || elem2 == nullptr) {
        fatal_error("SidePair pointer constructor must be used with non-null pointers");
      }
      set_order();
      set_side_num();
    }

    SidePair(std::pair<const libMesh::Elem*, const libMesh::Elem*> side_pair) : side(side_pair) {
      if (side.first == nullptr || side.second == nullptr) {
        fatal_error("SidePair pair constructor must be used with non-null pointers");
      }
      set_order();
      set_side_num();
    }

    void set_order() {
      if (side.first == nullptr && side.second == nullptr) {
        fatal_error("SidePair created with null elements");
      }

      // null pointers come second always
      if (side.first == nullptr) {
        std::swap(side.first, side.second);
        return;
      }

      if (side.second == nullptr) {
        return;
      }

      if (side.first->id() == side.second->id()) {
        fatal_error("SidePair created with same element on both sides");
      }

      if (side.first->id() > side.second->id()) {
        std::swap(side.first, side.second);
      }
    }

    void set_side_num() {
      for (int i = 0; i < side.first->n_sides(); i++) {
        if (side.first->neighbor_ptr(i) == side.second) {
          side_num_ = i;
          break;
        }
      }
      if (side_num_ == -1) {
        fatal_error("SidePair created with elements that are not neighbors");
      }
    }


    bool operator==(const SidePair& other) const
    {
      return first() == other.first() && side_num() == other.side_num(); // || side == std::make_pair(other.side.second, other.side.first);
    }

    bool operator<(const SidePair& other) const
    {
      return side < other.side;
    }

    const libMesh::Elem* first() const { return side.first; }
    const libMesh::Elem* second() const { return side.second; }

    const int32_t side_num() const { return side_num_; }

    const std::unique_ptr<const libMesh::Elem> face_ptr() const {
      return first()->side_ptr(side_num());
    }

    std::pair<const libMesh::Elem*, const libMesh::Elem*> side {nullptr, nullptr};
    int32_t side_num_ {-1};
  }; // SidePair

  struct SidePairHash {

    std::size_t operator()(const SidePair& p) const
    {
      auto hash1 = std::hash<const libMesh::Elem*>{}(p.first());
      auto hash2 = std::hash<int32_t>{}(p.side_num());
      // Combine the two hashes into a single hash value
      // This can be done using a common technique that combines them with XOR and bit shifting
      return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
  };

  MeshID sidepair_id(const SidePair& sidepair) {
    if (sidepair_to_mesh_id_.count(sidepair) == 0) {
      MeshID id = next_sidepair_id();
      mesh_id_to_sidepair_[id] = sidepair;
      if (sidepair_to_mesh_id_.count(sidepair) > 0) {
        fatal_error("SidePair already exists in sidepair_to_mesh_id_");
      }
      sidepair_to_mesh_id_[sidepair] = id;
      return id;
    } else {
      return sidepair_to_mesh_id_[sidepair];
    }
  }

  MeshID sidepair_id(const libMesh::Elem* elem1, const libMesh::Elem* elem2) {
    return sidepair_id(SidePair(elem1, elem2));
  }

  SidePair sidepair(MeshID sidepair) const {
    return mesh_id_to_sidepair_.at(sidepair);
  }

  MeshID next_sidepair_id() const {
    if (mesh_id_to_sidepair_.size() == 0) {
      return 1;
    }
    return std::max_element(mesh_id_to_sidepair_.begin(), mesh_id_to_sidepair_.end())->first + 1;
  }

  struct MeshIDPairHash {
    std::size_t operator()(const std::pair<MeshID, MeshID>& p) const
    {
      // Combine the hashes of the two integers
      std::size_t h1 = std::hash<int32_t>{}(p.first);
      std::size_t h2 = std::hash<int32_t>{}(p.second);
      // Use a bitwise combination for mixing
      return h1 ^ (h2 << 1); // XOR and shift for a simple combination
    }
  };

  // Attributes
  protected:
  std::unique_ptr<libMesh::Mesh> mesh_ {nullptr};
  // TODO: make this global so it isn't owned by a single mesh manager
  std::unique_ptr<libMesh::LibMeshInit> libmesh_init {nullptr};

  // Ugh, double mapping
  std::unordered_map<MeshID, SidePair> mesh_id_to_sidepair_;
  std::unordered_map<SidePair, MeshID, SidePairHash> sidepair_to_mesh_id_;

  // sideset face mapping, stores the element and the side number
  // for each face in the mesh that lies on a boundary
  std::unordered_map<MeshID, std::vector<SidePair>> sideset_element_map_;

  //! Mapping of subdomain interfaces (Identified by subdomain ID pairs) to the
  //! set of element faces that make up the interface
  std::unordered_map<std::pair<MeshID, MeshID>, std::vector<SidePair>, MeshIDPairHash>
  subdomain_interface_map_;

  //! Mapping of surface IDs to the set of element faces that make up the surface,
  //! with the element face represented by assigned XDG IDs
  // TODO: store proper data types here
  std::unordered_map<MeshID, std::vector<MeshID>> surface_map_;

  //! Mapping of surfaces to the volumes on either side. Volumes are ordered
  //! based on their sense with respect to the surface triangles
  std::unordered_map<MeshID, std::pair<MeshID, MeshID>> surface_senses_;
};

} // namespace xdg

#endif // include guard