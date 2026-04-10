#ifndef XDG_SHARED_CONSTANTS_H
#define XDG_SHARED_CONSTANTS_H

namespace xdg {

  // Mesh identifer type
  typedef int32_t MeshID;
  typedef int32_t MeshIndex;

  // Null mesh ID
  static const MeshID ID_NONE = -1;
  static const MeshIndex INDEX_NONE = -1;

  // Scene/Tree ID
  typedef int32_t TreeID;
  typedef TreeID SurfaceTreeID;
  typedef TreeID ElementTreeID;

  static const TreeID TREE_NONE = -1;

  enum PointInVolume : int { 
    UNSET = -1,
    OUTSIDE = 0, 
    INSIDE = 1 
  };

  enum HitOrientation : int {
    ANY = -1,
    EXITING = 0,
    ENTERING = 1,
  };

}

#endif // XDG_SHARED_CONSTANTS_H
