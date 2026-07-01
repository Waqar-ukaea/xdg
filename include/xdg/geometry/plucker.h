#ifndef _XDG_PLUCKER_H
#define _XDG_PLUCKER_H

#include "dp_math.h"

namespace xdg {

/*
 * Triangle vertex ordering convention:
 *
 *      v2
 *     /  \
 *    /    \
 *   /      \
 *  /        \
 * v0--------v1
 *
 * The vertices are ordered counter-clockwise when viewed from the front face
 * (normal pointing out of the plane). This ordering is based on the reference:
 * https://doi.org/10.1002/cnm.1237
 */

struct PluckerIntersectionResult {
  bool hit = false;    // Whether an intersection occurred
  double t = 0.0;      // Distance along the ray to the intersection point
};

/* Function to return the vertex with the lowest coordinates. To force the same
  ray-edge computation, the Plücker test needs to use consistent edge
  representation. This would be more simple with MOAB handles instead of
  coordinates... 
*/
inline bool first(dp::vec3 a, dp::vec3 b) {
  if (a[0] < b[0]) return true;  
  if (a[0] > b[0]) return false;

  if (a[1] < b[1]) return true;  
  if (a[1] > b[1]) return false;

  return a[2] < b[2];
}

inline double plucker_edge_test(dp::vec3 vertexa, dp::vec3 vertexb,
                                dp::vec3 ray, dp::vec3 ray_normal)
{
  const bool vertexa_is_first = first(vertexa, vertexb);

  const double edge_x = vertexa_is_first ? vertexb[0] - vertexa[0] : vertexa[0] - vertexb[0];
  const double edge_y = vertexa_is_first ? vertexb[1] - vertexa[1] : vertexa[1] - vertexb[1];
  const double edge_z = vertexa_is_first ? vertexb[2] - vertexa[2] : vertexa[2] - vertexb[2];

  const double origin_x = vertexa_is_first ? vertexa[0] : vertexb[0];
  const double origin_y = vertexa_is_first ? vertexa[1] : vertexb[1];
  const double origin_z = vertexa_is_first ? vertexa[2] : vertexb[2];

  const double edge_normal_x = edge_y * origin_z - edge_z * origin_y;
  const double edge_normal_y = edge_z * origin_x - edge_x * origin_z;
  const double edge_normal_z = edge_x * origin_y - edge_y * origin_x;

  double pip =
    ray[0] * edge_normal_x + ray[1] * edge_normal_y + ray[2] * edge_normal_z +
    ray_normal[0] * edge_x + ray_normal[1] * edge_y + ray_normal[2] * edge_z;

  if (!vertexa_is_first) {
    pip = -pip;
  }
  if (dp::DBL_ZERO_TOL > dp::abs(pip))  // <-- absd
    pip = 0.0;
  return pip;
}

inline PluckerIntersectionResult plucker_ray_tri_intersect(dp::vec3 vertices[3],
                                dp::vec3 origin,
                                dp::vec3 direction,
                                double tMax,
                                double tMin,
                                bool useOrientation,
                                int orientation)
{
  double dist_out = dp::INFTY;

  const dp::vec3 raya = direction;
  const dp::vec3 rayb = dp::cross(direction, origin);

  // Determine the value of the first Plucker coordinate from edge 0
  double plucker_coord0 =
    plucker_edge_test(vertices[0], vertices[1], raya, rayb);

  // If orientation is set, confirm that sign of plucker_coordinate indicate
  // correct orientation of intersection
  if (useOrientation && orientation * plucker_coord0 > 0) {
    return {false, 0.0};
  }

  // Determine the value of the second Plucker coordinate from edge 1
  double plucker_coord1 =
    plucker_edge_test(vertices[1], vertices[2], raya, rayb);

  // If orientation is set, confirm that sign of plucker_coordinate indicate
  // correct orientation of intersection
  if (useOrientation) {
    if (orientation * plucker_coord1 > 0) {
      return {false, 0.0};
    }
    // If the orientation is not specified, all plucker_coords must be the same
    // sign or zero.
  } else if ((0.0 < plucker_coord0 && 0.0 > plucker_coord1) ||
             (0.0 > plucker_coord0 && 0.0 < plucker_coord1)) {
    return {false, 0.0};
  }

  // Determine the value of the third Plucker coordinate from edge 2
  double plucker_coord2 =
    plucker_edge_test(vertices[2], vertices[0], raya, rayb);

  // If orientation is set, confirm that sign of plucker_coordinate indicate
  // correct orientation of intersection
  if (useOrientation) {
    if (orientation * plucker_coord2 > 0) {
      return {false, 0.0};
    }
    // If the orientation is not specified, all plucker_coords must be the same
    // sign or zero.
  } else if ((0.0 < plucker_coord1 && 0.0 > plucker_coord2) ||
             (0.0 > plucker_coord1 && 0.0 < plucker_coord2) ||
             (0.0 < plucker_coord0 && 0.0 > plucker_coord2) ||
             (0.0 > plucker_coord0 && 0.0 < plucker_coord2)) {
    return {false, 0.0};
  }

  // check for coplanar case to avoid dividing by zero
  if (0.0 == plucker_coord0 && 0.0 == plucker_coord1 && 0.0 == plucker_coord2) {
    return {false, 0.0};
  }

  // To minimize numerical error, get index of largest magnitude direction.
  int idx = 0;
  double max_abs_dir = 0;
  for (uint i = 0; i < 3; ++i) {
    if (dp::abs(direction[i]) > max_abs_dir) {
      idx = i;
      max_abs_dir = dp::abs(direction[i]);
    }
  }

  // get the distance to intersection
  const double inverse_sum =
    1.0 / (plucker_coord0 + plucker_coord1 + plucker_coord2);

  const double intersection_component =
    (plucker_coord0 * vertices[2][idx] +
     plucker_coord1 * vertices[0][idx] +
     plucker_coord2 * vertices[1][idx]) * inverse_sum;

  dist_out = (intersection_component - origin[idx]) / direction[idx];

  // Barycentric coords check
  double u = plucker_coord2 * inverse_sum;
  double v = plucker_coord0 * inverse_sum;

  // Barycentric coords check
  if (u < 0.0 || v < 0.0 || (u + v) > 1.0) {
      dist_out = -1.0;
  }

  // is the intersection within distance limits?
  if (dist_out < tMin || dist_out > tMax) return {false, 0.0};

  return {true, dist_out};
}

} // namespace xdg

#endif // include guard
