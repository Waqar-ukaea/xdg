#include "xdg/overlap.h"
#include "xdg/progressBar.h"
#include <fstream>
#include <map>


using namespace xdg;

void check_location_for_overlap(std::shared_ptr<XDG> xdg,
                                const std::vector<MeshID>& allVols, Vertex loc,
                                Direction dir, OverlapMap& overlap_map) {

  std::set<MeshID> vols_found;
  double bump = 1E-9;

  // move the point slightly off the vertex
  loc += dir * bump;

  for (const auto& vol : allVols) {
    bool pointInVol = false;
    pointInVol = xdg->point_in_volume(vol, loc, &dir, nullptr);

    if (pointInVol) {
      vols_found.insert(vol);
    }
  }

  if (vols_found.size() > 1) {
#pragma omp critical
    overlap_map[vols_found] = loc;
  }

  // move the point slightly off the vertex
  dir *= -1;
  loc += dir * 2.0 * bump;
  vols_found.clear();

  for (const auto& vol : allVols) {
		bool pointInVol = false;
    pointInVol = xdg->point_in_volume(vol, loc, &dir, nullptr);

    if (pointInVol) {
      vols_found.insert(vol);
    }
  }

  if (vols_found.size() > 1) {
#pragma omp critical
    overlap_map[vols_found] = loc;
  }
}

void check_instance_for_overlaps(std::shared_ptr<XDG> xdg,
                                 OverlapMap& overlap_map,
                                 bool checkEdges = false,
                                 bool verboseOutput = false) {
  auto mm = xdg->mesh_manager();
  auto allVols = mm->volumes();
  auto allSurfs = mm->surfaces();
  std::vector<Vertex> allVerts;
  int totalElements = 0;

  /* Loop over surface instead of all volumes as it results in duplicating the number of checks when it does the nodes in the
     implicit complement as well as the explicit volumes. Also removes an uneccesary layer of nesting. */

  for (const auto& surf:allSurfs){  
      auto surfElements = mm->get_surface_elements(surf);
      totalElements += surfElements.size();
      for (const auto& tri:surfElements){
        auto triVert = mm->triangle_vertices(tri);
        // Push vertices in triangle to end of array
        allVerts.push_back(triVert[0]);
        allVerts.push_back(triVert[1]);
        allVerts.push_back(triVert[2]);
      }
    }

  std::cout << "Number of vertices checked = " << allVerts.size() << "\n" << std::endl;
  // number of locations we'll be checking
  int numLocations = allVerts.size(); // + pnts_per_edge * all_edges.size();
  int numChecked = 1;

  Direction dir = {0.1, 0.1, 0.1};
  dir = dir.normalize();

  ProgressBar vertexProgBar;

  std::cout << "Checking for overlapped regions at element vertices..." << std::endl;
// first check all triangle vertex locations
#pragma omp parallel shared(overlap_map, numChecked)
  {
#pragma omp for schedule(auto)
    for (size_t i = 0; i < allVerts.size(); i++) {
      Vertex vert = allVerts[i];
      check_location_for_overlap(xdg, allVols, vert, dir, overlap_map);

#pragma omp critical
      vertexProgBar.set_value(100.0 * (double)numChecked++ / (double)numLocations);
    }
  }

  if (overlap_map.empty()) { 
    std::cout << "No Overlaps found at vertices! \n" << std::endl;
  }

  // if we aren't checking along edges, return early
  if (!checkEdges) {
    return;
  }

  std::cout << "Checking for overlapped regions along element edges..." << std::endl; 
  
  ProgressBar edgeProgBar;
  std::vector<Position> edgeOverlapLocs;

  // Number of rays cast along edges = number_of_elements * (number_of_edges * 2) * (number_of_vols - parent_vols)
  int totalEdgeRays = totalElements*(3)*(allVols.size()-2); 
  int edgeRaysCast = 0;

#pragma omp parallel shared(overlap_map, edgeRaysCast)
  {
  // now check along triangle edges
  // (curve edges are likely in here too,
  //  but it isn't hurting anything to check more locations)

#pragma omp for schedule(auto)
    for (const auto& surf:allSurfs)
    {
      auto parentVols = mm->get_parent_volumes(surf);
      std::vector<MeshID> volsToCheck;
      std::copy_if(allVols.begin(), allVols.end(), std::back_inserter(volsToCheck), [&parentVols](MeshID vol)
      {
        return vol != parentVols.first && vol != parentVols.second;
      });
      auto elementsOnSurf = mm->get_surface_elements(surf);
      for (const auto& element:elementsOnSurf) {
        auto tri = mm->triangle_vertices(element);
        auto rayQueries = return_ray_queries(tri);
        for (const auto& query:rayQueries)
        {
          auto volHit = check_along_edge(xdg, mm, query, volsToCheck, edgeOverlapLocs);
          if (volHit != -1) 
          {
            overlap_map[{volHit, parentVols.first}] = edgeOverlapLocs.back();
          }
          #pragma omp atomic
            edgeRaysCast++;
          #pragma omp critical
            edgeProgBar.set_value(100.0 * (double)edgeRaysCast / (double)totalEdgeRays);
        }
      }
    }
  }

  if (overlap_map.empty()) { 
    std::cout << "No Overlaps found along edges! \n" << std::endl;
  }

  if (verboseOutput) {
    // Write out edge overlap locations to stdout
    std::cout << "\nVerbose ouptut enabled. Printing the locations of all overlaps along edges..." << std::endl;
    for (auto& loc:edgeOverlapLocs)
    {
      std::cout << loc.x << ", " << loc.y << ", " << loc.z << "\n";
    }
  }
  return;
}

void report_overlaps(const OverlapMap& overlap_map) {
  std::cout << "Overlap locations found: " << overlap_map.size() << std::endl;

  for (const auto& entry : overlap_map) {
    std::set<MeshID> overlap_vols = entry.first;
    Position loc = entry.second;

    std::cout << "Overlap Location: " << loc[0] << " " << loc[1] << " "
              << loc[2] << std::endl;
    std::cout << "Overlapping volumes: ";
    for (const auto& i : overlap_vols) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
}

/* Return rayQueries along element edges. Currently limited to Triangles as ElementVertices is defined as a std::array<xdg::vertex, 3> 
   but the rest of the function body could easily work with a container of any size so could readily be generalised 
   to work with quads. */
std::vector<EdgeRayQuery> return_ray_queries(const ElementVertices &element)
{
  std::vector<EdgeRayQuery> rayQueries;
  // Loop through each edge of the element (triangle or quad)
  for (size_t vertex = 0; vertex < element.size(); ++vertex) {
    // Wrap around to the first vertex when at the final
    size_t nextVertex = (vertex + 1) % element.size();
    const auto& v1 = element[vertex];
    const auto& v2 = element[nextVertex];

    Direction dir = v2 - v1;    
    double edgeLength = dir.length();
    dir /= edgeLength; 

    // Add the edge ray query
    rayQueries.push_back({v1, dir, edgeLength});
  }
  return rayQueries;
}

// Fire a ray along a single edge direction firing against all volumes except for the current surfaces' parent volumes (fowards+reverse sense). Returns volume ID of the surface hit
MeshID check_along_edge(std::shared_ptr<XDG> xdg, 
                       std::shared_ptr<MeshManager> mm, 
                       const EdgeRayQuery& rayquery, 
                       const std::vector<MeshID>& volsToCheck, 
                       std::vector<Position>& edgeOverlapLocs)
{
  auto origin = rayquery.origin;
  auto direction = rayquery.direction;
  auto distanceMax = rayquery.edgeLength;
  std::pair<MeshID, MeshID> volHit;
  int counter=0;
  for (const auto& testVol:volsToCheck)
  {
    auto ray = xdg->ray_fire(testVol, origin, direction, distanceMax); 
    double rayDistance = ray.first;
    MeshID surfHit = ray.second;
    if (surfHit != -1) // if surface hit (Valid MeshID returned)
    {
      counter++;
      volHit = mm->get_parent_volumes(surfHit);

      Position collisionPoint = {origin.x + rayDistance*direction.x, origin.y + rayDistance*direction.y, origin.z + rayDistance*direction.z}; 
      // Check if overlap location already added to list from another ray
      if (std::find(edgeOverlapLocs.begin(), edgeOverlapLocs.end(), collisionPoint) == edgeOverlapLocs.end()) {
        edgeOverlapLocs.push_back(collisionPoint);
      }
      return volHit.first;
    }
  }
  return -1;
}

