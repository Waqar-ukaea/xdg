
#include <map>
#include <memory>
#include <set>

#include "xdg/mesh_manager_interface.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

using namespace xdg;

using OverlapMap = std::map<std::set<int>, Position>;

// check mesh manager instance for overlaps
void check_instance_for_overlaps(std::shared_ptr<XDG> xdg,
                                 OverlapMap& overlap_map);

void report_overlaps(const OverlapMap& overlap_map);