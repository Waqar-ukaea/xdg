#ifndef _XDG_CUBQL_BACKEND_H
#define _XDG_CUBQL_BACKEND_H

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "cuBQL/bvh.h"
#include "xdg/error.h"

namespace xdg::cubql {

using BinaryBVH = cuBQL::BinaryBVH<float, 3>;
using WideBVH4 = cuBQL::WideBVH<float, 3, 4>;
using WideBVH8 = cuBQL::WideBVH<float, 3, 8>;
using WideBVH16 = cuBQL::WideBVH<float, 3, 16>;

using ActiveBVHType = WideBVH16;

struct Context {
  int gpuID {0};
  int hostID {omp_get_initial_device()};
};



} // namespace xdg::cubql

#endif // include guard
