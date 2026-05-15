#ifndef _XDG_CUBQL_BACKEND_H
#define _XDG_CUBQL_BACKEND_H

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

// Guards to prevent CUDA headers from being included in host code, which causes
// failed compilation with LLVM-clang.
#if defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#undef __CUDA_ARCH__
#endif

#include "xdg/error.h"

namespace xdg_cubql_backend {

struct Context {
  int gpuID {0};
  int hostID {omp_get_initial_device()};
};

template<typename T, typename InputT = T>
struct AutoUploadArray {
  AutoUploadArray() = default;

  AutoUploadArray(Context context, const InputT* input, std::size_t count)
    : context(context), count(count)
  {
    if (count == 0) return;

    elements = static_cast<T*>(omp_target_alloc(count * sizeof(T), context.gpuID));
    if (!elements) {
      fatal_error("Failed to allocate cuBQL device array.");
    }
    needsFree = true;

    int result = 0;
    if constexpr (std::is_same_v<T, InputT>) {
      result = omp_target_memcpy(elements,
                                 input,
                                 count * sizeof(T),
                                 0,
                                 0,
                                 context.gpuID,
                                 context.hostID);
    } else {
      std::vector<T> converted(count);
      for (std::size_t i = 0; i < count; ++i) {
        converted[i] = T(input[i]);
      }

      result = omp_target_memcpy(elements,
                                 converted.data(),
                                 count * sizeof(T),
                                 0,
                                 0,
                                 context.gpuID,
                                 context.hostID);
    }

    if (result != 0) {
      reset();
      fatal_error("Failed to upload cuBQL device array.");
    }
  }

  ~AutoUploadArray()
  {
    reset();
  }

  AutoUploadArray(const AutoUploadArray&) = delete;
  AutoUploadArray& operator=(const AutoUploadArray&) = delete;

  AutoUploadArray(AutoUploadArray&& other) noexcept
  {
    *this = std::move(other);
  }

  AutoUploadArray& operator=(AutoUploadArray&& other) noexcept
  {
    if (this != &other) {
      reset();

      context = other.context;
      elements = other.elements;
      count = other.count;
      needsFree = other.needsFree;

      other.elements = nullptr;
      other.count = 0;
      other.needsFree = false;
    }

    return *this;
  }

  void reset()
  {
    if (needsFree && elements) {
      omp_target_free(elements, context.gpuID);
    }

    elements = nullptr;
    count = 0;
    needsFree = false;
  }

  T* get() const { return elements; }

  T* elements {nullptr};
  std::size_t count {0};
  bool needsFree {false};
  Context context {};
};

} // namespace xdg_cubql_backend

#endif // include guard
