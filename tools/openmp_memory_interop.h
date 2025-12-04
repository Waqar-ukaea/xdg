#ifndef OPENMP_MEMORY_INTEROP_h
#define OPENMP_MEMORY_INTEROP_h

#include <cstddef>

// Openmp-Memory-Interop (OMI) namespace 
namespace OMI {

// simple error code, mirror cudaError_t semantics
  enum class Error : int {
    Success = 0
    // maybe more later
  };
// Allocate device memory
Error malloc(void** ptr, std::size_t bytes);

// Free device memory
Error free(void* ptr);

// host <-> device copies
Error memcpy_to_device(void* dst_device, const void* src_host, std::size_t bytes);
Error memcpy_from_device(void* dst_host, const void* src_device, std::size_t bytes);

// Get error string
const char* get_error_string(Error err);

} // namespace Openmp-Memory-Interop (OMI)

#endif