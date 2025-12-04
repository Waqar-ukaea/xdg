#include "openmp_memory_interop.h"

// This TU is the ONLY place we include CUDA runtime
#include <cuda_runtime.h>

namespace OMI {

Error malloc(void** ptr, std::size_t bytes)
{
  return static_cast<Error>(cudaMalloc(ptr, bytes));
}

Error free(void* ptr)
{
  return static_cast<Error>(cudaFree(ptr));
}

Error memcpy_to_device(void* dst_device, const void* src_host, std::size_t bytes) {
  return static_cast<Error>(cudaMemcpy(dst_device, src_host, bytes,
                                       cudaMemcpyHostToDevice));
}

Error memcpy_from_device(void* dst_host, const void* src_device, std::size_t bytes) {
  return static_cast<Error>(cudaMemcpy(dst_host, src_device, bytes,
                                       cudaMemcpyDeviceToHost));
}

const char* get_error_string(Error err)
{
  // reinterpret the OMI::Error as a cudaError_t and ask CUDA for the string
  return cudaGetErrorString(static_cast<cudaError_t>(err));
}

} // namespace OMI
