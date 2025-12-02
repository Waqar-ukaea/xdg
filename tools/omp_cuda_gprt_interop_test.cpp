#include <cassert>
#include <cstdio>
#include <vector>
#include <iostream>

#include <omp.h>
#include "openmp_memory_interop.h"

int main() 
{
  const int N = 1024;
  float* dPtr = nullptr;

  // Create CUDA device pointer outside of OMP
  auto cerr = OMI::malloc(reinterpret_cast<void**>(&dPtr), N * sizeof(float));
  if (cerr != OMI::Error::Success) {
    std::cerr << "cudaMalloc failed: " << OMI::get_error_string(cerr) << "\n";
    return 1;
  }
  std::cout << "Successfully created CUDA device pointer: " << dPtr << "\n";

  // Initailise host data
  std::vector<float> hData(N, 1.0f); // 1024 len array of 1.0f

  // memcpy to device
  cerr = OMI::memcpy_to_device(dPtr, hData.data(), N * sizeof(float));
  if (cerr != OMI::Error::Success) {
    std::cerr << "cudaMemcpy H2D failed: "
              << OMI::get_error_string(cerr) << "\n";
    OMI::free(dPtr);
    return 1;
  }
  std::cout << "Succesfully called cuda memcpy via OMI interop layer" << std::endl;

  // Lets multiply every element by i*2 on device
  #pragma omp target teams distribute parallel for is_device_ptr(dPtr)
    for (int i = 0; i < N; ++i) {
      dPtr[i] = i*2.0f;
      if (i == 0) printf("dPtr[%d] = %f \n", i, dPtr[i]);
      if (i == 512) printf("dPtr[%d] = %f \n", i, dPtr[i]);
      if (i == 1023) printf("dPtr[%d] = %f \n", i, dPtr[i]);      
    }


  return 0;

}