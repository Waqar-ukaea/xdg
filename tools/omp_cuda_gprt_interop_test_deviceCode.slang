#include "omp_cuda_gprt_interop_test_shared.h"

/* 
    An empty compute shader so that I have a single entry point for compilation
*/
[shader("compute")]
[numthreads(64, 1, 1)]
void generate_random_rays(uint3 DispatchThreadID: SV_DispatchThreadID) {


}