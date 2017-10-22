#include "cuda_util.h"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"

int main() {
    InitGPU(DEVICE::getInstance(), 4000000000, 1);
    TestCudaUtil();
    cudaDeviceSynchronize();
}
