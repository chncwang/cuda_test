#include "cuda_util.h"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"
#include <iostream>
#include <array>

using namespace std;

int main() {
    std::vector<int> dims = {50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000};
    InitGPU(DEVICE::getInstance(), 6000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (auto dim : dims) {
        dtype *gpu_vec_a = NewGPUVector(dim);
        dtype *gpu_vec_b = NewGPUVector(dim);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int i = 0; i < 1000000; ++i)
            CUBLASAdd(handle, gpu_vec_a, gpu_vec_b, dim);

        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "dim:" << dim << " time:" << milliseconds << endl;
    }
}
