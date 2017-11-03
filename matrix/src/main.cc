#include "cuda_util.h"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"
#include <iostream>
#include <array>

using namespace std;

int main() {
    std::vector<int> dims = {50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000};
    InitGPU(DEVICE::getInstance(), 6000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (auto dim : dims) {
        dtype *cpu_vec_a = NewCPUVector(dim);
        dtype *cpu_vec_b = NewCPUVector(dim);
        dtype *cpu_vec_c = NewCPUVector(dim);
        Mat mat_a = Mat(cpu_vec_a, dim, 1);
        Mat mat_b = Mat(cpu_vec_b, dim, 1);
        Mat mat_c = Mat(cpu_vec_c, dim, 1);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int i = 0; i < 1000000; ++i)
            mat_c = mat_a + mat_b;

        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "dim:" << dim << " time:" << milliseconds << endl;
    }
}
