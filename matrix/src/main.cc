#include "cuda_util.h"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"
#include <iostream>
#include <array>
#include <chrono>

using namespace std;

int main() {
    std::vector<int> dims = {50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000};
    InitGPU(DEVICE::getInstance(), 4000000000, 0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    for (auto dim : dims) {
        float *gpu_vec_a = NewGPUVector(dim);
        float *gpu_vec_b = NewGPUVector(dim);
        cout << "begin cal" << endl;
        float sum = 0;
        int iter = 100000;
        for (int i = 0; i < iter; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cublasSaxpy(handle, dim, &alpha, gpu_vec_a, 1, gpu_vec_b, 1);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float mill;
            cudaEventElapsedTime(&mill, start, stop);
            sum += mill;
            cudaDeviceSynchronize();
        }

        cout << "dim:" << dim << " time:" << sum * 1000 / iter  << endl;
    }
}
