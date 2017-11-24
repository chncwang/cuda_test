#include "cuda_util.h"
#include "custom_kernel.cuh"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"
#include <iostream>
#include <array>
#include <chrono>

using namespace std;

std::vector<float*> NewGPUVectors(int count, int dim) {
    std::vector<float*> result;
    for (int i = 0; i<count; ++i) {
        float * vec = NewGPUVector(dim);
        result.push_back(vec);
    }
    return result;
}

void ReleaseGPUVectors(std::vector<float *>* vec) {
    for (float *v : *vec) {
        cnmemFree(v, NULL);
    }
}

int main() {
    std::vector<int> dims = {50, 100, 200, 500, 1000};
    std::vector<int> counts = {10, 20, 50, 100, 200, 500, 1000};
    InitGPU(DEVICE::getInstance(), 4000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto gpu_vec_a = NewGPUVectors(count, dim);
            auto gpu_vec_b = NewGPUVectors(count, dim);
            cout << "begin cal" << endl;
            float sum = 0;
            int iter = 1000;
            for (int i = 0; i < iter; ++i) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                for (int j = 0; j < count; ++j) {
                    N3LDGTanh(gpu_vec_a.at(j), gpu_vec_b.at(j), dim);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float mill;
                cudaEventElapsedTime(&mill, start, stop);
                sum += mill;
                cudaDeviceSynchronize();
            }
            cout << "dim:" << dim << " count:" <<count << " time:" << sum * 1000 / iter  << endl;
        }
    }

    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
}
