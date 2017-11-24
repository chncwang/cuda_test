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
            std::chrono::steady_clock::time_point begin =
                std::chrono::steady_clock::now();

            for (int i = 0; i < 100000; ++i) {
                for (int j = 0; j < count; ++j) {
                    N3LDGTanh(gpu_vec_a.at(j), gpu_vec_b.at(j), dim);
                }
                cudaDeviceSynchronize();
            }
            std::chrono::steady_clock::time_point end =
                std::chrono::steady_clock::now();

            cout << "dim:" << dim << " count:" << count << " time:" <<
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                        begin).count() / 1000<< endl;
        }
    }

    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
}
