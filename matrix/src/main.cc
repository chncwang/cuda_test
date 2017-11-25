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

float** NewGPUVectors(int count, int dim) {
    float**  result = (float**)malloc(count * sizeof(float*));
    for (int i = 0; i<count; ++i) {
        float * vec = NewGPUVector(dim);
        result[i] = vec;
    }
    return result;
}

float **ToGpuVectorArray(float** vec, int len) {
    float **result;
    int size = len * sizeof(float*);
    assert(cudaSuccess == cudaMalloc((void **)&result, size));
    assert(cudaMemcpy(result, vec, size, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    return result;
}

int main() {
    std::vector<int> dims = {50/* , 100, 200, 500, 1000*/};
    std::vector<int> counts = {10/*, 20, 50, 100, 200, 500, 1000*/};
    InitGPU(DEVICE::getInstance(), 4000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    for (auto dim : dims) {
        for (auto count : counts) {
            float** gpu_vec_a = NewGPUVectors(count, dim);
            float **a = ToGpuVectorArray(gpu_vec_a, count);
            float** gpu_vec_b = NewGPUVectors(count, dim);
            float **b = ToGpuVectorArray(gpu_vec_b, count);
            cout << "begin cal" << endl;
            float sum = 0;
            int iter = 1;
            for (int i = 0; i < iter; ++i) {
                //cudaEvent_t start, stop;
                //cudaEventCreate(&start);
                //cudaEventCreate(&stop);
                //cudaEventRecord(start);
                N3LDGTanh(a, b, dim, count);
                //cudaEventRecord(stop);
                //cudaEventSynchronize(stop);
                float mill;
                //cudaEventElapsedTime(&mill, start, stop);
                sum += mill;
                cudaDeviceSynchronize();
            }
            PrintGPUVector(gpu_vec_b[0], 10);
            cudaDeviceSynchronize();
            cout << "dim:" << dim << " count:" <<count << " time:" << sum * 1000 / iter  << endl;
        }
    }

    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
}
