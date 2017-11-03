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
    InitGPU(DEVICE::getInstance(), 6000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (auto dim : dims) {
        dtype *gpu_vec_a = NewGPUVector(dim);
        dtype *gpu_vec_b = NewGPUVector(dim);
        cout << "begin cal" << endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        for (int i = 0; i < 1000000; ++i)
            CUBLASAdd(handle, gpu_vec_a, gpu_vec_b, dim);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        cout << "dim:" << dim << " time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000<< endl;
    }
}
