#include "cuda_util.h"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"
#include <iostream>
#include <array>
#include <utility>
#include <chrono>

using namespace std;

int main() {
    std::vector<pair<int, int>> dims = {
        make_pair(50, 50),
        make_pair(100, 100),
        make_pair(200, 200),
        make_pair(2000, 100),
        make_pair(5000, 100),
        make_pair(10000, 100)
    };
    InitGPU(DEVICE::getInstance(), 6000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (auto dim : dims) {
        dtype *gpu_vec_a = NewGPUVector(dim.first);
        dtype *gpu_vec_b = NewGPUVector(dim.first * dim.second);
        dtype *gpu_vec_c = NewGPUVector(dim.first);

        cout << "begin cal" << endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            CUBLASProduct(handle, gpu_vec_a, gpu_vec_b, gpu_vec_c, dim.first,
                    dim.second);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        cout << "dim:" << dim.first << "," << dim.second << " time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << endl;
    }
}
