#include "cuda_util.h"
#include <cuda_profiler_api.h>
#include "cnmem.h"
#include <cassert>
#include "gpu_matrix.h"
#include <iostream>
#include <array>
#include <utility>
#include <chrono>
#include <tuple>

using namespace std;

int main() {
    std::vector<tuple<int, int, int>> dims = {
        make_tuple(100, 20, 100),
        make_tuple(100, 50, 100),
        make_tuple(100, 100, 100)
    };
    InitGPU(DEVICE::getInstance(), 6000000000, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    using std::get;
    for (auto dim : dims) {
        int m = get<0>(dim);
        int n = get<1>(dim);
        int k = get<2>(dim);
        dtype *gpu_vec_a = NewGPUVector(m * k);
        dtype *gpu_vec_b = NewGPUVector(n * k);
        dtype *gpu_vec_c = NewGPUVector(m * n);

        cout << "begin cal" << endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            CUBLASProduct(handle, gpu_vec_a, gpu_vec_b, gpu_vec_c, m, n, k);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        cout << "dim:" << m << "," << n << "," << k << " time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << endl;
    }
}
