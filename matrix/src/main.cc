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

        dtype *a = NewCPUVector(m * k);
        Mat mat_a = Mat(a, m, k);
        dtype *b = NewCPUVector(n * k);
        Mat mat_b = Mat(b, k, n);
        dtype *c = NewCPUVector(m * n);
        Mat mat_c = Mat(c, m, n);

        cout << "begin cal" << endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            mat_c = mat_a * mat_b;
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        cout << "dim:" << m << "," << n << "," << k << " time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << endl;
    }
}
