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
        int row = get<0>(dim);
        int n = get<1>(dim);
        int col = get<2>(dim);
        dtype **matrix_arr = (dtype**)malloc(n *sizeof(dtype*));
        assert(matrix_arr != NULL);
        for (int i = 0; i< n; ++i) {
            matrix_arr[i] = NewGPUVector(row * col);
        }
        dtype **gpu_matrix_arr;
        assert(cnmemMalloc((void**)&gpu_matrix_arr, n * sizeof(dtype*), NULL) ==
                CNMEM_STATUS_SUCCESS);
        CCE(cudaMemcpy(gpu_matrix_arr, matrix_arr, n * sizeof(dtype*),
                    cudaMemcpyHostToDevice));

        dtype **vec_arr = (dtype**)malloc(n * sizeof(dtype*));
        for (int i = 0; i< n; ++i) {
            vec_arr[i] = NewGPUVector(col);
        }
        dtype **gpu_vec_arr;
        assert(cnmemMalloc((void**)&gpu_vec_arr, n * sizeof(dtype*), NULL) ==
                CNMEM_STATUS_SUCCESS);
        CCE(cudaMemcpy(gpu_vec_arr, vec_arr, n * sizeof(dtype*),
                    cudaMemcpyHostToDevice));

        dtype **result_arr = (dtype**)malloc(n * sizeof(dtype*));
        for (int i = 0; i< n; ++i) {
            result_arr[i] = NewGPUVector(col);
        }
        dtype **gpu_result_arr;
        assert(cnmemMalloc((void**)&gpu_result_arr, n * sizeof(dtype*), NULL) ==
                CNMEM_STATUS_SUCCESS);
        CCE(cudaMemcpy(gpu_result_arr, result_arr, n * sizeof(dtype*),
                    cudaMemcpyHostToDevice));

        cout << "begin cal" << endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            CUBLASProductBatch(handle, (const dtype**)gpu_vec_arr,
                    (const dtype**)gpu_matrix_arr, gpu_result_arr, n, row, col);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        cout << "dim:" << row << "," << col << "," << n << " time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << endl;
    }
}
