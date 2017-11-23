#ifndef N3LDG_MATRIX_CUDA_UTIL_CUH
#define N3LDG_MATRIX_CUDA_UTIL_CUH

#include "cnmem.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

namespace n3ldg_kernel {
    constexpr int THREAD_COUNT_PER_BLOCK = 1000;
    constexpr int MAX_BLOCK_COUNT = 56;

    int BlockCount(int size) {
        int n = (size + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK;
        return n > MAX_BLOCK_COUNT ? MAX_BLOCK_COUNT : n;
    }

    __global__ void Copy(float *src, float *dest, int len) {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < len)
            dest[index] = src;
    }

    void CopyArray(float *src, float *dest, int len) {
        Copy<<<BlockCount(len) ,THREAD_COUNT_PER_BLOCK>>>(src, dest, len);
    }
}

};

#endif
