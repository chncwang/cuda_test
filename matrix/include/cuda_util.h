#ifndef N3LDG_MATRIX_CUDA_UTIL_H
#define N3LDG_MATRIX_CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

#if USE_FLOAT
typedef  float dtype;
#else
typedef  double dtype;
#endif

template<typename T>
void CopyGlobalArray(T *dest, T *src, int length);

void PrintGPUVector(dtype *vec, int dim);

void PrintCPUVector(dtype *vec, int dim);

void InitGPUVector(dtype *vec, int dim);

void InitCPUVector(dtype *vec, int dim);

dtype *NewGPUVector(int dim);

dtype *NewCPUVector(int dim);

void CUBLASAdd(cublasHandle_t handle, dtype *a, dtype *b, int dim);
void CUBLASProduct(cublasHandle_t handle, dtype *a, dtype *b,
        dtype *result, int m, int n, int k);
void CUBLASProductBatch(cublasHandle_t handle, const dtype **vec,
        const dtype **mat, dtype **result, int n, int row, int col);

void cudarraysCopy();

#endif
