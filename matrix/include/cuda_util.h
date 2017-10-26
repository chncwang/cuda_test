#ifndef N3LDG_MATRIX_CUDA_UTIL_H
#define N3LDG_MATRIX_CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

template<typename T>
void CopyGlobalArray(T *dest, T *src, int length);

void TestCublasSum();

void TestCudaUtil();


#endif
