#ifndef N3LDG_MATRIX_CUDA_UTIL_H
#define N3LDG_MATRIX_CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

template<typename T>
void CopyGlobalArray(T *dest, T *src, int length);

void SumGlobalSArray(float** arr, float* sum, int size, int vec_length);
void SumGlobalDArray(double** arr, double* sum, int size, int vec_length);

void TestCublasSum();

void TestCudaUtil();

void TestMultiply();

void TestSumGlobalDarray();

#endif
