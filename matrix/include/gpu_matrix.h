#ifndef _gpu_matrix_
#define _gpu_matrix_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include "cnmem.h"
#include <assert.h>
#include "cpu_matrix.h"



#include <iostream>
#include <vector>

using namespace std;

#define CCE(x) checkCudaErrors(x)

const int THREADS = 128;


#if USE_FLOAT
typedef  float dtype;
#else
typedef  double dtype;
#endif



class cpu_matrix;

class DEVICE {
public:
	cnmemDevice_t device;
public:
	//static void initial() { device.numStreams = 0;device.streams = NULL; device.streamSizes = NULL; }
	static cnmemDevice_t& getInstance() {
		static DEVICE D;
		return D.device;
	}
};

void InitGPU(cnmemDevice_t &device, size_t mem_size = 1000000000, int device_id = 0);
void FinalizeGPU();

class gpu_matrix
{
// public:
	// Device *dev;
public:
	dtype *v;
	int row, col, size;
public:
	gpu_matrix();
	~gpu_matrix();
	void init(int r, int c);
	gpu_matrix(dtype* v_data, size_t r, size_t c);
	void delloc();
	void resize(int r, int c);
	// void delloc() { cudaFree(v); }
	inline void zero() { if(v) cudaMemset((void*)v, 0, size * sizeof(dtype)); }
	void zeros();
	void ones();
	void random(dtype bound);//it is not implemented
	void assign(int icol, int jrow, dtype value){
		CCE(cudaMemcpy(v + icol*row + jrow, &value, sizeof(dtype), cudaMemcpyHostToDevice));
	}
	dtype get(int icol, int jrow){
		dtype r;
		CCE(cudaMemcpy(&r, v + icol*row + jrow, sizeof(dtype), cudaMemcpyDeviceToHost));
		return r;
	}
	gpu_matrix& operator=(const gpu_matrix &rhs);
	gpu_matrix& operator=(const cpu_matrix &rhs);
	inline dtype* operator[](const int icol){ return v + icol*row; }
	inline const dtype* operator[](const int icol)const{ return v+icol*row; }
	void transpose(const gpu_matrix &rhs); 
	// void transpose();
	void add(const gpu_matrix &a, const gpu_matrix &b);
	void sub(const gpu_matrix &a, const gpu_matrix &b);
	void multiply(const gpu_matrix &a, const gpu_matrix &b);
	void divide(const gpu_matrix &a, const gpu_matrix &b);
	void product(const gpu_matrix &a, const gpu_matrix &b);
	void product(dtype alpha, dtype beta, bool aTranspose, bool bTranspose, const gpu_matrix &a, const gpu_matrix &b);
	void self_add(const gpu_matrix &rhs);
	void self_sub(const gpu_matrix &rhs);
	void self_multiply(const gpu_matrix &rhs);
	void self_divide(const gpu_matrix &rhs);
	void tanh(const gpu_matrix &rhs);
	void sigmoid(const gpu_matrix &rhs);
	void relu(const gpu_matrix &rhs);
	void leaky_relu(const gpu_matrix &rhs);
	void exp(const gpu_matrix &rhs);
	void square(const gpu_matrix &rhs);
	void cube(const gpu_matrix &rhs);
	void dtanh(const gpu_matrix &a, const gpu_matrix &b);
	void dsigmoid(const gpu_matrix &a, const gpu_matrix &b);
	void drelu(const gpu_matrix &a, const gpu_matrix &b);
	void dleaky_relu(const gpu_matrix &a, const gpu_matrix &b);
	void dexp(const gpu_matrix &a, const gpu_matrix &b);
	void dsquare(const gpu_matrix &a, const gpu_matrix &b);
	void dcube(const gpu_matrix &a, const gpu_matrix &b);
	void activate(const gpu_matrix &rhs, int functor);
	void dactivate(const gpu_matrix &a, const gpu_matrix &b, int functor);
	// void max_pooling(const gpu_matrix &rhs);
	// void min_pooling(const gpu_matrix &rhs);
	// void average_pooling(const gpu_matrix &rhs);
	
	void dropout(gpu_matrix &drop_mask, dtype drop_value, int seed);
	void assign(dtype scale);
	void lookup(const gpu_matrix &E, int idx){
		assert(E.row == size);
		CCE(cudaMemcpy(v, E[idx], sizeof(dtype) * size, cudaMemcpyDeviceToDevice));
	}

	void display(){
		dtype *tv = new dtype[size];
		CCE(cudaMemcpy(tv, v, size * sizeof(dtype), cudaMemcpyDeviceToHost));
		for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
				std::cout << tv[j*row + i] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
		
		delete[] tv;
	};
	
	void concat(const vector<gpu_matrix> &rhs_vec);
	
	void big_copy_small(int offset, const gpu_matrix &rhs);
	
	void small_copy_big(const gpu_matrix &rhs, int offset);
	
	void short_add_long(const gpu_matrix &a, const gpu_matrix &b, int offset);

	dtype square_sum();
	
	dtype square_sum(int icol);
	
	void mat_copy_vec(int icol, const gpu_matrix &rhs);
	
	void vec_copy_mat(const gpu_matrix &rhs, int icol);
	
	void vec_add_mat(const gpu_matrix &a, const gpu_matrix &b, int icol);
	
	void mat_add_vec(const gpu_matrix &a, int icol, const gpu_matrix &b);
	
	void zeros(int icol);
	
	void multiply(const gpu_matrix &rhs, dtype scale);

	void multiply(const gpu_matrix &rhs, int icol, dtype scale);
	
	void norm2one();
	
	void self_add(int icol, int irow, dtype scale);
	
	void special_add(int index, const gpu_matrix &a, dtype m, const gpu_matrix &b, dtype n);
	
	void special_add1(int index, const gpu_matrix &a, const gpu_matrix &b);
	
	void special_add2(int index, const gpu_matrix &a, const gpu_matrix &b, const gpu_matrix &c, dtype  alpha, dtype eps);

	void special_add(const gpu_matrix &a, dtype m, const gpu_matrix &b, dtype n);

	void special_add1(const gpu_matrix &a, const gpu_matrix &b);

	void special_add2(const gpu_matrix &a, const gpu_matrix &b, const gpu_matrix &c, dtype  alpha, dtype eps);
	
	void save(std::ofstream &os) const {
		dtype *tv = new dtype[size];
		CCE(cudaMemcpy(tv, v, sizeof(dtype) * size, cudaMemcpyDeviceToHost));
		os << size << " " << row << " " << col << std::endl;
		os << tv[0];
		for (int idx = 1; idx < size; idx++) {
			os << " " << tv[idx];
		}
		os << std::endl;
		delete[] tv;
	}
	
	void load(std::ifstream &is) {
		int curSize, curRow, curCol;
		is >> curSize;
		is >> curRow;
		is >> curCol;
		init(curRow, curCol);
		dtype *tv = new dtype[curSize];
		for (int idx = 0; idx < size; idx++) {
			is >> tv[idx];
		}
		CCE(cudaMemcpy(v, tv, sizeof(size), cudaMemcpyHostToDevice));
		delete []tv;
	}
	
	
	// void sqrt(const gpu_matrix &a);
	void mat_combine_from_vecs(const vector<gpu_matrix*> &ins);
	void dense_to_sparse_block_assign(vector<gpu_matrix*> &outs, vector<int> &indices, int n);
	void sparse_to_dense_block_add(vector<gpu_matrix*> &losses, vector<int> &indices, int n);
	void vec_accumulate_from_mat(vector<gpu_matrix*> &outs);
	void vec_accumulate_from_mat(gpu_matrix* out);
	void vec_separate_from_mat(vector<gpu_matrix*> &outs);
	
		void save(int icol, std::ofstream &os){
		dtype *tv = new dtype[row];
		CCE(cudaMemcpy(tv, v + row*icol, sizeof(dtype) * row, cudaMemcpyDeviceToHost));
		os << size << " " << row << " " << col << std::endl;
		os << tv[0];
		for (int idx = 1; idx < row; idx++) {
			os << " " << tv[idx];
		}
		os << std::endl;
		delete[] tv;
	}
};

void concatenate(vector<gpu_matrix*> &in, int stride, vector<gpu_matrix*> &out);
void max_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask);
void min_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask);
inline int n_blocks(int size, int block_size) {
		return size / block_size + ((size % block_size == 0)? 0 : 1);
	}


// template <typename T>
// gpu_matrix<T> gpu_matrix<T>::operator * (const device_matrix<T>& rhs) const
// {
	// gpu_matrix<T> res(row, col, 0);
// #if USE_FLOAT
	// CCE(cublasSgemm(,,))
// #else
	// dtype 
	// CCE(cublasDgemm(d->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, ,,, ));
// #endif
	
// }

#endif
