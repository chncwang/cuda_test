#include "functors.h"
#include "cnmem.h"
#include "gpu_matrix.h"
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include<thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include "cuda_util.h"
#include <random>

#define threadsize  32

#if USE_FLOAT
#define CALL_CUBLAS(func) cublasS##func
#else
#define CALL_CUBLAS(func) cublasD##func
#endif

class CUBLAS_HANDLE {
    private:
        cublasHandle_t handle;
        CUBLAS_HANDLE() {
            CCE(cublasCreate(&handle));
        }
        ~CUBLAS_HANDLE() { CCE(cublasDestroy(handle)); }
    public:
        static cublasHandle_t& getInstance() {
            static CUBLAS_HANDLE H;
            return H.handle;
        }
};


struct prg
{
    dtype a, b, dp_val;

    __host__ __device__
        prg(dtype _a = 0.0, dtype _b = 1.0, dtype _dp_val = 0.0) : a(_a), b(_b), dp_val(_dp_val) {};


    __host__ __device__ inline
        dtype operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<dtype> dist(a, b);
            rng.discard(n);

            if (dist(rng) <= dp_val)
                return 0;
            else
                return 1;
        }
};

struct gRand
{
    dtype a, b;

    __host__ __device__
        gRand(dtype _a = 0.0, dtype _b = 1.0) : a(_a), b(_b) {}


    __host__ __device__ inline
        dtype operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<dtype> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

void InitGPU(cnmemDevice_t &device, size_t mem_size, int device_id)
{	
    memset(&device, 0, sizeof(device));
    device.device = device_id;
    device.size = mem_size;
    cudaSetDevice(device_id);
    std::cout << "device id " << device_id << std::endl;
    assert(CNMEM_STATUS_SUCCESS == cnmemInit(1, &device, CNMEM_FLAGS_CANNOT_GROW));
    cudaSetDevice(device_id);
}

void FinalizeGPU()
{
    cnmemStatus_t status = cnmemFinalize();
    if (status != CNMEM_STATUS_SUCCESS) {
    }
}

void gpu_matrix::random(dtype bound){
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::device_ptr<dtype> ptr(v);
    thrust::transform(index_sequence_begin, index_sequence_begin + size, ptr, gRand(-bound, bound));
}

gpu_matrix::~gpu_matrix(){
    delloc();
    row = 0;
    col = 0;
    size = 0;
}

void gpu_matrix::delloc(){
    if(v && !is_v_weak_ref){
        cnmemStatus_t status = cnmemFree(v, NULL);
        if (status != CNMEM_STATUS_SUCCESS) {
            // TODO
        }
    }
    v = NULL;
}

void gpu_matrix::init(int r, int c, bool b_is_v_weak_ref){
    is_v_weak_ref = b_is_v_weak_ref;
    row = r;
    col = c;
    size = row * col; // TODO this member should be removed
    if(size != 0 && !is_v_weak_ref){
        assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&v, sizeof(dtype) * size, NULL));
        zero();
    }
} 

gpu_matrix::gpu_matrix():row(0), col(0), size(0), v(NULL) {}

gpu_matrix::gpu_matrix(dtype* v_data, size_t r, size_t c){
    abort();
    init(r, c);
    CCE(cudaMemcpy(v, v_data, sizeof(dtype) * row * col, cudaMemcpyHostToDevice));
}

void gpu_matrix::resize(int r, int c)
{
    if(row == r && col == c)
        return;

    if(v){
        assert(CNMEM_STATUS_SUCCESS == cnmemFree(v, NULL));
    }

    init(r, c);
}

void gpu_matrix::zeros(){
    CCE(cudaMemset((void*)v, 0, sizeof(dtype) * size));
}

void gpu_matrix::ones(){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::transform(ptr_a, ptr_a + row * col, ptr_a, Assign(1));
}

gpu_matrix& gpu_matrix::operator=(const gpu_matrix &rhs){
    assert((row == rhs.row) && (col == rhs.col) && (size == rhs.size));
    CCE(cudaMemcpy(v, rhs.v, row * col * sizeof(dtype), cudaMemcpyDeviceToDevice)); // TODO may be too slow
    return *this;
}

gpu_matrix& gpu_matrix::operator=(const cpu_matrix &rhs){
    assert((row == rhs.row) && (col == rhs.col) && (size == rhs.size));
    CCE(cudaMemcpy(v, rhs.v, row * col * sizeof(dtype), cudaMemcpyHostToDevice));
    return *this;
}

void gpu_matrix::add(const gpu_matrix &a, const gpu_matrix &b){
    dtype alpha = 1.0;
    if (typeid(dtype) == typeid(float)) {
        cublasScopy(CUBLAS_HANDLE::getInstance(), row * col, (float*)a.v, 1,
                (float*)v, 1);
        cublasSaxpy(CUBLAS_HANDLE::getInstance(), row * col, (float*)&alpha,
                (float*)b.v, 1, (float*)v, 1);
    } else if (typeid(dtype) == typeid(double)) {
        cublasDcopy(CUBLAS_HANDLE::getInstance(), row * col, (double*)a.v, 1,
                (double*)v, 1);
        cublasDaxpy(CUBLAS_HANDLE::getInstance(), row * col, (double*)&alpha,
                (double*)b.v, 1, (double*)v, 1);
    } else {
        abort();
    }
}

void gpu_matrix::sub(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr0(v);
    thrust::device_ptr<dtype> ptr1(a.v);
    thrust::device_ptr<dtype> ptr2(b.v);
    thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::minus<dtype>());
}

void gpu_matrix::multiply(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr0(v);
    thrust::device_ptr<dtype> ptr1(a.v);
    thrust::device_ptr<dtype> ptr2(b.v);
    thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::multiplies<dtype>());
}

void gpu_matrix::divide(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr0(v);
    thrust::device_ptr<dtype> ptr1(a.v);
    thrust::device_ptr<dtype> ptr2(b.v);
    thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::divides<dtype>());
}

void gpu_matrix::self_add(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::plus<dtype>());
}

void gpu_matrix::self_sub(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::minus<dtype>());
}

void gpu_matrix::self_multiply(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::multiplies<dtype>());
}

void gpu_matrix::self_divide(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::divides<dtype>());
}

void gpu_matrix::product(const gpu_matrix &a, const gpu_matrix &b){
    int m = row;
    int n = col;
    int k = a.col;
    int lda = a.row;
    int ldb = b.row;
    int ldc = row;
    dtype alpha = 1.0;
    dtype beta = 0.0;

#if USE_FLOAT
    CCE(cublasSgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#else
    CCE(cublasDgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#endif
}

void gpu_matrix::product(dtype alpha, dtype beta, bool aTranspose, bool bTranspose, const gpu_matrix &a, const gpu_matrix &b){
    int m = row;
    int  n = col;
    int k = aTranspose ? a.row : a.col;
    int lda = a.row;
    int ldb = b.row;
    int ldc = row;
    cublasOperation_t opa = aTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opb = bTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;

#if USE_FLOAT
    CCE(cublasSgemm(CUBLAS_HANDLE::getInstance(), opa, opb, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#else
    CCE(cublasDgemm(CUBLAS_HANDLE::getInstance(), opa, opb, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#endif
}


void gpu_matrix::tanh(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Tanh());
}

void gpu_matrix::sigmoid(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Sigmoid());
}

void gpu_matrix::relu(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Relu());
}

void gpu_matrix::leaky_relu(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);	
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Leaky_relu());
}

void gpu_matrix::exp(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);	
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Exp());
}

void gpu_matrix::square(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);	
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Square());
}

void gpu_matrix::cube(const gpu_matrix &rhs){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);	
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Cube());
}

void gpu_matrix::activate(const gpu_matrix &rhs, int functor){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v);	
    thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Activate(functor));
}

void gpu_matrix::dactivate(const gpu_matrix &a, const gpu_matrix &b, int functor){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);	
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dActivate(functor));
}

void gpu_matrix::dtanh(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dTanh());
}

void gpu_matrix::dsigmoid(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dSigmoid());
}

void gpu_matrix::drelu(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dRelu());
}

void gpu_matrix::dleaky_relu(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dLeaky_relu());
}

void gpu_matrix::dexp(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dExp());
}

void gpu_matrix::dsquare(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dSquare());
}

void gpu_matrix::dcube(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(a.v);	
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dCube());
}

void gpu_matrix::dropout(gpu_matrix &drop_mask, dtype drop_value, int seed){
    thrust::counting_iterator<unsigned int> index_sequence_begin(seed);
    thrust::device_ptr<dtype> ptr(drop_mask.v);
    thrust::transform(index_sequence_begin, index_sequence_begin + size, ptr, prg(0.0, 1.0, drop_value));

    self_multiply(drop_mask);
}

void gpu_matrix::assign(dtype scale){
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::transform(ptr_a, ptr_a + row * col, ptr_a, Assign(scale));
}


void gpu_matrix::concat(const vector<gpu_matrix> &rhs_vec){
    thrust::device_ptr<dtype> ptr_a(v);	
    assert(col == rhs_vec.size());
    assert(row == rhs_vec[0].size);
    for(int i=0; i<col; i++){
        thrust::device_ptr<dtype> ptr_b(rhs_vec[i].v);
        //CCE(cudaMemcpy(v + i*row, rhs_vec[i].v, sizeof(dtype) * row, cudaMemcpyDeviceToDevice));
        thrust::transform(ptr_b, ptr_b + row, ptr_a + i*row, Assignab());
    }
}


void gpu_matrix::big_copy_small(int offset, const gpu_matrix &rhs){
    CopyGlobalArray<dtype>(v+offset, rhs.v, rhs.size);
}

void gpu_matrix::big_copy_small_async(int offset, const gpu_matrix &rhs, cudaStream_t stream){
    CCE(cudaMemcpyAsync(v+offset, rhs.v, rhs.size*sizeof(dtype), cudaMemcpyDeviceToDevice, stream));
}

void gpu_matrix::small_copy_big(const gpu_matrix &rhs, int offset){
    CCE(cudaMemcpy(v, rhs.v+offset, size*sizeof(dtype), cudaMemcpyDeviceToDevice));
}


void gpu_matrix::short_add_long(const gpu_matrix &a, const gpu_matrix &b, int offset){
    if (typeid(dtype) == typeid(float)) {
        static float alpha = 1.0;
        cublasSaxpy(CUBLAS_HANDLE::getInstance(), b.size, &alpha, (float*)(b.v + offset), 1, (float*)a.v, 1);
    } else if (typeid(dtype) == typeid(double)) {
        static double alpha = 1.0;
        cublasDaxpy(CUBLAS_HANDLE::getInstance(), b.size, &alpha, (double*)(b.v + offset), 1, (double*)a.v, 1);
    } else {
        abort();
    }
}


dtype gpu_matrix::square_sum(){
    thrust::device_ptr<dtype> ptr_a(v);
    dtype init = 0.0;
    return thrust::transform_reduce(ptr_a, ptr_a + size, Square(), init, thrust::plus<dtype>());
}

dtype gpu_matrix::square_sum(int icol){
    thrust::device_ptr<dtype> ptr_a(v + icol * row);
    dtype init = 0.0;
    return thrust::transform_reduce(ptr_a, ptr_a + row, Square(), init, thrust::plus<dtype>());
}


void gpu_matrix::mat_copy_vec(int icol, const gpu_matrix &rhs){
    assert(rhs.size == row);
    thrust::device_ptr<dtype> ptr_a(v + icol*row);
    thrust::device_ptr<dtype> ptr_b(rhs.v);
    thrust::transform(ptr_b, ptr_b+rhs.size, ptr_a, Assignab());
}

void gpu_matrix::vec_copy_mat(const gpu_matrix &rhs, int icol){
    assert(rhs.row == size);
    thrust::device_ptr<dtype> ptr_a(v);
    thrust::device_ptr<dtype> ptr_b(rhs.v + icol*rhs.row);
    thrust::transform(ptr_b, ptr_b+rhs.row, ptr_a, Assignab());
}

void gpu_matrix::vec_add_mat(const gpu_matrix &a, const gpu_matrix &b, int icol){
    assert(a.size == b.row);
    thrust::device_ptr<dtype> ptr_0(v);
    thrust::device_ptr<dtype> ptr_1(a.v);
    thrust::device_ptr<dtype> ptr_2(b.v + icol*b.row);
    thrust::transform(ptr_1, ptr_1 + a.size, ptr_2, ptr_0, thrust::plus<dtype>());
}

void gpu_matrix::mat_add_vec(const gpu_matrix &a, int icol, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr_a(v + icol*row);
    thrust::device_ptr<dtype> ptr_b(a.v + icol * a.row);
    thrust::device_ptr<dtype> ptr_c(b.v);
    thrust::transform(ptr_b, ptr_b + a.row, ptr_c, ptr_a, thrust::plus<dtype>());
}

void gpu_matrix::zeros(int icol) {
    thrust::device_ptr<dtype> ptr(v+icol*row);
    thrust::transform(ptr, ptr+row, ptr, Assign(0));
}


void gpu_matrix::multiply(const gpu_matrix &rhs, int icol, dtype scale){
    thrust::device_ptr<dtype> ptr0(v + icol*row);
    thrust::device_ptr<dtype> ptr1(rhs.v + icol*row);
    thrust::transform(ptr1, ptr1 + row, ptr0, multi_c(scale));
}

void gpu_matrix::multiply(const gpu_matrix &rhs, dtype scale) {
    thrust::device_ptr<dtype> ptr0(v);
    thrust::device_ptr<dtype> ptr1(rhs.v);
    thrust::transform(ptr1, ptr1 + size, ptr0, multi_c(scale));
}

void gpu_matrix::norm2one(){
    dtype sum;
    for (int idx = 0; idx < col; idx++) {
        sum = 0.000001;
        sum = this->square_sum(idx);
        dtype scale = std::sqrt(sum);
        this->multiply(*this, idx, 1.0/scale);
    }
}

void gpu_matrix::self_add(int icol, int irow, dtype scale) {
    thrust::device_ptr<dtype> ptr(v + icol*row + irow);
    thrust::transform(ptr, ptr + 1, ptr, self_add_c(scale));
}

void gpu_matrix::special_add(int index, const gpu_matrix &a, dtype m, const gpu_matrix &b, dtype n){
    thrust::device_ptr<dtype> ptr0(v+row*index);
    thrust::device_ptr<dtype> ptr1(a.v+a.row*index);
    thrust::device_ptr<dtype> ptr2(b.v+b.row*index);
    thrust::transform(ptr1, ptr1 + a.row, ptr2, ptr0, special_add_func(m, n));
}

void gpu_matrix::special_add1(int index, const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr0(v+row*index);
    thrust::device_ptr<dtype> ptr1(a.v+a.row*index);
    thrust::device_ptr<dtype> ptr2(b.v+b.row*index);
    thrust::transform(ptr1, ptr1 + a.row, ptr2, ptr0, special_add1_func());
}

void gpu_matrix::special_add2(int index, const gpu_matrix &a, const gpu_matrix &b, const gpu_matrix &c, dtype alpha, dtype eps){
    gpu_matrix tmp;
    tmp.init(row, 1);

    thrust::device_ptr<dtype> ptr0(tmp.v);
    thrust::device_ptr<dtype> ptr1(b.v+b.row*index);
    thrust::device_ptr<dtype> ptr2(c.v+c.row*index);
    thrust::transform(ptr1, ptr1 + b.row, ptr2, ptr0, special_add2_func(alpha, eps));

    thrust::device_ptr<dtype> ptr3(a.v+a.row*index);
    thrust::device_ptr<dtype> ptr4(v+row*index);
    thrust::transform(ptr3, ptr3 + a.row, ptr0, ptr4, thrust::minus<dtype>());
}

void gpu_matrix::special_add(const gpu_matrix &a, dtype m, const gpu_matrix &b, dtype n){
    thrust::device_ptr<dtype> ptr0(v);
    thrust::device_ptr<dtype> ptr1(a.v);
    thrust::device_ptr<dtype> ptr2(b.v);
    thrust::transform(ptr1, ptr1 + a.size, ptr2, ptr0, special_add_func(m, n));
}

void gpu_matrix::special_add1(const gpu_matrix &a, const gpu_matrix &b){
    thrust::device_ptr<dtype> ptr0(v);
    thrust::device_ptr<dtype> ptr1(a.v);
    thrust::device_ptr<dtype> ptr2(b.v);
    thrust::transform(ptr1, ptr1 + a.size, ptr2, ptr0, special_add1_func());
}

void gpu_matrix::special_add2(const gpu_matrix &a, const gpu_matrix &b, const gpu_matrix &c, dtype alpha, dtype eps){
    gpu_matrix tmp;
    tmp.init(row, 1);

    thrust::device_ptr<dtype> ptr0(tmp.v);
    thrust::device_ptr<dtype> ptr1(b.v);
    thrust::device_ptr<dtype> ptr2(c.v);
    thrust::transform(ptr1, ptr1 + b.size, ptr2, ptr0, special_add2_func(alpha, eps));

    thrust::device_ptr<dtype> ptr3(a.v);
    thrust::device_ptr<dtype> ptr4(v);
    thrust::transform(ptr3, ptr3 + a.size, ptr0, ptr4, thrust::minus<dtype>());
}

// void gpu_matrix::sqrt(const gpu_matrix &a) {
// thrust::device_ptr<dtype> ptr0(v);
// thrust::device_ptr<dtype> ptr1(a.v);
// thrust::transform(ptr0, ptr0 + size, ptr1, xxx());
// }

__global__ void get_max_index(dtype** matrixes, dtype** mask, int size) {
    int max_iter = -1;
    int i = threadIdx.x;
    for (int j = 0; j<size; j++) {
        dtype * matrixesj = matrixes[j];
        dtype matrixesji = matrixesj[i];
        if ((max_iter == -1) || (matrixesji > matrixes[max_iter][i])) {
            max_iter = j;
        }
    }
    //mask is on gpu
    mask[max_iter][i] = 1.0;
}

dtype **copy_matrix_ptrs_to_array(vector<gpu_matrix> &matrix) {
    int size = matrix.size();
    void *ptr;
    int mem_size = sizeof(dtype*) * size;
    cnmemStatus_t status = cnmemMalloc(&ptr, mem_size, NULL);
    assert(CNMEM_STATUS_SUCCESS == status);
    dtype **vs = static_cast<dtype**>(ptr);
    dtype **cpu_mem = (dtype**)malloc(mem_size);
    for (int i = 0; i < size; ++i) {
        cpu_mem[i] = matrix.at(i).v;
    }
    CCE(cudaMemcpy(vs, cpu_mem, mem_size, cudaMemcpyHostToDevice));

    return vs;
}

void max_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask) {
    int dim = ins[0].size;
    dtype **ins_arr = copy_matrix_ptrs_to_array(ins);
    dtype **mask_arr = copy_matrix_ptrs_to_array(mask);
    get_max_index<<<1, dim>>>(ins_arr, mask_arr, ins.size());
    checkCudaErrors(cudaDeviceSynchronize());
    assert(cnmemFree(ins_arr, NULL) == CNMEM_STATUS_SUCCESS);
    assert(cnmemFree(mask_arr, NULL) == CNMEM_STATUS_SUCCESS);
}

void min_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask) {
    int dim = ins[0].size;
    int size = ins.size();
    vector<cpu_matrix> t_ins;// needn't delloc manually

    t_ins.resize(ins.size());
    for (int i = 0; i<t_ins.size(); i++) {
        t_ins[i].init(ins[i].row, ins[i].col);
        t_ins[i] = ins[i];
    }


    for (int i = 0; i<dim; i++) {
        int min_iter = -1;
        for (int j = 0; j<size; j++) {
            if ((min_iter == -1) || (t_ins[j].get(0, i) < t_ins[min_iter].get(0, i))) {
                min_iter = j;
            }
        }
        //mask is on gpu
        mask[min_iter].assign(0, i, 1.0);
    }
}

#if __CUDA_ARCH__ < 600 
static __inline__ __device__ double m_atomicAdd(double* address, double val) 
{ 
    unsigned long long int* address_as_ull = (unsigned long long int*)address; 
    unsigned long long int old = *address_as_ull, assumed; 
    do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, 
                __double_as_longlong(val + 
                    __longlong_as_double(assumed))); 

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
    } while (assumed != old); 

    return __longlong_as_double(old); 
}
#endif


template<typename Real>
__global__ void _mat_combine_from_vecs(Real* trg, Real** srcs, int row, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.
    int index_out = i + j * row;

    if (i < row && j < col){
        trg[index_out] = static_cast<Real>(srcs[j][i]);
    }
}



void gpu_matrix::mat_combine_from_vecs(const vector<gpu_matrix*> &ins){
    int ins_size = ins.size();


    assert(col == ins.size());
    void *ptr;

    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * ins_size, NULL));	

    dtype** srcs = static_cast<dtype**>(ptr);


    vector<dtype*> locs(ins_size);

    for(int i=0; i<ins_size; i++){

        locs[i] = ins[i]->v;
    }

    CCE(cudaMemcpy(srcs, &(locs)[0], sizeof(dtype**) * ins_size, cudaMemcpyHostToDevice));


    dim3 dimBlock( threadsize,  threadsize);
    dim3 dimGrid(n_blocks(row,  threadsize), n_blocks(col,  threadsize));

    _mat_combine_from_vecs<<<dimGrid , dimBlock>>>(v, srcs, row, col);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}


template<typename Real>
__global__ void _vec_accumulate_from_mat(Real** trgs, Real* src, int row, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.
    int index_out = i + j * row;

    if (i < row && j < col){
        trgs[j][i] += static_cast<Real>(src[index_out]);
    }
}

void gpu_matrix::vec_accumulate_from_mat(vector<gpu_matrix*> &outs) {
    int outs_size = outs.size();
    assert(outs_size == col && row == outs[0]->row);

    void *ptr;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * outs_size, NULL));	
    dtype** trgs = static_cast<dtype**>(ptr);

    vector<dtype*> locs(outs_size);

    for(int i=0; i<outs_size; i++){
        locs[i] = outs[i]->v;
    }

    CCE(cudaMemcpy(trgs, &(locs)[0], sizeof(dtype**) * outs_size, cudaMemcpyHostToDevice));

    dim3 dimBlock( threadsize,  threadsize);
    dim3 dimGrid(n_blocks(row,  threadsize), n_blocks(col,  threadsize));

    _vec_accumulate_from_mat<<<dimGrid , dimBlock>>>(trgs, v, row, col);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}

template<typename Real>
__global__ void _vec_accumulate_from_mat(Real* trgs, Real* src, int row, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.
    int src_index = i + j * row;

    if (i < row && j < col){
        //m_atomicAdd((double*)(trgs+i), (double)src[src_index]);
    }
}


void gpu_matrix::vec_accumulate_from_mat(gpu_matrix* out){
    assert(row == out->row);

    dim3 dimBlock( threadsize,  threadsize);
    dim3 dimGrid(n_blocks(row,  threadsize), n_blocks(col,  threadsize));

    _vec_accumulate_from_mat<<<dimGrid, dimBlock>>>(out->v, v, row, col);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}



__global__ void _vec_separate_from_mat(dtype** trgs, dtype* src, int row, int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.
    int index_out = i + j * row;

    if (i < row && j < col){
        trgs[j][i] = src[index_out];
    }
}


void gpu_matrix::vec_separate_from_mat(vector<gpu_matrix*> &outs) {
    int outs_size = outs.size();
    assert(outs_size == col && row == outs[0]->row);

    void *ptr;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * outs_size, NULL));	
    dtype** trgs = static_cast<dtype**>(ptr);

    vector<dtype*> locs(outs_size);

    for(int i=0; i<outs_size; i++){
        locs[i] = outs[i]->v;
    }

    CCE(cudaMemcpy(trgs, &(locs)[0], sizeof(dtype**) * outs_size, cudaMemcpyHostToDevice));

    dim3 dimBlock( threadsize,  threadsize);
    dim3 dimGrid(n_blocks(row,  threadsize), n_blocks(col,  threadsize));

    _vec_separate_from_mat<<<dimGrid , dimBlock>>>(trgs, v, row, col);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}


template<typename Real>
__global__ void _dense_to_sparse_block_assign(Real** trg, Real* src, int *idx, int row, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.

    if (i<row && j<n){
        trg[j][i] = src[idx[j]*row+i];
    }
}

void gpu_matrix::dense_to_sparse_block_assign(vector<gpu_matrix*> &outs, vector<int> &indices, int n){
    void* ptr_a;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr_a, sizeof(dtype*) * n, NULL));
    dtype** trg = static_cast<dtype**>(ptr_a);
    void* ptr_b;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr_b, sizeof(int) * n, NULL));
    int* idx =   static_cast<int*>(ptr_b);

    vector<dtype*> locs(n);
    for(int i=0; i<n; i++){
        locs[i] = outs[i]->v;
    }


    CCE(cudaMemcpy(trg, &(locs)[0], sizeof(dtype**) * n, cudaMemcpyHostToDevice));
    CCE(cudaMemcpy(idx, &(indices)[0], sizeof(int) * n, cudaMemcpyHostToDevice));

    dim3 dimBlock( threadsize,  threadsize);
    dim3 dimGrid(n_blocks(row,  threadsize), n_blocks(n,  threadsize));

    _dense_to_sparse_block_assign<<<dimGrid , dimBlock>>>(trg, v, idx, row, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr_a, NULL));
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr_b, NULL));
}



__global__ void _sparse_to_dense_block_add(dtype* trg, dtype** srcs, int *idx, int bsize, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.

    if (i<bsize && j<n){
        //m_atomicAdd((double*)(trg + idx[j]*bsize + i),(double)srcs[j][i]);
    }
}

void gpu_matrix::sparse_to_dense_block_add(vector<gpu_matrix*> &losses, vector<int> &indices, int n){
    void* ptr_a;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr_a, sizeof(dtype*) * n, NULL));
    dtype** srcs = static_cast<dtype**>(ptr_a);
    void* ptr_b;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr_b, sizeof(int) * n, NULL));
    int* idx =   static_cast<int*>(ptr_b);

    vector<dtype*> locs(n);
    for(int i=0; i<n; i++){
        locs[i] = losses[i]->v;
    }


    CCE(cudaMemcpy(srcs, &(locs)[0], sizeof(dtype**) * n, cudaMemcpyHostToDevice));
    CCE(cudaMemcpy(idx, &(indices)[0], sizeof(int) * n, cudaMemcpyHostToDevice));

    dim3 dimBlock( threadsize,  threadsize);
    dim3 dimGrid(n_blocks(row,  threadsize), n_blocks(n,  threadsize));

    _sparse_to_dense_block_add<<<dimGrid , dimBlock>>>(v, srcs, idx, row, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr_a, NULL));
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr_b, NULL));
}

__global__ void concatenate(dtype** srcs, dtype **trgs, dtype **len, int stride, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.

    // srcs[0][0] = 1.0;
    // srcs[3][0] = 1.0;
    // srcs[6][0] = 1.0;

    // trgs[0][0] = 1.0;
    // trgs[1][0] = 1.0;
    // trgs[2][0] = 1.0;

    if(i<n && (i/stride)<m  && j<(unsigned long)len[i]) {
        int offset = 0;
        for(int index = i - i%stride; index < i; index++) {
            offset += (unsigned long)len[index];
        }
        trgs[i/stride][offset+j] = srcs[i][j];
    }
}

void concatenate(vector<gpu_matrix*> &in, int stride, vector<gpu_matrix*> &out) {
    int n = in.size();
    int m = out.size();
    assert(m == n/stride);

    size_t* len = new size_t[n];
    size_t max_len = 0;

    for(int i=0; i<n; i++) {
        len[i] = in[i]->size;
        if(max_len < len[i]) {
            max_len = len[i];
        }
    }

    vector<dtype*> locs(2*n+m);
    for(int i=0; i<n; i++) {
        locs[i] = in[i]->v;
    }
    for(int i=0; i<m; i++) {
        locs[i+n] = out[i]->v;
    }
    for(int i=0; i<n; i++) {
        locs[i+n+m] = (dtype*)len[i];
    }

    void *ptr;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * (locs.size()), NULL));
    dtype **srcs = static_cast<dtype**>(ptr);
    CCE(cudaMemcpy(srcs, &(locs)[0], sizeof(dtype**) * locs.size(), cudaMemcpyHostToDevice));

    dtype **trgs = srcs + n;
    dtype **lens = srcs + n + m;


    dim3 dimBlock(32, 32);
    dim3 dimGrid(n_blocks(n, 32), n_blocks(max_len, 32));

    concatenate<<<dimBlock, dimBlock>>>(srcs, trgs, lens, stride, n, m);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));


    delete [] len;
}

__global__  void kernel_addition(dtype **srcs1, dtype **srcs2, dtype **trgs, int n, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j =  blockIdx.y * blockDim.y + threadIdx.y; // col-index.

    if(i<n && j<dim){
        trgs[i][j] = srcs1[i][j] + srcs2[i][j];
    }
}

void Addition(vector<gpu_matrix*> &out, vector<gpu_matrix*> &in1, vector<gpu_matrix*> &in2) {
    assert(out.size() == in1.size());
    assert(in1.size() == in2.size());
    int nSize = in1.size();
    int dim = in1[0]->size;

    vector<dtype*> locs(nSize*3);
    for(int i=0; i<nSize; i++) {
        locs[i] = out[i]->v;
    }
    for(int i=0; i<nSize; i++) {
        locs[i+nSize] = in1[i]->v;
    }
    for(int i=0; i<nSize; i++) {
        locs[i+nSize*2] = in2[i]->v;
    }

    void *ptr;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * (locs.size()), NULL));
    dtype **trgs = static_cast<dtype**>(ptr);
    CCE(cudaMemcpy(trgs, &(locs)[0], sizeof(dtype**) * locs.size(), cudaMemcpyHostToDevice));

    dtype **srcs1 = trgs + nSize;
    dtype **srcs2 = trgs + nSize*2;


    dim3 dimBlock(32, 32);
    dim3 dimGrid(n_blocks(nSize, 32), n_blocks(dim, 32));
    kernel_addition<<<dimGrid, dimBlock>>>(srcs1, srcs2, trgs, nSize, dim);
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}



__global__  void kernel_multi(dtype **srcs1, dtype **srcs2, dtype **trgs, int n, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j =  blockIdx.y * blockDim.y + threadIdx.y; // col-index.

    if(i<n && j<dim){
        trgs[i][j] = srcs1[i][j] * srcs2[i][j];
    }
}

void Multi(vector<gpu_matrix*> &out, vector<gpu_matrix*> &in1, vector<gpu_matrix*> &in2) {
    assert(out.size() == in1.size());
    assert(in1.size() == in2.size());
    int nSize = in1.size();
    int dim = in1[0]->size;

    vector<dtype*> locs(nSize*3);
    for(int i=0; i<nSize; i++) {
        locs[i] = out[i]->v;
    }
    for(int i=0; i<nSize; i++) {
        locs[i+nSize] = in1[i]->v;
    }
    for(int i=0; i<nSize; i++) {
        locs[i+nSize*2] = in2[i]->v;
    }

    void *ptr;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * (locs.size()), NULL));
    dtype **trgs = static_cast<dtype**>(ptr);
    CCE(cudaMemcpy(trgs, &(locs)[0], sizeof(dtype**) * locs.size(), cudaMemcpyHostToDevice));

    dtype **srcs1 = trgs + nSize;
    dtype **srcs2 = trgs + nSize*2;


    dim3 dimBlock(32, 32);
    dim3 dimGrid(n_blocks(nSize, 32), n_blocks(dim, 32));
    kernel_multi<<<dimGrid, dimBlock>>>(srcs1, srcs2, trgs, nSize, dim);
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}


__global__  void kernel_accumulate_addition_multi(dtype **srcs1, dtype **srcs2, dtype **trgs, int n, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j =  blockIdx.y * blockDim.y + threadIdx.y; // col-index.

    if(i<n && j<dim){
        trgs[i][j] += srcs1[i][j] * srcs2[i][j];
    }
}

void Accumulate_Addition_Multi(vector<gpu_matrix*> &out, vector<gpu_matrix*> &in1, vector<gpu_matrix*> &in2){
    assert(out.size() == in1.size());
    assert(in1.size() == in2.size());
    int nSize = in1.size();
    int dim = in1[0]->size;

    vector<dtype*> locs(nSize*3);
    for(int i=0; i<nSize; i++) {
        locs[i] = out[i]->v;
    }
    for(int i=0; i<nSize; i++) {
        locs[i+nSize] = in1[i]->v;
    }
    for(int i=0; i<nSize; i++) {
        locs[i+nSize*2] = in2[i]->v;
    }


    void *ptr;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(dtype*) * (locs.size()), NULL));
    dtype **trgs = static_cast<dtype**>(ptr);
    CCE(cudaMemcpy(trgs, &(locs)[0], sizeof(dtype**) * locs.size(), cudaMemcpyHostToDevice));

    dtype **srcs1 = trgs + nSize;
    dtype **srcs2 = trgs + nSize*2;

    dim3 dimBlock(32, 32);
    dim3 dimGrid(n_blocks(nSize, 32), n_blocks(dim, 32));
    kernel_accumulate_addition_multi<<<dimGrid, dimBlock>>>(srcs1, srcs2, trgs, nSize, dim);
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}


__global__ void kernel_param_update_adam(dtype *g, dtype *v, dtype *mean, dtype *square, int row, int col, dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps, dtype lr_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j =  blockIdx.y * blockDim.y + threadIdx.y; 

    if(i<row && j<col) {
        int index = j*row + i;
        if(col > 1 && row >1){  
            g[index] = g[index] + v[index]*reg;
        }
        mean[index] = belta1 * mean[index] + (1-belta1)*g[index];
        square[index] = belta2*square[index] + (1-belta2)*g[index]*g[index];
        v[index] = v[index] - mean[index] * lr_t / sqrt((square[index] + eps));
    }
}

void Param_Update_Adam(gpu_matrix& grad, gpu_matrix &val, gpu_matrix &aux_mean, gpu_matrix &aux_square, dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps, int iter) {


    int row = val.row;
    int col = val.col;

    dim3 dimBlock(32, 32);
    dim3 dimGrid(n_blocks(row, 32), n_blocks(col, 32));
    dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
    kernel_param_update_adam<<<dimGrid, dimBlock>>>(grad.v, val.v, aux_mean.v, aux_square.v, row,  col, belta1, belta2, alpha, reg, eps, lr_t);
}


__global__ void kernel_sparseparam_update_adam(dtype *g, dtype *v, dtype *mean, dtype *square, dtype* last_update, int* ids , int row, int col, dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j =  blockIdx.y * blockDim.y + threadIdx.y; 

    //if(i<row && j<n && indexers[ids[j]])
    if(i<row && j<col) {
        if(ids[j] == 1){
            int index = j*row + i;
            g[index] = g[index] + v[index]*reg;
            mean[index] = belta1 * mean[index] + (1-belta1)*g[index];
            square[index] = belta2*square[index] + (1-belta2)*g[index]*g[index];

            dtype lr_t = alpha * sqrt(1 - pow(belta2, last_update[j] + 1)) / (1 - pow(belta1, last_update[j] + 1));;

            v[index] = v[index] - mean[index] * lr_t / sqrt((square[index] + eps));

            last_update[j] += 1;
        }
    }
}

void SparseParam_Update_Adam(gpu_matrix& grad, 
        gpu_matrix &val, 
        gpu_matrix &aux_mean,
        gpu_matrix &aux_square,
        gpu_matrix &last_update,
        vector<int>& ids,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps
        )  
{
    int *ptr = NULL;
    assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&ptr, sizeof(int) * (ids.size()), NULL));


    CCE(cudaMemcpy(ptr, &(ids)[0], sizeof(int) * (ids.size()), cudaMemcpyHostToDevice));

    int row = val.row;
    int col = val.col;

    dim3 dimBlock(32, 32);
    dim3 dimGrid(n_blocks(row, 32), n_blocks(col, 32));

    kernel_sparseparam_update_adam<<<dimGrid, dimBlock>>>(grad.v, val.v, aux_mean.v, aux_square.v, last_update.v, ptr, row, col, belta1, belta2, alpha, reg, eps);
    assert(CNMEM_STATUS_SUCCESS == cnmemFree(ptr, NULL));
}

__global__ void ker_add(dtype *v, dtype scale, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n) {
        v[id] += n;
    }
} 


void gpu_matrix::add(const gpu_matrix &rhs, dtype scale) {
    dim3 dimBlock(32);
    dim3 dimGrid(n_blocks(size, 32));
    ker_add<<<dimGrid, dimBlock>>>(rhs.v, scale, rhs.size);
}

__global__ void CopyElement(char *dest, char *src) {
    int index = threadIdx.x;
    dest[index] = src[index];
}

template<typename T>
void CopyGlobalArray(T *dest, T *src, int length) {
    char *dest_int = (char*)(dest);
    char *src_int = (char*)(src);
    CopyElement<<<1, length * sizeof(T)>>>(dest_int, src_int);
}

__global__ void GlobalAssignVector(dtype *vec, int index, dtype v) {
    vec[index] = v;
}

__global__ void GlobalPrintVector(dtype *vec, int dim) {
    for (int i = 0; i < dim; ++i) {
        printf("%f, ", vec[i]);
    }
    printf("\n");
}

void PrintGPUVector(dtype *vec, int dim) {
    GlobalPrintVector<<<1, 1>>>(vec, dim);
}

void PrintCPUVector(dtype *vec, int dim) {
    for (int i = 0; i < dim; ++i ) {
        cout << vec[dim] << ", ";
    }
    cout << endl;
}

void InitGPUVector(dtype *vec, int dim) {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<> dist(-10, 10);
    for (int i = 0; i < dim; ++i) {
        GlobalAssignVector<<<1, 1>>>(vec, i, dist(mt));
    }
}

void InitCPUVector(dtype *vec, int dim) {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<> dist(-10, 10);
    for (int i = 0; i < dim; ++i) {
        vec[i] = dist(mt);
    }
}

dtype *NewGPUVector(int dim) {
    dtype *v;
    assert(cnmemMalloc((void**)&v, sizeof(dtype) * dim, NULL) ==
            CNMEM_STATUS_SUCCESS);
    InitGPUVector(v, dim);
    return v;
}

dtype *NewCPUVector(int dim) {
    dtype *v = (dtype*)malloc(sizeof(dtype) * dim);
    InitCPUVector(v, dim);
    return v;
}

void CUBLASAdd(cublasHandle_t handle, dtype *a, dtype *b, int dim) {
    static dtype alpha = 1.0;
    CALL_CUBLAS(axpy)(handle, dim, &alpha, a, 1, b, 1);
}
