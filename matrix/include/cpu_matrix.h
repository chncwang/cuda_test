#ifndef _Ccpu_matrix_
#define _Ccpu_matrix_

#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cstring>

#include <iostream>
#include <fstream>
#include <vector>

#if USE_GPU

#include "gpu_matrix.h"

#endif

class gpu_matrix;

#if USE_FLOAT
typedef  float dtype;
#else
typedef  double dtype;
#endif

using namespace std;
using namespace Eigen;
#if USE_FLOAT
typedef Eigen::Map<MatrixXf> Mat;
typedef Eigen::TensorMap<Eigen::Tensor<float, 1>>  Vec;
#else
typedef Eigen::Map<MatrixXd> Mat;
typedef Eigen::TensorMap<Eigen::Tensor<double, 1>>  Vec;
#endif


struct cRand
{
    dtype a, b, dp_val;

    cRand(dtype _a=0.0, dtype _b=1.0, dtype _dp_val=0.0) : a(_a), b(_b), dp_val(_dp_val) {};

	inline dtype operator()(const unsigned int n) const
	{
		dtype t = (dtype(rand()) / RAND_MAX) * (b - a) + a;
		
		if(t <= dp_val)
			return 0;
		else 
			return 1;
	}
};


class cpu_matrix{
public:
	dtype *v;
	int row, col, size;
public:
	const Mat mat() const {
		return Mat(v, row, col);
	}
	Mat mat(){
		return Mat(v, row, col);
	}
	const Vec vec() const {
		return Vec(v, size);
	}
	Vec vec(){
		return Vec(v, size);
	}
public:
	cpu_matrix();
	~cpu_matrix();
	void init(int r, int c);
	cpu_matrix(dtype* v_data, size_t r, size_t c);
	void delloc();
	void resize(int r, int c){
		if(row == r && col == c)
			return;
		
		if(v){
			delete[] v;
		}
		init(r, c);
	}
	inline void zero() { if(v) memset((void*)v, 0, size * sizeof(dtype)); }
	void zeros();
	void ones();
	void random(dtype bound);
	void assign(int icol, int jrow, dtype value){
		v[icol*row +jrow] = value;
	}
	dtype get(int icol, int jrow){
		return v[icol*row + jrow];
	}
	cpu_matrix& operator = (const cpu_matrix &rhs);
	cpu_matrix& operator = (const gpu_matrix &rhs);
	
	/*inline dtype* operator[](const int icol) { 
		if (col == 1 && icol != 0) { 
			std::cout << "err"; 
			return NULL;
		} 
		else {
			return v + icol*row;
		}
	}
	inline const dtype* operator[](const int icol)const{ 
		if (col == 1 && icol != 0) {
			std::cout << "err";
			return NULL;
		}
		else {
			return v + icol*row;
		}
	}*/

	void transpose(const cpu_matrix &rhs);
	// void transpose();
	void add(const cpu_matrix &a, const cpu_matrix &b);
	void add(const cpu_matrix &a, const dtype &scale)
	{
		this->vec() = a.vec() + scale;
	}
	void sub(const cpu_matrix &a, const cpu_matrix &b);
	void multiply(const cpu_matrix &a, const cpu_matrix &b);
	void multiply(const cpu_matrix &a, const dtype scale) {
		this->vec() = a.vec() * scale;
	}
	void divide(const cpu_matrix &a, const cpu_matrix &b);
	void product(const cpu_matrix &a, const cpu_matrix &b);
	void product(dtype alpha, dtype beta, bool aTranspose, bool bTranspose, const cpu_matrix &a, const cpu_matrix &b) {
		if (aTranspose == false && bTranspose == false)
			this->mat() = this->mat() * beta + a.mat() * b.mat() * alpha;
		else if(aTranspose == false && bTranspose == true)
			this->mat() = this->mat() * beta + a.mat() * b.mat().transpose() * alpha;
		else if (aTranspose == true && bTranspose == false)
			this->mat() = this->mat() * beta + a.mat().transpose() * b.mat() * alpha;
		else
			this->mat() = this->mat() * beta + a.mat().transpose() * b.mat().transpose() * alpha;
	}
	//======================================/
	void self_add(const cpu_matrix &rhs);
	void self_sub(const cpu_matrix &rhs);
	void self_multiply(const cpu_matrix &rhs);
	void self_multiply(const dtype scale) {
		this->vec() = this->vec() * scale;
	}
	void self_divide(const cpu_matrix &rhs);
	//======================================/
	void self_square() {
		this->vec() = this->vec().square();
	}
	void self_sqrt() {
		this->vec() = this->vec().sqrt();
	}
	void tanh(const cpu_matrix &rhs);
	void sigmoid(const cpu_matrix &rhs);
	void relu(const cpu_matrix &rhs);
	void leaky_relu(const cpu_matrix &rhs);
	void square(const cpu_matrix &rhs);
	void cube(const cpu_matrix &rhs);
	void dtanh(const cpu_matrix &a, const cpu_matrix &b);
	void dsigmoid(const cpu_matrix &a, const cpu_matrix &b);
	void drelu(const cpu_matrix &a, const cpu_matrix &b);
	void dleaky_relu(const cpu_matrix &a, const cpu_matrix &b);
	void dsquare(const cpu_matrix &a, const cpu_matrix &b);
	void dcube(const cpu_matrix &a, const cpu_matrix &b);
	void activate(const cpu_matrix &rhs, int functor);
	void dactivate(const cpu_matrix &a, const cpu_matrix &b, int functor);
	
	
	void dropout(cpu_matrix &drop_mask, dtype drop_value, int seed){
		srand(seed);
		drop_mask.vec() = drop_mask.vec().unaryExpr(cRand(0, 1, drop_value));
		this->vec() =  this->vec() * drop_mask.vec();
	}
	void assign(dtype scale);
	void lookup(const cpu_matrix &E, int idx){
		assert(E.row == size);
		memcpy(v, E.v+idx*E.row, sizeof(dtype) * size);
	}
	void concat(const vector<cpu_matrix> &rhs_vec){
		assert(col == rhs_vec.size());
		assert(row == rhs_vec[0].size);
		for(int i=0; i<col; i++){
			memcpy(v + i*row, rhs_vec[i].v, sizeof(dtype) * row);
		}
	}
	void display(){
		std::cout << mat() << "\n" << "\n";
	}

	void big_copy_small(int offset, const cpu_matrix &rhs) {  //this[offset:rhs.size] = rhs[0:size]
		Vec(v + offset, rhs.size) = Vec(rhs.v, rhs.size);
	}

	void short_add_long(const cpu_matrix &a, const cpu_matrix &b, int offset) {
		Vec(v, size) = Vec(a.v, a.size) + Vec(b.v + offset, size);
	}

	dtype square_sum() {
		dtype sum = 0.0;
		for (int i = 0; i < size; i++) {
			sum += v[i] * v[i];
		}

		return sum;
	}


	// template<typename CustomUnaryOp>
	// void unary(const cpu_matrix &rhs, const CustomUnaryOp& op){this->vec() = rhs.vec().unaryExpr(op);} 
	// template<typename CustomBinaryOp>
	// void binary(const cpu_matrix &a, const cpu_matrix &b, const CustomBinaryOp& op)
	// {this->vec() = a.vec().binaryExpr(b.vec(), op);}

	void save(std::ostream &os) const {
		os << size << " " << row << " " << col << std::endl;
		os << v[0];
		for (int idx = 1; idx < size; idx++) {
			os << " " << v[idx];
		}
		os << std::endl;
	}

	void load(std::ifstream &is) {
		int curSize, curRow, curCol;
		is >> curSize;
		is >> curRow;
		is >> curCol;
		init(curRow, curCol);
		for (int idx = 0; idx < size; idx++) {
			is >> v[idx];
		}
	}

	void mat_copy_vec(int icol, const cpu_matrix &rhs) {
		assert(row == rhs.size);
		memcpy(v + icol*row, rhs.v, rhs.size * sizeof(dtype));
	}

	void vec_copy_mat(const cpu_matrix &rhs, int icol) {
		assert(size == rhs.row);
		memcpy(v, rhs.v + icol*rhs.row, rhs.row * sizeof(dtype));
	}

	void vec_add_mat(const cpu_matrix &a, const cpu_matrix &b, int icol) {
		assert(a.size == b.row);
		Vec(v, size) = Vec(a.v, a.size) + Vec(b.v + b.row * icol, b.row);
	} 

	void mat_add_vec(const cpu_matrix &a, int icol, const cpu_matrix &b) {
		assert(a.row == b.size);
		Vec(v + icol*row, row) = Vec(a.v + icol * a.row, a.row) + Vec(b.v, b.size);
	}

	void zeros(int icol) {
		memset(v + icol * row, 0, row * sizeof(dtype));
	}


	dtype square_sum(int icol) {
		dtype sum = 0.0;
		for (int idx = icol * row; idx < icol * row + row; idx++) {
			sum += v[idx] * v[idx];
		}

		return sum;
	}

	void multiply(const cpu_matrix &rhs, int icol, dtype scale) {
		Vec(v + icol*row, row) = Vec(rhs.v + icol*rhs.row, rhs.row) * scale;
	}

	void norm2one() {
		dtype sum;
		for (int idx = 0; idx < col; idx++) {
			sum = 0.000001;
			sum = this->square_sum(idx);
			dtype scale = sqrt(sum);
			this->multiply(*this, idx, 1.0/scale);
		}
	}

	void special_add(
		int index,
		const cpu_matrix &a,
		dtype m,
		const cpu_matrix &b,
		dtype n) {
		
		Vec(v+row*index, row) = Vec(a.v + a.row * index, a.row) * m + Vec(b.v + b.row * index, b.row) * n;

	}

	void special_add1(
		int index,
		const cpu_matrix &a,
		const cpu_matrix &b) {
		Vec(v + row*index, row) = Vec(a.v + a.row * index, a.row) + Vec(b.v + b.row * index, b.row).square();
	}

	void special_add2(
		int index,
		const cpu_matrix &a,
		const cpu_matrix &b,
		const cpu_matrix &c,
		dtype  alpha,
		dtype eps) {
		Vec(v + row*index, row) = Vec(a.v + a.row * index, a.row) - Vec(b.v + b.row*index, b.row) * alpha / (Vec(c.v + c.row*index, c.row) + eps).sqrt();
	}


	void special_add(
		const cpu_matrix &a,
		dtype m,
		const cpu_matrix &b,
		dtype n) {
		this->vec() = a.vec() * m + b.vec() * n;
	}

	void special_add1(
		const cpu_matrix &a,
		const cpu_matrix &b) {
		this->vec() = a.vec() + b.vec().square();
	}

	void special_add2(
		const cpu_matrix &a,
		const cpu_matrix &b,
		const cpu_matrix &c,
		dtype  alpha,
		dtype eps) {
		this->vec() = a.vec() - b.vec() * alpha / (c.vec() + eps).sqrt();
	}

	void self_add(int icol, int irow, dtype scale) {
		v[icol*row + irow] += scale;
	}
};

void max_pooling_helper(vector<cpu_matrix> &ins, vector<cpu_matrix> &mask);

void min_pooling_helper(vector<cpu_matrix> &ins, vector<cpu_matrix> &mask);


#endif
