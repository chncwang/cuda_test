#ifndef __FUNCTORS__
#define __FUNCTORS__

#if USE_FLOAT
typedef  float dtype;
#else
typedef  double dtype;
#endif


#if USE_GPU
#define DEVICE_FUNC __host__ __device__
#else
#define DEVICE_FUNC
#endif

struct Activate {
	enum FUNC_TYPE { Tanh_type = 0, Sigmoid_type = 1, Relu_type = 2 };
	int type;
	Activate(int t) : type(t) {}
	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		if (type == Tanh_type) {
			return tanh(x);
		}
		if (type == Sigmoid_type) {
			return 1.0 / (1.0 + exp(-x)); ;
		}
		if (type == Relu_type) {
			if (x <= 0) return 0;
			return x;
		}
	}
};

struct dActivate {
	enum DFUNC_TYPE { dTanh_type = 0, dSigmoid_type = 1, dRelu_type = 2 };
	int type;
	dActivate(int t) : type(t) {}
	
	DEVICE_FUNC inline dtype operator()(const dtype& x, const dtype& y) const {
		if (type == dTanh_type) {
			return (1 + y) * (1 - y);
		}
		if (type == dSigmoid_type) {
			return  (1 - y) * y;
		}
		if (type == dRelu_type) {
			if (x <= 0) return 0;
			return 1;
		}
	}
};

struct special_add_func {
	dtype _a;
	dtype _b;
	special_add_func(dtype a, dtype b) : _a(a), _b(b) {}

	DEVICE_FUNC	inline dtype operator()(const dtype &x, const dtype &y) const {
		return x*_a + y*_b;
	}
};

struct Sqrt {

	DEVICE_FUNC	inline dtype operator()(const dtype &x) const {
		return sqrt(x);
	}
};

struct special_add1_func {
	DEVICE_FUNC	inline dtype operator()(const dtype &x, const dtype &y) const {
		return x + y*y;
	}
};

struct special_add2_func {
	dtype _a;
	dtype _b;
	special_add2_func(dtype a, dtype b) : _a(a), _b(b) {}
	DEVICE_FUNC	inline dtype operator()(const dtype &x, const dtype &y) const {
		return x*_a / sqrt(y + _b);
	}
};


struct self_add_c {
	dtype _a;
	self_add_c(dtype a) : _a(a) {}

	DEVICE_FUNC	inline dtype operator()(const dtype &x) const {
		return x + _a;
	}
};

struct multi_c {
	dtype _a;
	multi_c(dtype a) : _a(a) {}

	DEVICE_FUNC	inline dtype operator()(const dtype &x) const {
		return x * _a;
	}
};

struct Assign {
	dtype _a;
	Assign(dtype a) : _a(a) {}

	DEVICE_FUNC	inline dtype operator()(const dtype &x) const {
		return _a;
	}
};

struct Assignab {

	DEVICE_FUNC	inline dtype operator()(const dtype &x) const {
		return x;
	}
};


struct Tanh {
 
	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		return tanh(x);
	}
};

struct dTanh {
	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		return (1 + y) * (1 - y);
	}
};

struct Sigmoid {
 
	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		return 1.0 / (1.0 + exp(-x));
	}
};

struct dSigmoid {

	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		return  (1 - y) * y;
	}
};

struct Relu {

	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		if (x <= 0) return 0;
		return x;
	}
};

struct dRelu {

	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		if (x <= 0) return 0;
		return 1;
	}
};

struct Leaky_relu {

	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		if (x < 0) return (0.1*x);
		return x;
	}
};

struct dLeaky_relu {

	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		if (x < 0) return 0.1;
		return 1;
	}
};

struct Exp {

	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		return exp(x);
	}
};

struct dExp {

	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		return y;
	}
};

struct Square {

	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		return x*x;
	}
};

struct dSquare {

	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		return 2 * x;
	}
};

struct Cube {

	DEVICE_FUNC	inline dtype operator()(const dtype& x) const {
		return x*x*x;
	}
};

struct dCube {

	DEVICE_FUNC	inline dtype operator()(const dtype& x, const dtype& y) const {
		return 3 * x*x;
	}
};



#endif
