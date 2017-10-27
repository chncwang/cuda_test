#ifndef BasicNode
#define BasicNode

/*
*  Node.h:
*  basic processing unit in a neural network
*  (1) we have a node structure to build user graph
*  (2) we have a execute structure to merge similar nodes that can be execute together
*  The real forward and backward are defined in Execute.
*  Every operation should define a node class and a execute class together.
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "cpu_matrix.h"

#if USE_GPU
	#include "gpu_matrix.h"
#endif

class Execute;

// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Node {
public:
	vector<Node*> parents;
public:
	/*Tensor1D val;
	Tensor1D loss*/;
#if USE_GPU
	  gpu_matrix val, loss, mask;
	  cpu_matrix cval, closs, cmask;
#else
	  cpu_matrix val, loss, mask;
#endif

public:
	int dim;
	int degree;
	string node_type;
	dtype drop_value;

public:
	Node() {
		dim = 0;
		degree = 0;
		parents.clear();
		node_type = "interface";
		drop_value = -1;
	}

	virtual ~Node() {
	}

public:
	virtual inline void clearValue() {
		val.zeros();
		loss.zeros();
		degree = 0;
		if (drop_value > 0) mask.ones();
		parents.clear();
	}

	virtual inline void init(int ndim, dtype dropout, bool is_matrix_v_weak_ref = false) {
		dim = ndim;
#if USE_GPU
		val.init(dim, 1, is_matrix_v_weak_ref);
#else
		val.init(dim, 1);
#endif
		loss.init(dim, 1);
		mask.init(dim, 1);
		if (dropout > 0 && dropout <= 1) {
			drop_value = dropout;
		}
		else {
			drop_value = -1;
		}
		parents.clear();
	}


	template<class Matrix>
	inline void dropout(Matrix& val, Matrix& mask, bool bTrain) {
		if (bTrain) {
			int seed = rand();
			val.dropout(mask, drop_value, seed);
			val.multiply(val, mask);
		}
		else {
			val.multiply(val, 1 - drop_value);
		}
	}

	inline void forward_drop(bool bTrain) {
		if (drop_value > 0) {
			dropout(val, mask, bTrain);
			degree = -1;
		}
	}

    inline void backward_drop() {
		if (drop_value > 0) {
			loss.multiply(loss, mask);
		}
    }
  public:

    virtual inline Execute* generate(bool bTrain) = 0;

    virtual inline bool typeEqual(Node* other) {
        if (node_type.compare(other->node_type) == 0) {
            return true;
        }
        return false;
    }

  public:
    virtual inline void addParent(Node* parent) {
        if (degree >= 0) {
            parents.push_back(parent);
            parent->degree++;
        }
    }
};

typedef  Node* PNode;


class Execute {
  public:
    vector<PNode> batch;

  public:
    virtual ~Execute() {
        batch.clear();
    }

  public:
    virtual inline void forward() = 0;
    virtual inline void backward() = 0;


    virtual inline bool addNode(PNode in) {
        if (batch.empty()) {
            std::cout << "empty batch, strange...." << std::endl;
            return false;
        }

        if (batch[0]->typeEqual(in)) {
            batch.push_back(in);
            return true;
        }

        return false;
    }
};


typedef  Execute* PExecute;

#endif
