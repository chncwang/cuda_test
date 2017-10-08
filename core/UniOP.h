#ifndef UNIOP_H_
#define UNIOP_H_

/*
*  UniOP.h:
*  a simple feed forward neural operation, unary input.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class UniParams {
  public:
    Param W;
    Param b;
    bool bUseB;

  public:
    UniParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize, bool useB = true) {
        W.initial(nOSize, nISize);

        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};


inline dtype ftanh(const dtype& x) {
	return tanh(x);
}


inline dtype dtanh(const dtype& x, const dtype& y) {
	return (1 + y) * (1 - y);
}
// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class UniNode : public Node {
  public:
    PNode in;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
	int activate_function;
	int derivate_function;

  public:
    UniNode() : Node() {
        in = NULL;
        activate = ftanh;
        derivate = dtanh;
		activate_function = 0;
		derivate_function = 0;
        param = NULL;
        node_type = "uni";
    }

    ~UniNode() {
        in = NULL;
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }

    // define the activate function and its derivation form
    inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
	template<class Matrix>
    inline void compute(Matrix& ty) {
		ty.product(1, 0, false, false, param->W.val, in->val);
		if (param->bUseB) {
			ty.add(ty, param->b.val);
		}
		val.activate(ty, activate_function);
        /*ty.mat() = param->W.val.mat() * in->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));*/
    }

	template<class Matrix>
    inline void backward(Matrix& ty, Matrix& lty) {
		Matrix mt;
		mt.init(val.row, val.col);
		mt.dactivate(ty, val, derivate_function);
		lty.multiply(loss, mt);
		param->W.grad.product(1, 1, false, true, lty, in->val);
		if (param->bUseB) {
			param->b.grad.add(param->b.grad, lty);
		}
		in->loss.product(1, 1, true, false, param->W.val, lty);
       /* lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));
        param->W.grad.mat() += lty.mat() * in->val.tmat();
        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }
        in->loss.mat() += param->W.val.mat().transpose() * lty.mat();*/
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        UniNode* conv_other = (UniNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }

        return true;
    }

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
//class LinearUniNode : public Node {
//  public:
//    PNode in;
//    UniParams* param;
//
//  public:
//    LinearUniNode() : Node() {
//        in = NULL;
//        param = NULL;
//        node_type = "linear_uni";
//    }
//
//
//    inline void setParam(UniParams* paramInit) {
//        param = paramInit;
//    }
//
//    inline void clearValue() {
//        Node::clearValue();
//        in = NULL;
//    }
//
//
//  public:
//    void forward(Graph *cg, PNode x) {
//        in = x;
//        degree = 0;
//        in->addParent(this);
//        cg->addNode(this);
//    }
//
//  public:
//    inline void compute() {
//        val.mat() = param->W.val.mat() * in->val.mat();
//        if (param->bUseB) {
//            val.vec() += param->b.val.vec();
//        }
//    }
//
//    inline void backward() {
//        param->W.grad.mat() += loss.mat() * in->val.tmat();
//        if (param->bUseB) {
//            param->b.grad.vec() += loss.vec();
//        }
//        in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
//    }
//
//  public:
//    inline PExecute generate(bool bTrain);
//
//    // better to rewrite for deep understanding
//    inline bool typeEqual(PNode other) {
//        bool result = Node::typeEqual(other);
//        if (!result) return false;
//
//        LinearUniNode* conv_other = (LinearUniNode*)other;
//        if (param != conv_other->param) {
//            return false;
//        }
//
//        return true;
//    }
//
//};



// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearNode : public Node {
  public:
    PNode in;
    UniParams* param;

  public:
    LinearNode() : Node() {
        in = NULL;
        param = NULL;
        node_type = "linear";
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }


  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
		val.product(1, 0, false, false, param->W.val, in->val);
       /* val.mat() = param->W.val.mat() * in->val.mat();*/
    }

    inline void backward() {
		param->W.grad.product(1, 1, false, true, loss, in->val);
		in->loss.product(1, 1, true, false, param->W.val, loss);
        /*param->W.grad.mat() += loss.mat() * in->val.tmat();
        in->loss.mat() += param->W.val.mat().transpose() * loss.mat();*/
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        LinearNode* conv_other = (LinearNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


class UniExecute :public Execute {
  public:
// #if USE_GPU
	// vector<gpu_matrix> ty; 
// #else
	// vector<cpu_matrix> ty;
// #endif

#if USE_GPU
		gpu_matrix x, ty, y, b;
#else
		cpu_matrix x, ty, y, b;
#endif
    int inDim, outDim;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
	int activate_function;
	int derivate_function;
    bool bTrain;

  public:
    // inline void  forward() {
        // int count = batch.size();
		// ty.resize(count);
		// for (int idx = 0; idx < count; idx++) {
			// UniNode* ptr = (UniNode*)batch[idx];
			// ty[idx].resize(outDim, 1);
			// ptr->compute(ty[idx]);
			// ptr->forward_drop(bTrain);
		// }
    // }

	inline void  forward() {
		//  *** time test ***
		ofstream out("time", ios::app);
	    auto start = std::chrono::high_resolution_clock::now();
#if USE_GPU
		int count = batch.size();
		x.init(inDim, count);
		b.init(outDim, count);
		ty.init(outDim, count);
		y.init(outDim, count);

		// cout << "Uni" << endl;
	
		//out << "in" << inDim << " out" << outDim << " count" << count << std::endl;*/
		
		vector<gpu_matrix*> x_vec(count);
		vector<gpu_matrix*> b_vec(count);
		
	
		for (int idx = 0; idx < count; idx++) {
			
			UniNode* ptr = (UniNode*)batch[idx];
			x_vec[idx] = &(ptr->in->val);
		
			
			 // UniNode* ptr = (UniNode*)batch[idx];
			 // x.mat_copy_vec(idx, ptr->in->val);
			
			
			/* for (int idy = 0; idy < inDim; idy++) {
			x[idx][idy] = ptr->in->val[idy];
			}*/
			
			if (param->bUseB) {
				b_vec[idx] = &(param->b.val);
				
				
				// b.mat_copy_vec(idx, param->b.val);
				
				
				/*for (int idy = 0; idy < outDim; idy++) {
				b[idx][idy] = param->b.val.v[idy];
				}*/
			}
		}
		x.mat_combine_from_vecs(x_vec);
		if(param->bUseB) {
			b.mat_combine_from_vecs(b_vec);
		}
		
		
		ty.product(1, 0, false, false, param->W.val, x);
		/*ty.mat() = param->W.val.mat() * x.mat();*/

		if (param->bUseB) {
			ty.add(ty, b);
			/*ty.vec() = ty.vec() + b.vec();*/
		}

		y.activate(ty, activate_function);
		/*y.vec() = ty.vec().unaryExpr(ptr_fun(activate));*/
		
		vector<gpu_matrix*> val_vec(count);
		for (int idx = 0; idx < count; idx++) {
			UniNode* ptr = (UniNode*)batch[idx];
			val_vec[idx] = &(ptr->val);
			// ptr->val.vec_copy_mat(y, idx);
			
			/* for (int idy = 0; idy < outDim; idy++) {
			ptr->val[idy] = y[idx][idy];
			}*/
			// ptr->forward_drop(bTrain);
		}
		
		y.vec_separate_from_mat(val_vec);
		
		
		
		for (int idx = 0; idx < count; idx++) {
			UniNode* ptr = (UniNode*)batch[idx];
			ptr->forward_drop(bTrain);
		}
#else
		int count = batch.size();
        x.init(inDim, count);
        b.init(outDim, count);
        ty.init(outDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			x.mat_copy_vec(idx, ptr->in->val);
            if (param->bUseB) {
				b.mat_copy_vec(idx, param->b.val);
            }
        }

		ty.product(1, 0, false, false, param->W.val, x);

        if (param->bUseB) {
			ty.self_add(b);
        }

		y.activate(ty, activate_function);

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			ptr->val.vec_copy_mat(y, idx);
            ptr->forward_drop(bTrain);
        }
#endif	
		auto end = std::chrono::high_resolution_clock::now();
		out << "uni-forward " << std::chrono::duration<double>(end - start).count() << endl; 
	}


	// inline void backward() {
		// int count = batch.size();
		// for (int idx = 0; idx < count; idx++) {
			// UniNode *ptr = (UniNode*)batch[idx];
			// ptr->backward_drop();
// #if USE_GPU
			// gpu_matrix lty;
// #else
			// cpu_matrix lty;
// #endif
			// lty.resize(outDim, 1);
			// ptr->backward(ty[idx], lty);
		// }

	// }
    inline void backward() {
		ofstream out("time", ios::app);
	    auto start = std::chrono::high_resolution_clock::now();
#if USE_GPU
        int count = batch.size();
		gpu_matrix lx, lty, ly;
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

		vector<gpu_matrix*> ly_vec(count);

		
			
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
          /*  for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }*/
			
			
			//ly.mat_copy_vec(idx, ptr->loss);
			
			ly_vec[idx] = &(ptr->loss);
        }
		ly.mat_combine_from_vecs(ly_vec);


		gpu_matrix tmp;
		tmp.init(outDim, count);
		tmp.dactivate(ty, y, derivate_function);
		lty.multiply(ly, tmp);
        /*lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));*/

		param->W.grad.product(1, 1, false, true, lty, x);
        /*param->W.grad.mat() += lty.mat() * x.mat().transpose();*/

        if (param->bUseB) {
            // for (int idx = 0; idx < count; idx++) {
				// param->b.grad.vec_add_mat(param->b.grad, lty, idx);
                // /*for (int idy = 0; idy < outDim; idy++) {
                    // param->b.grad.v[idy] += lty[idx][idy];
                // }*/
            // }
			lty.vec_accumulate_from_mat(&(param->b.grad));
        }

		lx.product(1, 1, true, false, param->W.val, lty);
        /*lx.mat() += param->W.val.mat().transpose() * lty.mat();*/

		vector<gpu_matrix*> loss_vec(count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			loss_vec[idx] = &(ptr->in->loss);
			// ptr->in->loss.vec_add_mat(ptr->in->loss, lx, idx);
           /* for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idx][idy];
            }*/
        }
		lx.vec_accumulate_from_mat(loss_vec);
    
#else 
		int count = batch.size();
        cpu_matrix lx, lty, ly;
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

		
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
			ly.mat_copy_vec(idx, ptr->loss);
          /*  for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss.v[idy];
            }*/
        }

		/*lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));*/
		cpu_matrix ctmp;
		ctmp.init(outDim, count);
		ctmp.dactivate(ty, y, derivate_function);
		lty.multiply(ly, ctmp);
        
		param->W.grad.product(1, 1, false, true, lty, x);
       /* param->W.grad.mat() += lty.mat() * x.mat().transpose();*/

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
				param->b.grad.vec_add_mat(param->b.grad, lty, idx);
               /* for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }*/
            }
        }

		lx.product(1, 1, true, false, param->W.val, lty);
       /* lx.mat() += param->W.val.mat().transpose() * lty.mat();*/

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			ptr->in->loss.vec_add_mat(ptr->in->loss, lx, idx);
            /*for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss.v[idy] += lx[idx][idy];
            }*/
        }
#endif
		auto end = std::chrono::high_resolution_clock::now();
		out << "uni-backward " << std::chrono::duration<double>(end - start).count() << endl; 
	}

};


class LinearExecute :public Execute {
public:
#if USE_GPU
	gpu_matrix x, y;
#else
	cpu_matrix x, y;
#endif
	int inDim, outDim, count;
	UniParams* param;
	bool bTrain;

public:
	// inline void  forward() {
		// count = batch.size();
		// for (int idx = 0; idx < count; idx++) {
			// LinearNode* ptr = (LinearNode*)batch[idx];
			// ptr->compute();
			// ptr->forward_drop(bTrain);
		// }
	// }
	  inline void  forward() {
		  // std::cout << "linear" << endl;
		  ofstream out("time", ios::app);
		  auto start = std::chrono::high_resolution_clock::now();
#if USE_GPU
	      count = batch.size();
	      x.init(inDim, count);
	      y.init(outDim, count);

		  vector<gpu_matrix*> x_vec(count);
		  // ofstream out("linear", ios::app);
		  //out << "in" << inDim << " out" << outDim << " count" << count << std::endl;*/
	      for (int idx = 0; idx < count; idx++) {
			  LinearNode* ptr = (LinearNode*)batch[idx]; 
			  x_vec[idx] = &(ptr->in->val);
			  // ptr->in->val.save(out);
	           // LinearNode* ptr = (LinearNode*)batch[idx]; 
			   // x.mat_copy_vec(idx, ptr->in->val);
			  
			  
	         /* for (int idy = 0; idy < inDim; idy++) {
	              x[idx][idy] = ptr->in->val[idy];
	          }*/
	      }
		  x.mat_combine_from_vecs(x_vec);

		  y.product(1, 0, false, false, param->W.val, x);
	     /* y.mat() = param->W.val.mat() * x.mat();*/

		 vector<gpu_matrix*> val_vec(count);
		 for (int idx = 0; idx < count; idx++) {
			LinearNode* ptr = (LinearNode*)batch[idx];
			val_vec[idx] = &(ptr->val);
		 }
		 
		 y.vec_separate_from_mat(val_vec);
		 
	      for (int idx = 0; idx < count; idx++) {
	          LinearNode* ptr = (LinearNode*)batch[idx];
		  	 //ptr->val.vec_copy_mat(y, idx);
	         /* for (int idy = 0; idy < outDim; idy++) {
	              ptr->val[idy] = y[idx][idy];
	          }*/
			  
	          ptr->forward_drop(bTrain);
	      }
#else
	    count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
			x.mat_copy_vec(idx, ptr->in->val);
           /* for (int idy = 0; idy < inDim; idy++) {
                x[idx][idy] = ptr->in->val.v[idy];
            }*/
        }

		y.product(1, 0, false, false, param->W.val, x);
        /*y.mat() = param->W.val.mat() * x.mat();*/

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
			ptr->val.vec_copy_mat(y, idx);
            /*for (int idy = 0; idy < outDim; idy++) {
                ptr->val.v[idy] = y[idx][idy];
            }*/
            ptr->forward_drop(bTrain);
        }
#endif
		auto end = std::chrono::high_resolution_clock::now();
		out << "linear-forward " << std::chrono::duration<double>(end - start).count() << endl; 
	  }



	// inline void backward() {
		// for (int idx = 0; idx < count; idx++) {
			// LinearNode *ptr = (LinearNode*)batch[idx];
			// ptr->backward_drop();
			// ptr->backward();
		// }
	// }

    inline void backward() {
		ofstream out("time", ios::app);
		auto start = std::chrono::high_resolution_clock::now();
#if USE_GPU
		gpu_matrix lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

		vector<gpu_matrix*> ly_vec(count);
		//vector<gpu_matrix*> ly_vec(count);
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            ptr->backward_drop();
			
			ly_vec[idx] = &(ptr->loss);
			//ly.mat_copy_vec(idx, ptr->loss);
			
           /* for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }*/
			//ly_vec[idx] = &(ptr->loss);
        }
		
		ly.mat_combine_from_vecs(ly_vec);

		param->W.grad.product(1, 1, false, true, ly, x);
        /*param->W.grad.mat() += ly.mat() * x.mat().transpose();*/

		lx.product(1, 1, true, false, param->W.val, ly);
        /*lx.mat() += param->W.val.mat().transpose() * ly.mat();*/

		vector<gpu_matrix*> loss_vec(count);
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
			loss_vec[idx] = &(ptr->in->loss);
			//ptr->in->loss.vec_add_mat(ptr->in->loss, lx, idx);
            /*for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idx][idy];
            }*/
        }
		lx.vec_accumulate_from_mat(loss_vec);
#else
		cpu_matrix lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

		
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            ptr->backward_drop();
			
			ly.mat_copy_vec(idx, ptr->loss);
          /*  for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss.v[idy];
            }*/
        }

		param->W.grad.product(1, 1, false, true, ly, x);
      /*  param->W.grad.mat() += ly.mat() * x.mat().transpose();*/

		lx.product(1, 1, true, false, param->W.val, ly);
        /*lx.mat() += param->W.val.mat().transpose() * ly.mat();*/

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
			ptr->in->loss.vec_add_mat(ptr->in->loss, lx, idx);
			
			//ptr->in->loss.save(out);
           /* for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss.v[idy] += lx[idx][idy];
            }*/
        }
#endif
		auto end = std::chrono::high_resolution_clock::now();
		out << endl;
		out << "linear-backward " << std::chrono::duration<double>(end - start).count() << endl; 
     }
};


inline PExecute UniNode::generate(bool bTrain) {
    UniExecute* exec = new UniExecute();
    exec->batch.push_back(this);
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;

	exec->activate_function = activate_function;
	exec->derivate_function = derivate_function;

    exec->bTrain = bTrain;
    return exec;
}


//inline PExecute LinearUniNode::generate(bool bTrain) {
//    LinearUniExecute* exec = new LinearUniExecute();
//    exec->batch.push_back(this);
//    exec->inDim = param->W.inDim();
//    exec->outDim = param->W.outDim();
//    exec->param = param;
//    exec->bTrain = bTrain;
//    return exec;
//}

inline PExecute LinearNode::generate(bool bTrain) {
    LinearExecute* exec = new LinearExecute();
    exec->batch.push_back(this);
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->bTrain = bTrain;
    return exec;
}


#endif /* UNIOP_H_ */
