#ifndef POOLING
#define POOLING

/*
 *  Pooling.h:
 *  pool operation, max, min, average and sum pooling
 *
 *  Created on: Apr 22, 2017
 *      Author: mszhang
 */

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "cuda_util.h"
#include <typeinfo>


class PoolNode : public Node {
    public:
#if USE_GPU
        vector<gpu_matrix> masks;
#else
        vector<cpu_matrix> masks;
#endif
        vector<PNode> ins;

    public:
        PoolNode() : Node() {
            ins.clear();
        }

        ~PoolNode() {
            masks.clear();
            ins.clear();
        }

        inline void clearValue() {
            Node::clearValue();
            ins.clear();
        }

        inline void setParam(int maxsize) {
            masks.resize(maxsize);
        }


        inline void init(int ndim, dtype dropout) {
            Node::init(ndim, dropout);
            int count = masks.size();
            for (int idx = 0; idx < count; idx++) {
                masks[idx].init(ndim, 1);
            }
        }

    public:
        void forward(Graph *cg, const vector<PNode>& x) {
            if (x.size() == 0) {
                std::cout << "empty inputs for max|min|sum|avg pooling" << std::endl;
                return;
            }
            int nSize = x.size();
            ins.clear();
            for (int i = 0; i < nSize; i++) {
                if (x[i]->val.size != dim) {
                    std::cout << "input matrixes are not matched" << std::endl;
                    clearValue();
                    return;
                }
                ins.push_back(x[i]);
            }

            degree = 0;
            for (int i = 0; i < nSize; i++) {
                ins[i]->addParent(this);
            }

            cg->addNode(this);
        }

    public:
        inline PExecute generate(bool bTrain);

        // better to rewrite for deep understanding
        inline bool typeEqual(PNode other) {
            return Node::typeEqual(other);
        }

    public:
        virtual void setMask() {}

        virtual void compute() {
            int nSize = ins.size();
            val.zeros();
            setMask();
#if USE_GPU
            gpu_matrix t;
#else
            cpu_matrix t;
#endif
            t.init(val.row, val.col);
            for (int i = 0; i < nSize; ++i) {
                t.multiply(masks[i], ins[i]->val);
                val.add(val, t);
            }
        }

        virtual void backward() {
            int nSize = ins.size();
#if USE_GPU
            gpu_matrix t;
#else
            cpu_matrix t;
#endif
            t.init(val.row, val.col);
            for (int i = 0; i < nSize; i++) {
                t.multiply(loss, masks[i]);
                ins[i]->loss.add(ins[i]->loss, t);
            }
        }

};

class MaxPoolNode : public PoolNode {
    public:
        MaxPoolNode() : PoolNode() {
            node_type = "max-pooling";
        }

    public:
        //Be careful that the row is the dim of input vector, and the col is the number of input vectors
        //Another point is that we change the input vectors directly.
        void setMask() {
            int nSize = ins.size();
            for (int i = 0; i < nSize; ++i) {
                masks[i].zeros();
            }
#if USE_GPU
            vector<gpu_matrix> vals;
#else
            vector<cpu_matrix> vals;
#endif
            vals.resize(nSize);
            for (int i = 0; i < nSize; i++) {
                vals[i].init(ins[i]->val.row, ins[i]->val.col);
                vals[i] = ins[i]->val;
            }
            max_pooling_helper(vals, masks);
        }

};


class MinPoolNode : public PoolNode {
    public:
        MinPoolNode() : PoolNode() {
            node_type = "min-pooling";
        }

    public:
        //Be careful that the row is the dim of input vector, and the col is the number of input vectors
        //Another point is that we change the input vectors directly.
        void setMask() {
            int nSize = ins.size();
            for (int i = 0; i < nSize; ++i) {
                masks[i].zeros();
            }

#if USE_GPU
            vector<gpu_matrix> vals(nSize);
#else
            vector<cpu_matrix> vals(nSize);
#endif
            for (int i = 0; i < nSize; i++) {
                vals[i].init(ins[i]->val.row, ins[i]->val.col);
                vals[i] = ins[i]->val;
            }
            min_pooling_helper(vals, masks);
        }

};



class AvgPoolNode : public PoolNode {
    public:
        AvgPoolNode() : PoolNode() {
            node_type = "avg-pooling";
        }

        void compute() override {
            int nSize = ins.size();
            val.zeros();
#if USE_GPU
            masks[0].assign((dtype)(1.0 / nSize));

            gpu_matrix t;
            assert(val.col == 1);
            assert(val.row > 0);
            t.init(val.row, val.col);
            dtype **vs = (dtype**)malloc(sizeof(dtype*) * nSize);
            for (int i = 0; i < nSize; ++i) {
                vs[i] = ins[i]->val.v;
                t.add(t, ins[i]->val);
            }
            cout << "add by normal:" << endl;
            HostPrintArr((double*)t.v, val.row);

#if USE_FLOAT
            SumGlobalSArray((float**)vs, (float*)t.v, nSize, val.row);
#else
            SumGlobalDArray((double**)vs, (double*)t.v, nSize, val.row);
#endif
            free(vs);
            cout << "add by multiply:" << endl;
            HostPrintArr((double*)t.v, val.row);
            val.multiply(masks[0], t);
            cout << "after mask" << endl;
            HostPrintArr((double*)val.v, val.row);
#else
            setMask();
            cpu_matrix t;
            t.init(val.row, val.col);
            for (int i = 0; i < nSize; ++i) {
                t.multiply(masks[i], ins[i]->val);
                val.add(val, t);
            }
#endif
        }

        void backward() {
            int nSize = ins.size();
#if USE_GPU
            gpu_matrix t;
#else
            cpu_matrix t;
#endif
            t.init(val.row, val.col);
            t.multiply(loss, masks[0]);
            for (int i = 0; i < nSize; i++) {
                ins[i]->loss.add(ins[i]->loss, t);
            }
        }
};


class PoolExecute : public Execute {
    public:
        bool bTrain;
    public:
        inline void  forward() {
            int count = batch.size();
            for (int idx = 0; idx < count; idx++) {
                PoolNode* ptr = (PoolNode*)batch[idx];
                ptr->compute();
                ptr->forward_drop(bTrain);
            }
        }

        inline void backward() {
            int count = batch.size();
            for (int idx = 0; idx < count; idx++) {
                PoolNode* ptr = (PoolNode*)batch[idx];
                ptr->backward_drop();
                ptr->backward();
            }
        }
};

inline PExecute PoolNode::generate(bool bTrain) {
    PoolExecute* exec = new PoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

#endif
