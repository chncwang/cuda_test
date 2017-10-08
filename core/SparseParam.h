/*
 * SparseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef SPARSEPARAM_H_
#define SPARSEPARAM_H_

#include "BaseParam.h"

// Notice: aux_square is an aux_squareiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
class SparseParam : public BaseParam {
  public:
#if USE_GPU
	  gpu_matrix aux_square, aux_mean;
#else
	  cpu_matrix aux_square, aux_mean;
#endif
    NRVec<bool> indexers;
	vector<int> indices;
    NRVec<int> last_update;


    // allow sparse and dense parameters have different parameter initialization methods
    inline void initial(int outDim, int inDim) {
        //not in the aligned memory pool
        val.init(outDim, inDim);
        grad.init(outDim, inDim);
        aux_square.init(outDim, inDim);
        aux_mean.init(outDim, inDim);
        indexers.resize(inDim);
        indexers = false;
        last_update.resize(inDim);
        last_update = 0;
		dtype bound = sqrt(3.0 / (outDim));
#if USE_GPU
		cval.init(outDim, inDim);
		cgrad.init(outDim, inDim);
		cval.random(bound);
		dump2gpu();
#else
		val.random(bound);
#endif
    }

    inline void clearGrad() {
        
#if USE_GPU
		grad.zeros();
#else 
		int inDim = indexers.size();
		for (int index = 0; index < inDim; index++) {
			if (!indexers[index]) continue;
			grad.zeros(index);
		   /* for (int idx = 0; idx < grad.row; idx++) {
				grad[index][idx] = 0;
			}*/
		}
		indexers = false;
#endif
    }

    inline int outDim() {
        return val.row;
    }

    inline int inDim() {
        return val.col;
    }

    inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
       
#if USE_GPU
		grad.special_add(grad, 1, val, reg);
		aux_square.special_add1(aux_square, grad);
		val.special_add2(val, grad, aux_square, alpha, eps);
#else
		int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
			grad.special_add(index, grad, 1, val, reg);
			aux_square.special_add1(index, aux_square, grad);
			val.special_add2(index, val, grad, aux_square, alpha, eps);
		}
           // for (int idx = 0; idx < grad.row; idx++) {
                // grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
                // aux_square[index][idx] = aux_square[index][idx] + grad[index][idx] * grad[index][idx];
                // val[index][idx] = val[index][idx] - grad[index][idx] * alpha / sqrt(aux_square[index][idx] + eps);
            // }
#endif
			// for(int index = 0; index < indexers.size(); index++){
				// grad.special_add(indexers[index], grad, 1, val, reg);
				// aux_square.special_add1(indexers[index], aux_square, grad);
				// val.special_add2(indexers[index], val, grad, aux_square, alpha, eps);
			// }
    }

 /*   inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
        dtype lr_t;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < grad.row; idx++) {
                grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
                aux_mean[index][idx] = belta1 * aux_mean[index][idx] + (1 - belta1) * grad[index][idx];
                aux_square[index][idx] = belta2 * aux_square[index][idx] + (1 - belta2) * grad[index][idx] * grad[index][idx];
                lr_t = alpha * sqrt(1 - pow(belta2, last_update[index] + 1)) / (1 - pow(belta1, last_update[index] + 1));
                val[index][idx] = val[index][idx] - aux_mean[index][idx] * lr_t / sqrt(aux_square[index][idx] + eps);
            }
            last_update[index]++;
        }
    }*/

    inline void randpoint(int& idx, int &idy) {
        //select indexes randomly
		indexers.resize(val.col);
        indexers = false;
		for(int i=0; i<indices.size(); i++){
			indexers[indices[i]] = true;
		}
        std::vector<int> idRows, idCols;
        idRows.clear();
        idCols.clear();
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            idCols.push_back(index);
        }
		// for(int index = 0; index < indexers.size(); index++) {
			// idCols.push_back(indexers[index]);
		// }

        for (int i = 0; i < val.row; i++) {
            idRows.push_back(i);
        }

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idx = idCols[0];
        idy = idRows[0];
    }

    inline dtype squareGradNorm() {
        dtype sumNorm = 0.0;
#if USE_GPU
		sumNorm = grad.square_sum();
     
#else
		int inDim = indexers.size();
		for (int index = 0; index < inDim; index++) {
			if (!indexers[index]) continue;
			sumNorm += grad.square_sum(index);
	   /* for (int idx = 0; idx < val.row; idx++) {
			sumNorm += grad[index][idx] * grad[index][idx];
		}*/
		}
#endif

		return sumNorm;
	}

    inline void rescaleGrad(dtype scale) {
#if USE_GPU
		grad.multiply(grad, scale);
		// for(int index = 0; index < indexers.size(); index++) {
			// grad.multiply(grad, indexers[index], scale);
		// }
#else
		int inDim = indexers.size();
		for (int index = 0; index < inDim; index++) {
			if (!indexers[index]) continue;
			grad.multiply(grad, index, scale);
		   /* for (int idx = 0; idx < val.row; idx++) {
				grad[index][idx] = grad[index][idx] * scale;
			}*/
		}
#endif
    }

	template<class Matrix>
    inline void value(const int& featId, Matrix& out) {
        if (out.size != val.row) {
            std::cout << "warning: output dim not equal lookup param dim." << std::endl;
        }
        /*for (int idx = 0; idx < val.row; idx++) {
            out[idx] = val[featId][idx];
        }*/
		out.lookup(val, featId);
    }

 /*   inline void value(const vector<int>& featIds, Tensor1D& out) {
        if (out.dim != val.row) {
            std::cout << "warning: output dim not equal lookup param dim." << std::endl;
        }
        int featNum = featIds.size();
        int featId;
        for (int i = 0; i < featNum; i++) {
            featId = featIds[i];
            for (int idx = 0; idx < val.row; idx++) {
                out[idx] += val[featId][idx];
            }
        }
    }*/

	template<class Matrix>
    inline void loss(const int& featId, const Matrix& loss) {
        if (loss.size != val.row) {
            std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
        }
        indexers[featId] = true;
		
		grad.mat_add_vec(grad, featId, loss);
		
        /*for (int idx = 0; idx < val.row; idx++) {
            grad[featId][idx] += loss[idx];
        }*/
    }

  /*  inline void loss(const vector<int>& featIds, const Tensor1D& loss) {
        if (loss.dim != val.row) {
            std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
        }
        int featNum = featIds.size();
        int featId;
        for (int i = 0; i < featNum; i++) {
            featId = featIds[i];
            indexers[featId] = true;
            for (int idx = 0; idx < val.row; idx++) {
                grad[featId][idx] += loss[idx];
            }
        }
    }*/

    inline void save(std::ofstream &os)const {
        val.save(os);
        aux_square.save(os);
        aux_mean.save(os);
        os << val.col << std::endl;
        for (int idx = 0; idx < val.col; idx++) {
            os << last_update[idx] << std::endl;
        }
    }

    inline void load(std::ifstream &is) {
        val.load(is);
        aux_square.load(is);
        aux_mean.load(is);
        int curInDim;
        is >> curInDim;
        last_update.resize(curInDim);
        for (int idx = 0; idx < curInDim; idx++) {
            is >> last_update[idx];
        }
    }

};

#endif /* SPARSEPARAM_H_ */
