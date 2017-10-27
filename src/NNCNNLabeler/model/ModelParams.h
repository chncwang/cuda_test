#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {

  public:
    Alphabet wordAlpha; // should be initialized outside
    LookupTable words; // should be initialized outside
    Alphabet featAlpha;
    UniParams hidden_linear;
    UniParams olayer_linear; // output
  public:
    Alphabet labelAlpha; // should be initialized outside
    SoftMaxLoss loss;


  public:
    bool initial(HyperParams& opts) {

        // some model parameters should be initialized outside
        if (words.nVSize <= 0 || labelAlpha.size() <= 0) {
            return false;
        }
        opts.wordDim = words.nDim;
        opts.wordWindow = opts.wordContext * 2 + 1;
        opts.windowOutput = opts.wordDim * opts.wordWindow;
        opts.labelSize = labelAlpha.size();
        hidden_linear.initial(opts.hiddenSize, opts.windowOutput, true);
       /* opts.inputSize = opts.hiddenSize * 3;*/
	    opts.inputSize = opts.hiddenSize;
        olayer_linear.initial(opts.labelSize, opts.inputSize, false);
        return true;
    }

    bool TestInitial(HyperParams& opts) {

        // some model parameters should be initialized outside
        if (words.nVSize <= 0 || labelAlpha.size() <= 0) {
            return false;
        }
        opts.wordDim = words.nDim;
        opts.wordWindow = opts.wordContext * 2 + 1;
        opts.windowOutput = opts.wordDim * opts.wordWindow;
        opts.labelSize = labelAlpha.size();
        opts.inputSize = opts.hiddenSize * 3;
        return true;
    }

    void exportModelParams(ModelUpdate& ada) {
        words.exportAdaParams(ada);
        hidden_linear.exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }


    void exportCheckGradParams(CheckGrad& checkgrad) {
        checkgrad.add(&words.E, "words E");
        checkgrad.add(&hidden_linear.W, "hidden W");
        checkgrad.add(&hidden_linear.b, "hidden b");
        checkgrad.add(&olayer_linear.W, "output layer W");
    }

    // will add it later
    void saveModel(std::ofstream &os) const {
        wordAlpha.write(os);
        words.save(os);
        hidden_linear.save(os);
        olayer_linear.save(os);
        labelAlpha.write(os);
    }

    void loadModel(std::ifstream &is) {
        wordAlpha.read(is);
        words.load(is, &wordAlpha);
        hidden_linear.load(is);
        olayer_linear.load(is);
        labelAlpha.read(is);
    }

};

#endif /* SRC_ModelParams_H_ */