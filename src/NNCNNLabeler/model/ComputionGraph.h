#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder {
  public:
    const static int max_sentence_length = 1024;

  public:
    // node instances
    vector<LookupNode> _word_inputs;
    WindowBuilder _word_window;
    vector<UniNode> _hidden;

    AvgPoolNode _pooling;

    LinearNode _neural_output;

  public:
    GraphBuilder() {
    }

    ~GraphBuilder() {
        clear();
    }

  public:
    //allocate enough nodes
    inline void createNodes(int sent_length) {
        _word_inputs.resize(sent_length);
        _word_window.resize(sent_length);
        _hidden.resize(sent_length);

        _pooling.setParam(sent_length);
    }

    inline void clear() {
        _word_inputs.clear();
        _word_window.clear();
        _hidden.clear();
    }

  public:
    inline void initial(ModelParams& model, HyperParams& opts) {
        for (int idx = 0; idx < _word_inputs.size(); idx++) {
            _word_inputs[idx].setParam(&model.words);
            _word_inputs[idx].init(opts.wordDim, opts.dropProb);
            _hidden[idx].setParam(&model.hidden_linear);
            _hidden[idx].init(opts.hiddenSize, opts.dropProb);
        }
        _word_window.init(opts.wordDim, opts.wordContext);
        _pooling.init(opts.hiddenSize, -1);
        _neural_output.setParam(&model.olayer_linear);
        _neural_output.init(opts.labelSize, -1);
    }


  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void forward(Graph* pcg, const Feature& feature) {
        // second step: build graph
        //forward
        int words_num = feature.m_words.size();
        if (words_num > max_sentence_length)
            words_num = max_sentence_length;
        for (int i = 0; i < words_num; i++) {
            _word_inputs[i].forward(pcg, feature.m_words[i]);
        }
        _word_window.forward(pcg, getPNodes(_word_inputs, words_num));

        for (int i = 0; i < words_num; i++) {
            _hidden[i].forward(pcg, &_word_window._outputs[i]);
        }
        _pooling.forward(pcg, getPNodes(_hidden, words_num));
        _neural_output.forward(pcg, &_pooling);
    }
};

#endif /* SRC_ComputionGraph_H_ */
