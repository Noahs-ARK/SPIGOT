//
// Created by hpeng on 7/5/17.
//

#ifndef SEMANTIC_PRUNER_H
#define SEMANTIC_PRUNER_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "SemanticPart.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDecoder.h"
#include "AlgUtils.h"

class SemanticPruner : public BiLSTM {

private:

	SemanticDecoder *decoder_;

public:

	explicit SemanticPruner(SemanticOptions *semantic_options,
	                        SemanticDecoder *decoder,
	                        ParameterCollection *model) :
			BiLSTM(semantic_options->pruner_num_lstm_layers(),
			       64,
			       semantic_options->pruner_lstm_dim(), model),
			decoder_(decoder) {
		WORD_DIM = 32;
		LEMMA_DIM = 16;
		POS_DIM = 16;
		LSTM_DIM = semantic_options->pruner_lstm_dim();
		MLP_DIM = semantic_options->pruner_mlp_dim();
		DROPOUT = 0.0;
		WORD_DROPOUT = semantic_options->word_dropout("parser");
		CHECK(DROPOUT >= 0.0 && DROPOUT < 1.0);
		CHECK(WORD_DROPOUT >= 0.0);
	}

	void InitParams(ParameterCollection *model);

	Expression
	BuildGraph(Instance *instance, Parts *parts, vector<double> *scores,
	           const vector<double> *gold_outputs,
	           vector<double> *predicted_outputs,
	           unordered_map<int, int> *form_count,
	           const bool &is_train, ComputationGraph &cg);

	void DecodeBasicMarginals(Instance *instance, Parts *parts,
	                          const vector<Expression> &scores,
	                          vector<double> *predicted_output,
	                          Expression &entropy, ComputationGraph &cg);
};

#endif //SEMANTIC_PRUNER_H