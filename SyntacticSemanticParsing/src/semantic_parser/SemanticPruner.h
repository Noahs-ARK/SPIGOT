//
// Created by hpeng on 2/1/18.
//

#ifndef SEMANTICPRUNER_H
#define SEMANTICPRUNER_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "SemanticPart.h"
#include "SemanticInstance.h"
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
			       64, semantic_options->pruner_lstm_dim(), model),
			decoder_(decoder) {
		WORD_DIM = semantic_options->word_dim();
		LEMMA_DIM = semantic_options->lemma_dim();
		POS_DIM = semantic_options->pos_dim();
		LSTM_DIM = semantic_options->pruner_lstm_dim();
		MLP_DIM = semantic_options->pruner_mlp_dim();
		WORD_DIM = 32;
		LEMMA_DIM = 16;
		POS_DIM = 16;
		DROPOUT = 0.0;
		WORD_DROPOUT = semantic_options->word_dropout("semantic");
		CHECK(WORD_DROPOUT >= 0.0);
	}

	void InitParams(ParameterCollection *model);

	void StartGraph(ComputationGraph &cg, bool is_train);

	void RunLSTM(Instance *instance,
	             LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
	             vector<Expression> &ex_lstm,
	             unordered_map<int, int> *form_count,
	             bool is_train, ComputationGraph &cg);

	Expression BuildGraph(Instance *instance,
	                      Parts *parts, vector<double> *scores,
	                      const vector<double> *gold_outputs,
	                      vector<double> *predicted_outputs,
	                      unordered_map<int, int> *form_count,
	                      bool is_train, ComputationGraph &cg);

	void DecodeBasicMarginals(Instance *instance, Parts *parts,
	                          const vector<Expression> &scores,
	                          vector<double> *predicted_output,
	                          Expression &entropy, ComputationGraph &cg);
};


#endif //SEMANTICPRUNER_H
