//
// Created by hpeng on 10/31/17.
//

#ifndef DEPENDENCY_H
#define DEPENDENCY_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "DependencyPart.h"
#include "DependencyInstance.h"
#include "DependencyInstanceNumeric.h"
#include "DependencyPart.h"
#include "DependencyDecoder.h"

class Dependency : public BiLSTM {

protected:
	float BIN_BASE;
	float MAX_BIN;
	bool PROJECT;

	DependencyDecoder *decoder_;

public:
	explicit Dependency(SemanticOptions *semantic_options,
	                    DependencyDecoder *decoder,  ParameterCollection *model) :
			BiLSTM(semantic_options->num_lstm_layers("dependency"),
			       semantic_options->word_dim() +
			       semantic_options->lemma_dim() +
			       semantic_options->pos_dim(),
			       semantic_options->lstm_dim("dependency"), model), decoder_(decoder) {
		WORD_DIM = semantic_options->word_dim();
		LEMMA_DIM = semantic_options->lemma_dim();
		POS_DIM = semantic_options->pos_dim();
		LSTM_DIM = semantic_options->lstm_dim("dependency");
		MLP_DIM = semantic_options->mlp_dim("dependency");

		BIN_BASE = 2.0;
		MAX_BIN = 4;

		DROPOUT = 0.0;
		WORD_DROPOUT = semantic_options->word_dropout("dependency");
		CHECK(DROPOUT >= 0.0 && DROPOUT < 1.0);
		CHECK(WORD_DROPOUT >= 0.0);

		PROJECT = semantic_options->proj();
	}

	float Bin(unsigned x, bool negative) {
		CHECK_GT(x, 0);
		float ret = logf(x) / logf(BIN_BASE);
		ret = min(ret, MAX_BIN);
		if (negative) ret *= -1.0;
		return ret;
	}

	void InitParams(ParameterCollection *model);

	Expression BuildGraph(Instance *instance, Parts *parts, vector<double> *scores,
	                      const vector<double> *gold_outputs, vector<double> *predicted_outputs,
	                      Expression &ex_score, Expression &y_pred, unordered_map<int, int> *form_count,
	                      bool is_train, ComputationGraph &cg);
};


#endif //DEPENDENCY_H
