//
// Created by hpeng on 10/16/17.
//

#ifndef SEMANTICPARSER_H
#define SEMANTICPARSER_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "AlgUtils.h"
#include "SemanticInstance.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDecoder.h"

class SemanticParser : public BiLSTM {

protected:
	unsigned ROLE_SIZE;
	bool PROJECT;
	SemanticDecoder *decoder_;

public:

	explicit SemanticParser(SemanticOptions *semantic_options, int num_roles,
	                        SemanticDecoder *decoder,
	                        ParameterCollection *model) :
			BiLSTM(semantic_options->num_lstm_layers("parser"),
			       semantic_options->word_dim("parser")
			       + semantic_options->lemma_dim()
			       + semantic_options->pos_dim(),
			       semantic_options->lstm_dim("parser"), model),
			decoder_(decoder) {
		WORD_DIM = semantic_options->word_dim("parser");
		LEMMA_DIM = semantic_options->lemma_dim();
		POS_DIM = semantic_options->pos_dim();
		LSTM_DIM = semantic_options->lstm_dim("parser");
		MLP_DIM = semantic_options->mlp_dim("parser");
		ROLE_SIZE = num_roles;
		DROPOUT = semantic_options->dropout("parser");
		WORD_DROPOUT = semantic_options->word_dropout("parser");
		CHECK(DROPOUT >= 0.0 && DROPOUT < 1.0);
		CHECK(WORD_DROPOUT >= 0.0);
		PROJECT = semantic_options->proj();
	}

	void InitParams(ParameterCollection *model);

	Expression BuildGraph(Instance *instance,
	                      Parts *parts, vector<double> *scores,
	                      const vector<double> *gold_outputs,
	                      vector<double> *predicted_outputs,
	                      Expression &ex_score, Expression &y_pred,
	                      unordered_map<int, int> *form_count,
	                      bool is_train, ComputationGraph &cg);
};


#endif //SEMANTICPARSER_H
