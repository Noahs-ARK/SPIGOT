//
// Created by hpeng on 8/28/17.
//

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "SstInstance.h"
#include "SstInstanceNumeric.h"
#include "AlgUtils.h"
#include "SemanticInstance.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticPart.h"

class Classifier : public BiLSTM {

private:

	unsigned NUM_CLASS;
	string FEATURE;
	bool UPDATE_PARSER;
	unsigned ROLE_SIZE = 40;

public:

	explicit Classifier(SemanticOptions *semantic_options, int num_class,
	                    ParameterCollection *model) :
			BiLSTM(semantic_options->num_lstm_layers("classification"),
			       semantic_options->word_dim("classification")
			        + semantic_options->lemma_dim()
			        + semantic_options->pos_dim(),
			       semantic_options->lstm_dim("classification"), model) {
		WORD_DIM = semantic_options->word_dim("classification");
		MLP_DIM = semantic_options->mlp_dim("classification");
		LEMMA_DIM = semantic_options->lemma_dim();
		POS_DIM = semantic_options->pos_dim();
		LSTM_DIM = semantic_options->lstm_dim("classification");
		NUM_CLASS = num_class;
		DROPOUT = semantic_options->dropout("classification");
		WORD_DROPOUT = semantic_options->word_dropout("classification");
		CHECK(DROPOUT >= 0.0 && DROPOUT < 1.0);
		CHECK(WORD_DROPOUT >= 0.0);
		FEATURE = semantic_options->feature();
		UPDATE_PARSER = semantic_options->update_parser();
	}

	void InitParams(ParameterCollection *model);

	void StartGraph(ComputationGraph &cg);

	Expression HeadWordRole(Instance *instance, Parts *parts,
	                    const Expression &y_pred,
	                    const vector<Expression> &ex_lstm,
	                    vector<Expression> &ex_feature,
	                    bool is_train, ComputationGraph &cg);

	void LSTM(Instance *instance, LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
	          vector<Expression> &ex_lstm, unordered_map<int, int> *form_count,
	          bool is_train, ComputationGraph &cg);

	Expression Sst(Instance *instance, Parts *parts,
	               const Expression y_pred,
	               int gold_label, int &predicted_label,
	               unordered_map<int, int> *form_count,
	               bool is_train, ComputationGraph &cg);
};

#endif //CLASSIFIER_H
