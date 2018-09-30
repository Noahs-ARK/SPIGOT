//
// Created by hpeng on 2/1/18.
//

#ifndef SEMANTICPARSER_H
#define SEMANTICPARSER_H

#include "SemanticDecoder.h"
#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "SemanticPart.h"
#include "SemanticInstanceNumeric.h"
#include "DependencyPart.h"

class SemanticParser : public BiLSTM {

protected:

	float BIN_BASE;
	float MAX_BIN;
	unsigned ROLE_SIZE;

	SemanticDecoder *decoder_;

public:

	explicit SemanticParser(SemanticOptions *semantic_options, int num_roles,
			SemanticDecoder *decoder,  ParameterCollection *model) :
			BiLSTM(semantic_options->num_lstm_layers("semantic"),
			       semantic_options->word_dim() +
			       semantic_options->lemma_dim() +
			       semantic_options->pos_dim(),
			       semantic_options->lstm_dim("semantic"), model),
			decoder_(decoder) {
			WORD_DIM = semantic_options->word_dim();
			LEMMA_DIM = semantic_options->lemma_dim();
			POS_DIM = semantic_options->pos_dim();
			LSTM_DIM = semantic_options->lstm_dim("semantic");
			MLP_DIM = semantic_options->mlp_dim("semantic");
			ROLE_SIZE = num_roles;

			// TODO: temp solution
			BIN_BASE = 2.0;
			MAX_BIN = 4;

			DROPOUT = 0.0;
			WORD_DROPOUT = semantic_options->word_dropout("semantic");
			CHECK(DROPOUT >= 0.0 && DROPOUT < 1.0);
			CHECK(WORD_DROPOUT >= 0.0);
	}

	float Bin(unsigned x, bool negative) {
		CHECK_GT(x, 0);
		float ret = logf(x) / logf(BIN_BASE);
		ret = min(ret, MAX_BIN);
		if (negative) ret *= -1.0;
		return ret;
	}

	void InitParams(ParameterCollection *model);

	void Feature(Instance *instance, Parts *parts, const Expression &y_pred,
	             const vector<Expression> &ex_words,
	             vector<Expression> &ex_preds,
	             vector<Expression> &ex_unlab_preds,
	             vector<Expression> &ex_unlab_args,
	             vector<Expression> &ex_lab_preds,
	             vector<Expression> &ex_lab_args,
	             ComputationGraph &cg);

	void RunLSTM(Instance *instance,
	             LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
	             vector<Expression> &ex_lstm,
	             unordered_map<int, int> *form_count,
	             bool is_train, ComputationGraph &cg);

	Expression BuildGraph(
			Instance *instance,
			Parts *parts,
			Parts *dependency_parts,
			vector<double> *scores,
			const vector<double> *gold_outputs,
			vector<double> *predicted_outputs,
			Expression &y_pred,
			unordered_map<int, int> *form_count,
			bool is_train, ComputationGraph &cg);
};


#endif //SEMANTICPARSER_H
