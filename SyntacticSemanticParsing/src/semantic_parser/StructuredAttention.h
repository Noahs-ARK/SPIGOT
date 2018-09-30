//
// Created by hpeng on 1/14/18.
//

#ifndef STRUCTUREDATTENTION_H
#define STRUCTUREDATTENTION_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "DependencyPart.h"
#include "DependencyInstance.h"
#include "DependencyInstanceNumeric.h"
#include "DependencyDecoder.h"
#include "AlgUtils.h"

class StructuredAttention : public BiLSTM {

protected:

	float BIN_BASE;
	float MAX_BIN;

	DependencyDecoder *decoder_;

public:
	explicit StructuredAttention(SemanticOptions *semantic_options,
	                    DependencyDecoder *decoder, ParameterCollection *model) :
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

		DROPOUT = 0.;
		WORD_DROPOUT = semantic_options->word_dropout("dependency");
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

	Expression BuildGraph(Instance *instance, Parts *parts,
	                      vector<double> *scores,
	                      const vector<double> *gold_outputs,
	                      vector<double> *predicted_outputs,
	                      Expression &ex_score, Expression &y_pred,
	                      unordered_map<int, int> *form_count,
	                      bool is_train, bool max_decode, ComputationGraph &cg);

	void RunEisnerInside(int slen,
	                     const vector<DependencyPartArc *> &arcs,
	                     const vector<Expression> &scores,
	                     vector<Expression> &inside_incomplete_spans,
	                     vector<vector<Expression> > &inside_complete_spans,
	                     vector<vector<bool> > &flag_ics,
	                     Expression &log_partition_function,
	                     ComputationGraph &cg);

	void RunEisnerOutside(int slen,
	                      const vector<DependencyPartArc *> &arcs,
	                      const vector<Expression> &scores,
	                      const vector<Expression> &inside_incomplete_spans,
	                      const vector<vector<Expression> > &inside_complete_spans,
	                      const vector<vector<bool> > &flag_ics,
	                      vector<Expression> &outside_incomplete_spans,
	                      vector<vector<Expression> > &outside_complete_spans,
	                      ComputationGraph &cg);

	void DecodeInsideOutside(Instance *instance, Parts *parts,
	                         const vector<Expression> &scores,
	                         vector<Expression> &predicted_output,
	                         Expression &entropy, ComputationGraph &cg);
};

#endif //STRUCTUREDATTENTION_H
