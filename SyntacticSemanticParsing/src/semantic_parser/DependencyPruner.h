//
// Created by hpeng on 9/13/17.
//

#ifndef DEPENDENCYPRUNER_H
#define DEPENDENCYPRUNER_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "DependencyDecoder.h"
#include "AlgUtils.h"

class DependencyPruner : public BiLSTM {
private:

	DependencyDecoder *decoder_;

public:

	explicit DependencyPruner(SemanticOptions *semantic_options,
	                          DependencyDecoder *decoder,
	                          ParameterCollection *model) :
			BiLSTM(semantic_options->pruner_num_lstm_layers(),
			       64, semantic_options->pruner_lstm_dim(), model), decoder_(decoder) {
		WORD_DIM = semantic_options->word_dim();
		LEMMA_DIM = semantic_options->lemma_dim();
		POS_DIM = semantic_options->pos_dim();
		LSTM_DIM = semantic_options->pruner_lstm_dim();
		MLP_DIM = semantic_options->pruner_mlp_dim();
		WORD_DIM = 32;
		LEMMA_DIM = 16;
		POS_DIM = 16;
		DROPOUT = 0.0;
		WORD_DROPOUT = semantic_options->word_dropout("dependency");
		CHECK(WORD_DROPOUT >= 0.0);
	}

	void InitParams(ParameterCollection *model);

	Expression BuildGraph(Instance *instance,
	                      Parts *parts, vector<double> *scores,
	                      const vector<double> *gold_outputs,
	                      vector<double> *predicted_outputs,
	                      unordered_map<int, int> *form_count,
	                      bool is_train, ComputationGraph &cg);

	void DecodeMatrixTree(Instance *instance, Parts *parts,
	                      const vector<Expression> &scores,
	                      vector<double> *predicted_output,
	                      Expression &entropy, ComputationGraph &cg);

	void DecodePruner(Instance *instance, Parts *parts,
	                  const vector<double> &scores,
	                  vector<double> *predicted_output);

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
	                         vector<double> *predicted_output,
	                         Expression &entropy, ComputationGraph &cg);
};


#endif //DEPENDENCYPRUNER_H
