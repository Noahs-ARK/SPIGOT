//
// Created by hpeng on 11/3/16.
//

#ifndef BILSTM_H
#define BILSTM_H

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "Instance.h"
#include "expr.h"
#include "SemanticInstanceNumeric.h"

using namespace std;
using namespace dynet;
const int UNK_ID = 0;
const unsigned VOCAB_SIZE = 50000;
const unsigned LEMMA_SIZE = 50000;
const unsigned POS_SIZE = 50;

class BiLSTM {

protected:

	unsigned WORD_DIM;
	unsigned LEMMA_DIM;
	unsigned POS_DIM;
	unsigned LSTM_DIM;
	unsigned MLP_DIM;
	float DROPOUT;
	float WORD_DROPOUT;

	LSTMBuilder l2rbuilder_;
	LSTMBuilder r2lbuilder_;

	unordered_map<string, Parameter> params_;
	unordered_map<string, Expression> cg_params_;
	unordered_map<string, LookupParameter> lookup_params_;

public:

	explicit BiLSTM(int num_layers, int input_dim, int lstm_dim,
	                ParameterCollection *model) :
			l2rbuilder_(num_layers, input_dim, lstm_dim, *model),
			r2lbuilder_(num_layers, input_dim, lstm_dim, *model) {}

	void InitParams(ParameterCollection *model);

	void StartGraph(ComputationGraph &cg, bool is_train);

	void LoadEmbedding(unordered_map<int, vector<float> > *Embedding) {
		for (auto it : (*Embedding)) {
			lookup_params_.at("embed_word_").initialize(it.first, it.second);
		}
	}

	void RunLSTM(Instance *instance,
	             LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
	             vector<Expression> &ex_lstm,
	             unordered_map<int, int> *form_count,
	             bool is_train, ComputationGraph &cg);
};

#endif //BILSTM_H