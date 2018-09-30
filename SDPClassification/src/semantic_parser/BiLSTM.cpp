//
// Created by hpeng on 9/12/17.
//

#include "BiLSTM.h"

void BiLSTM::InitParams(ParameterCollection *model) {
	lookup_params_ = {
			{"embed_word_", model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM})},
			{"embed_lemma_", model->add_lookup_parameters(LEMMA_SIZE, {LEMMA_DIM})},
			{"embed_pos_", model->add_lookup_parameters(POS_SIZE, {POS_DIM})}
	};
}

void BiLSTM::StartGraph(ComputationGraph &cg, bool is_train) {
	cg_params_.clear();
	if (DROPOUT > 0 && is_train) {
		l2rbuilder_.set_dropout(DROPOUT);
		r2lbuilder_.set_dropout(DROPOUT);
	} else {
		l2rbuilder_.disable_dropout();
		r2lbuilder_.disable_dropout();
	}
	l2rbuilder_.new_graph(cg);
	r2lbuilder_.new_graph(cg);
	for (auto it = params_.begin(); it != params_.end(); ++it) {
		cg_params_[it->first] = parameter(cg, it->second);
	}
}

void BiLSTM::RunLSTM(Instance *instance,
                     LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
                     vector<Expression> &ex_lstm,
                     unordered_map<int, int> *form_count,
                     bool is_train, ComputationGraph &cg) {
	auto sentence = dynamic_cast<DependencyInstanceNumeric *>(instance);
	const int slen = sentence->size();
	const vector<int> words = sentence->GetFormIds();
	const vector<int> pos = sentence->GetPosIds();
	l2rbuilder.start_new_sequence(); r2lbuilder.start_new_sequence();

	vector<Expression> ex_words(slen), ex_l2r(slen), ex_r2l(slen);
	ex_lstm.resize(slen);
	for (int i = 0; i < slen; ++i) {
		int word_idx = words[i];
		int lemma_idx = words[i];
		int pos_idx = pos[i];
		if (is_train && WORD_DROPOUT > 0.0 && word_idx != UNK_ID) {
			float count = static_cast<float> (form_count->at(word_idx));
			float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			if (rand_float < WORD_DROPOUT / (count + WORD_DROPOUT)) {
				word_idx = UNK_ID;
				lemma_idx = UNK_ID;
				pos_idx = UNK_ID;
			}
		}
		Expression x_word = lookup(cg, lookup_params_.at("embed_word_"), word_idx);
		Expression x_lemma = lookup(cg, lookup_params_.at("embed_lemma_"), lemma_idx);
		Expression x_pos = lookup(cg,lookup_params_.at("embed_pos_"), pos_idx);
		ex_words[i] = concatenate({x_word, x_lemma, x_pos});
		ex_l2r[i] = l2rbuilder.add_input(ex_words[i]);
	}
	for (int i = 0; i < slen; ++i) {
		ex_r2l[slen - i - 1] = r2lbuilder.add_input(ex_words[slen - i - 1]);
		ex_lstm[slen - i - 1] = concatenate({ex_l2r[slen - i - 1], ex_r2l[slen - i - 1]});
	}
}