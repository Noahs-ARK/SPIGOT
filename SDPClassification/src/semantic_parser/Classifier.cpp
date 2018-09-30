//
// Created by hpeng on 8/28/17.
//
#include "Classifier.h"

void Classifier::InitParams(ParameterCollection *model) {

	BiLSTM::InitParams(model);
	unsigned mlp_w1_dim;
	if (FEATURE == "average")
		mlp_w1_dim = 2 * LSTM_DIM;
	else if (FEATURE == "headword")
		mlp_w1_dim = MLP_DIM;
	else
		CHECK(false) << "should be either headword or averate";
	params_ = {
			// classifier
			{"w_in_self_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"w_in_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"b_in_", model->add_parameters({MLP_DIM})},

			{"mlp_w1_", model->add_parameters({MLP_DIM, mlp_w1_dim})},
			{"mlp_b1_", model->add_parameters({MLP_DIM})},
			{"mlp_wout_", model->add_parameters({NUM_CLASS, MLP_DIM})},
			{"mlp_bout_", model->add_parameters({NUM_CLASS})},
	};
	lookup_params_.insert({"embed_role_", model->add_lookup_parameters(ROLE_SIZE, {MLP_DIM})});
	params_.insert({"w_in_role_", model->add_parameters({MLP_DIM, MLP_DIM})});
}

void Classifier::StartGraph(ComputationGraph &cg) {
	cg_params_.clear();
	l2rbuilder_.new_graph(cg);
	r2lbuilder_.new_graph(cg);
	for (auto it = params_.begin(); it != params_.end(); ++it) {
		cg_params_[it->first] = parameter(cg, it->second);
	}
}

void Classifier::LSTM(Instance *instance,
                      LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
                      vector<Expression> &ex_lstm,
                      unordered_map<int, int> *form_count,
                      bool is_train, ComputationGraph &cg) {
	auto sentence = static_cast<SemanticInstanceNumeric *>(instance);
	const int slen = sentence->size();
	const vector<int> words = sentence->GetFormIds();
	const vector<int> lemmas = sentence->GetLemmaIds();
	const vector<int> pos = sentence->GetPosIds();
	l2rbuilder.start_new_sequence(); r2lbuilder.start_new_sequence();

	vector<Expression> ex_words(slen), ex_l2r(slen), ex_r2l(slen);
	ex_lstm.resize(slen);
	for (int i = 0; i < slen; ++i) {
		int word_idx = words[i];
		int lemma_idx = lemmas[i];
		int pos_idx = pos[i];

		Expression x_word;
		if (is_train && form_count->at(word_idx) == 0) {
			x_word = lookup(cg, lookup_params_.at("embed_word_"), word_idx);
		} else {
			x_word = const_lookup(cg, lookup_params_.at("embed_word_"), word_idx);
		}
		Expression x_lemma = lookup(cg, lookup_params_.at("embed_lemma_"), lemma_idx);
		Expression x_pos = lookup(cg,lookup_params_.at("embed_pos_"), pos_idx);

		ex_words[i] = concatenate({x_word, x_lemma, x_pos});
		if (is_train && WORD_DROPOUT > 0) {
			ex_words[i] = dropout(ex_words[i], WORD_DROPOUT);
		}
		ex_l2r[i] = l2rbuilder.add_input(ex_words[i]);
	}
	for (int i = 0; i < slen; ++i) {
		ex_r2l[slen - i - 1] = r2lbuilder.add_input(ex_words[slen - i - 1]);
		ex_lstm[slen - i - 1] = concatenate({ex_l2r[slen - i - 1], ex_r2l[slen - i - 1]});
	}
}

Expression Classifier::HeadWordRole(Instance *instance, Parts *parts,
	                                const Expression &y_pred,
	                                const vector<Expression> &ex_lstm,
	                                vector<Expression> &ex_feature,
	                                bool is_train, ComputationGraph &cg) {
	auto sent = static_cast<SemanticInstanceNumeric *> (instance);
	int slen = sent->size() - 1;
	auto semantic_parts = static_cast<SemanticParts *>(parts);

	int offset_arcs, num_arcs;
	semantic_parts->GetOffsetArc(&offset_arcs, &num_arcs);
	int offset_labeled_arcs, num_labeled_arcs;
	semantic_parts->GetOffsetLabeledArc(&offset_labeled_arcs, &num_labeled_arcs);

	int offset_var_arcs = 0, offset_var_labeled_arcs = num_arcs;

	vector<vector<int> > arcs_from_predicate(slen);
	vector<vector<int> > arcs_to_argument(slen);
	for (int i = 0; i < num_arcs; ++i) {
		int r = i + offset_var_arcs;
		auto arc = static_cast<SemanticPartArc *> ((*parts)[i + offset_arcs]);
		int p = arc->predicate();
		int a = arc->argument();
		arcs_from_predicate[p].push_back(r);
		arcs_to_argument[a].push_back(r);
	}

	Expression w_in_self = cg_params_.at("w_in_self_");
	Expression w_in_head = cg_params_.at("w_in_head_");
	Expression w_in_role = cg_params_.at("w_in_role_");
	Expression b_in = cg_params_.at("b_in_");

	vector<Expression> ex_selves(slen), ex_heads(slen);
	for (int i = 0;i < slen;++ i) {
		ex_selves[i] = w_in_self * ex_lstm[i];
		ex_heads[i] = w_in_head * ex_lstm[i];
	}

	ex_feature.resize(slen);
	vector<float> constant_y_pred = as_vector(cg.incremental_forward(y_pred));

	for (int i = 0; i < slen; ++i) {
		vector<int> arcs_to_i = arcs_to_argument[i];
		vector<Expression> ex_hs(arcs_to_i.size());
		dynet::real head_sum = 0;

		for (int j = 0; j < arcs_to_i.size(); ++j) {
			int r = arcs_to_i[j];
			auto arc = static_cast<SemanticPartArc *> (
					(*parts)[r - offset_var_arcs + offset_arcs]);
			CHECK_EQ(arc->argument(), i);
			int head_idx = arc->predicate();

			const vector<int> &index_labeled_parts =
					semantic_parts->FindLabeledArcs(arc->predicate(),
					                                arc->argument(),
					                                arc->sense());
			vector<Expression> ex_roles(index_labeled_parts.size());
			for (int k = 0; k < index_labeled_parts.size(); ++k) {
				int la_idx = index_labeled_parts[k];
				int role = static_cast<SemanticPartLabeledArc *>(
						(*parts)[la_idx])->role();
				Expression ex_role = w_in_role *
						lookup(cg, lookup_params_.at("embed_role_"), role);

				int la_var_idx = la_idx - offset_labeled_arcs
				                 + offset_var_labeled_arcs;
				if (UPDATE_PARSER) {
					ex_roles[k] = ex_role * pick(y_pred, la_var_idx);
				} else {
					ex_roles[k] = ex_role * input(cg, constant_y_pred[la_var_idx]);
				}
			}

			head_sum += constant_y_pred[r];
			if (UPDATE_PARSER) {
				ex_hs[j] = ex_heads[head_idx] * pick(y_pred, r);
			} else {
				ex_hs[j] = ex_heads[head_idx] * input(cg, constant_y_pred[r]);
			}
			ex_hs[j] = ex_hs[j] + sum(ex_roles);
		}
		if (arcs_to_i.size() > 0) {
			Expression ex_head = sum(ex_hs);
			if (head_sum > 1) {
				ex_head = ex_head / head_sum;
			}
			ex_feature[i] = rectify(b_in + ex_selves[i] + ex_head);
		} else {
			ex_feature[i] = rectify(b_in + ex_selves[i]);
		}
		if (is_train && DROPOUT > 0) {
			ex_feature[i] = dropout(ex_feature[i], DROPOUT);
		}
	}
}

Expression Classifier::Sst(Instance *instance, Parts *parts,
                           const Expression y_pred,
                           int gold_label, int &predicted_label,
                           unordered_map<int, int> *form_count,
                           bool is_train, ComputationGraph &cg) {
	Expression mlp_w1 = cg_params_.at("mlp_w1_");
	Expression mlp_b1 = cg_params_.at("mlp_b1_");
	Expression mlp_wout = cg_params_.at("mlp_wout_");
	Expression mlp_bout = cg_params_.at("mlp_bout_");

	vector<Expression> ex_words;
	LSTM(instance, l2rbuilder_, r2lbuilder_,
	     ex_words, form_count, is_train, cg);

	Expression ex_sent;
	if (FEATURE == "average") {
		ex_sent = sum(ex_words);
	} else if (FEATURE == "headword") {
		vector<Expression> ex_feature;
		HeadWordRole(instance, parts, y_pred, ex_words,
		             ex_feature, is_train, cg);
		ex_sent = sum(ex_feature);
	} else {
		CHECK(false);
	}

	Expression ex_h1 = rectify(affine_transform({mlp_b1, mlp_w1, ex_sent}));
	if (is_train && DROPOUT > 0) {
		ex_h1 = dropout(ex_h1, DROPOUT);
	}

	Expression ex_out = affine_transform({mlp_bout, mlp_wout, ex_h1});
	auto v = as_vector(cg.incremental_forward(ex_out));
	CHECK_EQ(v.size(), 2);
	int besti = 0;
	float best = v[0];
	for (unsigned i = 1; i < v.size(); ++i) {
		if (v[i] > best) {
			best = v[i];
			besti = i;
		}
	}

	predicted_label = besti;
	Expression loss = pickneglogsoftmax(ex_out, gold_label);
	return loss;
}