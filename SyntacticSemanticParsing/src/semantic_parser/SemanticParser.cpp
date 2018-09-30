//
// Created by hpeng on 2/1/18.
//

#include <src/util/AlgUtils.h>
#include "SemanticParser.h"

void SemanticParser::InitParams(ParameterCollection *model) {
	// shared
	BiLSTM::InitParams(model);
	// concate head to words after lstm
	params_ = {
			{"pred_w1_self_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"pred_w1_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"pred_b1_", model->add_parameters({MLP_DIM})},
			{"pred_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"pred_b2_", model->add_parameters({MLP_DIM})},
			{"pred_w3_", model->add_parameters({1, MLP_DIM})},
			{"pred_b3_", model->add_parameters({1})},

			{"unlab_w1_pred_self_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w1_pred_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w1_arg_self_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w1_arg_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_b1_", model->add_parameters({MLP_DIM})},
			{"unlab_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"unlab_b2_", model->add_parameters({MLP_DIM})},
			{"unlab_w3_", model->add_parameters({1, MLP_DIM})},
			{"unlab_b3_", model->add_parameters({1})},

			{"lab_w1_pred_self_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"lab_w1_pred_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"lab_w1_arg_self_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"lab_w1_arg_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"lab_b1_", model->add_parameters({MLP_DIM})},
			{"lab_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"lab_b2_", model->add_parameters({MLP_DIM})},
			{"lab_w3_", model->add_parameters({ROLE_SIZE, MLP_DIM})},
			{"lab_b3_", model->add_parameters({ROLE_SIZE})},
	};
}


void SemanticParser::RunLSTM(Instance *instance,
		                     LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
		                     vector<Expression> &ex_lstm,
		                     unordered_map<int, int> *form_count,
		                     bool is_train, ComputationGraph &cg) {
	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	const int slen = sentence->size();
	const vector<int> words = sentence->GetFormIds();
	const vector<int> lemmas = sentence->GetLemmaIds();
	const vector<int> pos = sentence->GetPosIds();

	l2rbuilder.start_new_sequence();
	r2lbuilder.start_new_sequence();

	vector<Expression> ex_words(slen), ex_l2r(slen), ex_r2l(slen);
	ex_lstm.resize(slen);
	for (int i = 0; i < slen; ++i) {
		int word_idx = words[i];
		int lemma_idx = lemmas[i];
		int pos_idx = pos[i];
		if (is_train && WORD_DROPOUT > 0.0 && word_idx != UNK_ID) {
			float count = static_cast<float> (form_count->at(word_idx));
			float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			if (rand_float <
			    WORD_DROPOUT / (count + WORD_DROPOUT)) {
				word_idx = UNK_ID;
			}
		}
		Expression x_word = lookup(cg, lookup_params_.at("embed_word_"), word_idx);
		Expression x_lemma = lookup(cg, lookup_params_.at("embed_lemma_"), lemma_idx);
		Expression x_pos = lookup(cg, lookup_params_.at("embed_pos_"), pos_idx);

		ex_words[i] = concatenate({x_word, x_lemma, x_pos});
		ex_l2r[i] = l2rbuilder.add_input(ex_words[i]);
	}
	for (int i = 0; i < slen; ++i) {
		ex_r2l[slen - i - 1] = r2lbuilder.add_input(ex_words[slen - i - 1]);
		ex_lstm[slen - i - 1] = concatenate(
				{ex_l2r[slen - i - 1], ex_r2l[slen - i - 1]});
	}
}

// concate head after lstm
void
SemanticParser::Feature(Instance *instance, Parts *parts, const Expression &y_pred,
                        const vector<Expression> &ex_words,
                        vector<Expression> &ex_preds,
                        vector<Expression> &ex_unlab_preds,
                        vector<Expression> &ex_unlab_args,
                        vector<Expression> &ex_lab_preds,
                        vector<Expression> &ex_lab_args,
                        ComputationGraph &cg) {
	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	const int slen = sentence->size() - 1;
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	int offset_arcs, num_arcs;
	dependency_parts->GetOffsetArc(&offset_arcs, &num_arcs);
	vector<vector<int> > arcs_to_mod(slen);

	for (int i = 0; i < num_arcs; ++i) {
		int r = i + offset_arcs;
		auto arc = static_cast<DependencyPartArc *> ((*parts)[r]);
		int m = arc->modifier();
		arcs_to_mod[m].push_back(r);
	}
	ex_preds.resize(slen);
	ex_unlab_preds.resize(slen);
	ex_unlab_args.resize(slen);
	ex_lab_preds.resize(slen);
	ex_lab_args.resize(slen);
	vector<float> constant_y_pred
			= as_vector(cg.incremental_forward(y_pred));
	CHECK_EQ(constant_y_pred.size(), num_arcs);

	Expression pred_w1_self = cg_params_.at("pred_w1_self_");
	Expression pred_w1_head = cg_params_.at("pred_w1_head_");

	Expression unlab_w1_pred_self = cg_params_.at("unlab_w1_pred_self_");
	Expression unlab_w1_pred_head = cg_params_.at("unlab_w1_pred_head_");
	Expression unlab_w1_arg_self = cg_params_.at("unlab_w1_arg_self_");
	Expression unlab_w1_arg_head = cg_params_.at("unlab_w1_arg_head_");

	Expression lab_w1_pred_self = cg_params_.at("lab_w1_pred_self_");
	Expression lab_w1_pred_head = cg_params_.at("lab_w1_pred_head_");
	Expression lab_w1_arg_self = cg_params_.at("lab_w1_arg_self_");
	Expression lab_w1_arg_head = cg_params_.at("lab_w1_arg_head_");

	vector<Expression> ex_pred_selves(slen), ex_pred_heads(slen);
	vector<Expression> ex_unlab_pred_selves(slen), ex_unlab_pred_heads(slen);
	vector<Expression> ex_unlab_arg_selves(slen), ex_unlab_arg_heads(slen);
	vector<Expression> ex_lab_pred_selves(slen), ex_lab_pred_heads(slen);
	vector<Expression> ex_lab_arg_selves(slen), ex_lab_arg_heads(slen);

	for (int i = 0;i < slen; ++ i) {
		ex_pred_selves[i] = pred_w1_self * ex_words[i];
		ex_unlab_pred_selves[i] = unlab_w1_pred_self * ex_words[i];
		ex_unlab_arg_selves[i] = unlab_w1_arg_self * ex_words[i];
		ex_lab_pred_selves[i] = lab_w1_pred_self * ex_words[i];
		ex_lab_arg_selves[i] = lab_w1_arg_self * ex_words[i];

		ex_pred_heads[i] = pred_w1_head * ex_words[i];
		ex_unlab_pred_heads[i] = unlab_w1_pred_head * ex_words[i];
		ex_unlab_arg_heads[i] = unlab_w1_arg_head * ex_words[i];
		ex_lab_pred_heads[i] = lab_w1_pred_head * ex_words[i];
		ex_lab_arg_heads[i] = lab_w1_arg_head * ex_words[i];
	}

	for (int i = 0; i < slen; ++i) {
		vector<int> arcs_to_i = arcs_to_mod[i];
		vector<Expression> ex_pred_hs(arcs_to_i.size());
		vector<Expression> ex_unlab_pred_hs(arcs_to_i.size());
		vector<Expression> ex_unlab_arg_hs(arcs_to_i.size());
		vector<Expression> ex_lab_pred_hs(arcs_to_i.size());
		vector<Expression> ex_lab_arg_hs(arcs_to_i.size());

		for (int j = 0; j < arcs_to_i.size(); ++j) {
			int r = arcs_to_i[j];
			auto arc = static_cast<DependencyPartArc *> ((*parts)[r]);
			int head_idx = arc->head();

			ex_pred_hs[j] = ex_pred_heads[head_idx] * pick(y_pred, r);
			ex_unlab_pred_hs[j] = ex_unlab_pred_heads[head_idx] * pick(y_pred, r);
			ex_unlab_arg_hs[j] = ex_unlab_arg_heads[head_idx] * pick(y_pred, r);
			ex_lab_pred_hs[j] = ex_lab_pred_heads[head_idx] * pick(y_pred, r);
			ex_lab_arg_hs[j] = ex_lab_arg_heads[head_idx] * pick(y_pred, r);
		}
		if (arcs_to_i.size() > 0) {
			ex_preds[i] = ex_pred_selves[i] + sum(ex_pred_hs);
			ex_unlab_preds[i] = ex_unlab_pred_selves[i] + sum(ex_unlab_pred_hs);
			ex_unlab_args[i] = ex_unlab_arg_selves[i] + sum(ex_unlab_arg_hs);
			ex_lab_preds[i] = ex_lab_pred_selves[i] + sum(ex_lab_pred_hs);
			ex_lab_args[i] = ex_lab_arg_selves[i] + sum(ex_lab_arg_hs);
		} else {
			ex_preds[i] = ex_pred_selves[i];
			ex_unlab_preds[i] = ex_unlab_pred_selves[i];
			ex_unlab_args[i] = ex_unlab_arg_selves[i];
			ex_lab_preds[i] = ex_lab_pred_selves[i];
			ex_lab_args[i] = ex_lab_arg_selves[i];
		}
	}
}

Expression SemanticParser::BuildGraph(
		Instance *instance,
		Parts *parts,
		Parts *dependency_parts,
		vector<double> *scores,
		const vector<double> *gold_outputs,
		vector<double> *predicted_outputs,
		Expression &y_pred,
		unordered_map<int, int> *form_count,
		bool is_train,
		ComputationGraph &cg) {

	vector<Expression> ex_lstm;
	RunLSTM(instance, l2rbuilder_, r2lbuilder_,
	        ex_lstm, form_count, is_train, cg);

	auto sentence = static_cast<SemanticInstanceNumeric *>(instance);
	auto semantic_parts = static_cast<SemanticParts *>(parts);

	Expression pred_b1 = cg_params_.at("pred_b1_");
	Expression pred_w2 = cg_params_.at("pred_w2_");
	Expression pred_b2 = cg_params_.at("pred_b2_");
	Expression pred_w3 = cg_params_.at("pred_w3_");
	Expression pred_b3 = cg_params_.at("pred_b3_");

	Expression unlab_b1 = cg_params_.at("unlab_b1_");
	Expression unlab_w2 = cg_params_.at("unlab_w2_");
	Expression unlab_b2 = cg_params_.at("unlab_b2_");
	Expression unlab_w3 = cg_params_.at("unlab_w3_");
	Expression unlab_b3 = cg_params_.at("unlab_b3_");

	Expression lab_b1 = cg_params_.at("lab_b1_");
	Expression lab_w2 = cg_params_.at("lab_w2_");
	Expression lab_b2 = cg_params_.at("lab_b2_");
	Expression lab_w3 = cg_params_.at("lab_w3_");
	Expression lab_b3 = cg_params_.at("lab_b3_");

	vector<Expression> ex_preds, ex_unlab_preds, ex_unlab_args,
			ex_lab_preds, ex_lab_args;

	Feature(instance, dependency_parts, y_pred,
	        ex_lstm,
	        ex_preds,
	        ex_unlab_preds,
	        ex_unlab_args,
	        ex_lab_preds,
	        ex_lab_args,
	        cg);

	vector<Expression> ex_scores(parts->size());
	scores->assign(parts->size(), 0.0);

	for (int r = 0; r < parts->size(); ++r) {
		if ((*parts)[r]->type() == SEMANTICPART_PREDICATE) {
			auto predicate = static_cast<SemanticPartPredicate *>((*parts)[r]);
			int idx_pred = predicate->predicate();
			Expression pred_MLP_in = tanh(pred_b1 + ex_preds[idx_pred]);
			Expression pred_phi = tanh(affine_transform({pred_b2, pred_w2, pred_MLP_in}));
			ex_scores[r] = affine_transform({pred_b3, pred_w3, pred_phi});
			(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
		} else if ((*parts)[r]->type() == SEMANTICPART_ARC) {
			auto arc = static_cast<SemanticPartArc *>((*parts)[r]);
			int idx_pred = arc->predicate();
			int idx_arg = arc->argument();
			Expression unlab_MLP_in = tanh(ex_unlab_preds[idx_pred]
			                               + ex_unlab_args[idx_arg] + unlab_b1);
			Expression unlab_phi = tanh(affine_transform({unlab_b2, unlab_w2, unlab_MLP_in}));
			ex_scores[r] = affine_transform({unlab_b3, unlab_w3, unlab_phi});
			(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));

			Expression lab_MLP_in = tanh(ex_lab_preds[idx_pred]
			                             + ex_lab_args[idx_arg] + lab_b1);
			Expression lab_phi = tanh(affine_transform({lab_b2, lab_w2, lab_MLP_in}));
			Expression lab_MLP_o = affine_transform({lab_b3, lab_w3, lab_phi});
			vector<float> label_scores = as_vector(cg.incremental_forward(lab_MLP_o));
			const vector<int> &index_labeled_parts =
					semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
			for (int k = 0; k < index_labeled_parts.size(); ++k) {
				auto labeled_arc = static_cast<SemanticPartLabeledArc *>(
						(*parts)[index_labeled_parts[k]]);
				ex_scores[index_labeled_parts[k]] = pick(lab_MLP_o, labeled_arc->role());
				(*scores)[index_labeled_parts[k]] = label_scores[labeled_arc->role()];
			}
		} else {
			CHECK_EQ((*parts)[r]->type(), SEMANTICPART_LABELEDARC);
		}
	}
	vector<Expression> i_errs;
	if (!is_train) {
		decoder_->Decode(instance, parts, *scores, predicted_outputs);
		for (int r = 0; r < parts->size(); ++r) {
			if (!NEARLY_EQ_TOL((*gold_outputs)[r], (*predicted_outputs)[r], 1e-6)) {
				Expression i_err = ((*predicted_outputs)[r] - (*gold_outputs)[r]) * ex_scores[r];
				i_errs.push_back(i_err);
			}
		}
		Expression loss = input(cg, 0.0);
		if (i_errs.size() > 0) {
			loss = loss + sum(i_errs);
		}
		return loss;
	}

	double s_loss = 0.0, s_cost = 0.0;
	decoder_->DecodeCostAugmented(instance, parts, *scores, *gold_outputs,
	                              predicted_outputs, &s_cost, &s_loss);
	for (int r = 0; r < parts->size(); ++r) {
		if (!NEARLY_EQ_TOL((*gold_outputs)[r], (*predicted_outputs)[r], 1e-6)) {
			Expression i_err = ((*predicted_outputs)[r] - (*gold_outputs)[r]) * ex_scores[r];
			i_errs.push_back(i_err);
		}
	}
	Expression loss = input(cg, s_cost);
	if (i_errs.size() > 0) {
		loss = loss + sum(i_errs);
	}
	return loss;
}