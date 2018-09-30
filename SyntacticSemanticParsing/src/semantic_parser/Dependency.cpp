//
// Created by hpeng on 10/31/17.
//

#include "Dependency.h"


void Dependency::InitParams(ParameterCollection *model) {

	// shared
	BiLSTM::InitParams(model);

	// dependency alone
	params_ = {
			{"unlab_w1_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w1_mod_",  model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_b1_",      model->add_parameters({MLP_DIM})},
			{"unlab_w2_",      model->add_parameters({MLP_DIM, MLP_DIM})},
			{"unlab_b2_",      model->add_parameters({MLP_DIM})},
			{"unlab_w3_",      model->add_parameters({1, MLP_DIM})},
			{"unlab_b3_",      model->add_parameters({1})}
	};
}

Expression
Dependency::BuildGraph(Instance *instance, Parts *parts, vector<double> *scores,
                       const vector<double> *gold_outputs, vector<double> *predicted_outputs,
                       Expression &ex_score, Expression &y_pred, unordered_map<int, int> *form_count,
                       bool is_train, ComputationGraph &cg) {
	vector<Expression> ex_lstm;
	RunLSTM(instance, l2rbuilder_, r2lbuilder_,
	        ex_lstm, form_count, is_train, cg);

	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	const int slen = sentence->size() - 1;

	Expression unlab_w1_head = cg_params_.at("unlab_w1_head_");
	Expression unlab_w1_mod = cg_params_.at("unlab_w1_mod_");
	Expression unlab_b1 = cg_params_.at("unlab_b1_");
	Expression unlab_w2 = cg_params_.at("unlab_w2_");
	Expression unlab_b2 = cg_params_.at("unlab_b2_");
	Expression unlab_w3 = cg_params_.at("unlab_w3_");
	Expression unlab_b3 = cg_params_.at("unlab_b3_");

	vector<Expression> ex_scores(parts->size());
	scores->assign(parts->size(), 0.0);
	vector<Expression> unlab_head_exs, unlab_mod_exs;
	for (int i = 0; i < slen; ++i) {
		unlab_head_exs.push_back(unlab_w1_head * ex_lstm[i]);
		unlab_mod_exs.push_back(unlab_w1_mod * ex_lstm[i]);
	}

	for (int r = 0; r < parts->size(); ++r) {
		if ((*parts)[r]->type() == DEPENDENCYPART_ARC) {
			auto arc = static_cast<DependencyPartArc *>((*parts)[r]);
			int h = arc->head();
			int m = arc->modifier();

			Expression unlab_MLP_in = tanh(
					unlab_head_exs[h] + unlab_mod_exs[m] + unlab_b1);
			Expression unlab_phi = tanh(
					affine_transform({unlab_b2, unlab_w2, unlab_MLP_in}));
			ex_scores[r] = affine_transform({unlab_b3, unlab_w3, unlab_phi});
			(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
		} else {
			CHECK(false);
		}
	}
	if (!is_train) {
		decoder_->Decode(instance, parts, *scores, predicted_outputs);
		int num_arcs, offset_arcs;
		dependency_parts->GetOffsetArc(&offset_arcs, &num_arcs);
		vector<dynet::real> float_predicted_outputs(num_arcs, 0.0);
		for (int i = 0; i < num_arcs; ++i) {
			int r = i + offset_arcs;
			float_predicted_outputs[i] = (*predicted_outputs)[r];
		}
		vector<Expression> score_arcs(ex_scores.begin() + offset_arcs,
		                              ex_scores.begin() + offset_arcs +
		                              num_arcs);
		Expression ex_p = input(cg, {num_arcs}, float_predicted_outputs);
		ex_score = concatenate(score_arcs);

		if (PROJECT) {
			y_pred = argmax_proj_singlehead(ex_score, ex_p, instance, parts);
		} else {
			y_pred = argmax_ste(ex_score, ex_p);
		}
		return input(cg, 0.0);
	}

	double s_loss = 0.0, s_cost = 0.0;
	decoder_->DecodeCostAugmented(instance, parts, (*scores), (*gold_outputs),
	                              predicted_outputs, &s_cost, &s_loss);

	vector<dynet::real> float_predicted_outputs(parts->size(), 0.0);
	vector<dynet::real> float_gold_outputs(parts->size(), 0.0);
	vector<dynet::real> float_scores(parts->size(), 0.0);
	for (int i = 0; i < parts->size(); ++i) {
		float_predicted_outputs[i] = (*predicted_outputs)[i];
		float_gold_outputs[i] = (*gold_outputs)[i];
		float_scores[i] = (*scores)[i];
	}
	Expression ex_pred = input(cg, {parts->size()}, float_predicted_outputs);
	Expression ex_gold = input(cg, {parts->size()}, float_gold_outputs);
	Expression loss = input(cg, s_cost) +
	                  dot_product(ex_pred - ex_gold, concatenate(ex_scores));
	return loss;
}