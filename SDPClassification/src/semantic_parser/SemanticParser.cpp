//
// Created by hpeng on 10/16/17.
//

#include "SemanticParser.h"

void SemanticParser::InitParams(ParameterCollection *model) {
	BiLSTM::InitParams(model);
	params_ = {
			// parser
			{"pred_w1_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"pred_b1_", model->add_parameters({MLP_DIM})},
			{"pred_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"pred_b2_", model->add_parameters({MLP_DIM})},
			{"pred_w3_", model->add_parameters({1, MLP_DIM})},
			{"pred_b3_", model->add_parameters({1})},
			// unlabeled arc
			{"unlab_w1_pred_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w1_arg_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_b1_", model->add_parameters({MLP_DIM})},
			{"unlab_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"unlab_b2_", model->add_parameters({MLP_DIM})},
			{"unlab_w3_", model->add_parameters({1, MLP_DIM})},
			{"unlab_b3_", model->add_parameters({1})},
			// labeled arc
			{"lab_w1_pred_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"lab_w1_arg_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"lab_b1_", model->add_parameters({MLP_DIM})},
			{"lab_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"lab_b2_", model->add_parameters({MLP_DIM})},
			{"lab_w3_", model->add_parameters({ROLE_SIZE, MLP_DIM})},
			{"lab_b3_", model->add_parameters({ROLE_SIZE})}
	};
}

Expression SemanticParser::BuildGraph(Instance *instance,
                              Parts *parts, vector<double> *scores,
                              const vector<double> *gold_outputs,
                              vector<double> *predicted_outputs,
                              Expression &ex_score, Expression &y_pred,
                              unordered_map<int, int> *form_count,
                              bool is_train, ComputationGraph &cg) {
	vector<Expression> ex_lstm;
	RunLSTM(instance, l2rbuilder_, r2lbuilder_,
	        ex_lstm, form_count, is_train, cg);

	auto sentence = dynamic_cast<SemanticInstanceNumeric *>(instance);
	const int slen = sentence->size() - 1;

	auto semantic_parts = dynamic_cast<SemanticParts *>(parts);

	int offset_arcs, num_arcs;
	semantic_parts->GetOffsetArc(&offset_arcs, &num_arcs);
	int offset_lab_arcs, num_lab_arcs;
	semantic_parts->GetOffsetLabeledArc(&offset_lab_arcs,
	                                    &num_lab_arcs);
	bool labeled_parsing = num_lab_arcs > 0;
	// parser
	Expression pred_w1 = cg_params_.at("pred_w1_");
	Expression pred_b1 = cg_params_.at("pred_b1_");
	Expression pred_w2 = cg_params_.at("pred_w2_");
	Expression pred_b2 = cg_params_.at("pred_b2_");
	Expression pred_w3 = cg_params_.at("pred_w3_");
	Expression pred_b3 = cg_params_.at("pred_b3_");

	Expression unlab_w1_pred = cg_params_.at("unlab_w1_pred_");
	Expression unlab_w1_arg = cg_params_.at("unlab_w1_arg_");
	Expression unlab_b1 = cg_params_.at("unlab_b1_");
	Expression unlab_w2 = cg_params_.at("unlab_w2_");
	Expression unlab_b2 = cg_params_.at("unlab_b2_");
	Expression unlab_w3 = cg_params_.at("unlab_w3_");
	Expression unlab_b3 = cg_params_.at("unlab_b3_");

	Expression lab_w1_pred = cg_params_.at("lab_w1_pred_");
	Expression lab_w1_arg = cg_params_.at("lab_w1_arg_");
	Expression lab_b1 = cg_params_.at("lab_b1_");
	Expression lab_w2 = cg_params_.at("lab_w2_");
	Expression lab_b2 = cg_params_.at("lab_b2_");
	Expression lab_w3 = cg_params_.at("lab_w3_");
	Expression lab_b3 = cg_params_.at("lab_b3_");
	vector<Expression> unlab_pred_exs, unlab_arg_exs;
	vector<Expression> lab_pred_exs, lab_arg_exs;
	for (int i = 0; i < slen; ++i) {
		unlab_pred_exs.push_back(unlab_w1_pred * ex_lstm[i]);
		unlab_arg_exs.push_back(unlab_w1_arg * ex_lstm[i]);
		if (labeled_parsing) {
			lab_pred_exs.push_back(lab_w1_pred * ex_lstm[i]);
			lab_arg_exs.push_back(lab_w1_arg * ex_lstm[i]);
		}
	}
	vector<Expression> ex_scores(parts->size());
	scores->resize(parts->size());
	for (int r = 0; r < parts->size(); ++r) {
		if ((*parts)[r]->type() == SEMANTICPART_PREDICATE) {
			auto predicate = dynamic_cast<SemanticPartPredicate *>((*parts)[r]);
			int idx_pred = predicate->predicate();
			Expression pred_ex = ex_lstm[idx_pred];
			Expression pred_MLP_in = tanh(
					affine_transform({pred_b1, pred_w1, pred_ex}));
			Expression pred_phi = tanh(
					affine_transform({pred_b2, pred_w2, pred_MLP_in}));
			ex_scores[r] = affine_transform({pred_b3, pred_w3, pred_phi});
			(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
		} else if ((*parts)[r]->type() == SEMANTICPART_ARC) {
			auto arc = dynamic_cast<SemanticPartArc *>((*parts)[r]);
			int idx_pred = arc->predicate();
			int idx_arg = arc->argument();

			Expression unlab_MLP_in = tanh(
					unlab_pred_exs[idx_pred] + unlab_arg_exs[idx_arg] +
					unlab_b1);
			Expression unlab_phi = tanh(
					affine_transform({unlab_b2, unlab_w2, unlab_MLP_in}));
			ex_scores[r] = affine_transform({unlab_b3, unlab_w3, unlab_phi});
			(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
			if (labeled_parsing) {
				Expression lab_MLP_in = tanh(
						lab_pred_exs[idx_pred] + lab_arg_exs[idx_arg] + lab_b1);
				Expression lab_phi = tanh(
						affine_transform({lab_b2, lab_w2, lab_MLP_in}));
				Expression lab_MLP_o = affine_transform(
						{lab_b3, lab_w3, lab_phi});
				vector<float> label_scores = as_vector(
						cg.incremental_forward(lab_MLP_o));
				const vector<int> &index_labeled_parts =
						semantic_parts->FindLabeledArcs(arc->predicate(),
						                                arc->argument(),
						                                arc->sense());
				for (int k = 0; k < index_labeled_parts.size(); ++k) {
					auto labeled_arc = dynamic_cast<SemanticPartLabeledArc *>(
									(*parts)[index_labeled_parts[k]]);
					ex_scores[index_labeled_parts[k]] = pick(lab_MLP_o,
					                                         labeled_arc->role());
					(*scores)[index_labeled_parts[k]] = label_scores[labeled_arc->role()];
				}
			}
		}
	}
	if (!is_train) {
		decoder_->Decode(instance, parts, *scores, predicted_outputs, false);
		int num_vars = num_arcs, offset_vars = offset_arcs;
		if (labeled_parsing) {
			num_vars = num_arcs + num_lab_arcs;
		}
		vector<dynet::real> float_predicted_outputs(num_vars, 0.0);
		for (int i = 0; i < num_vars; ++i) {
			float_predicted_outputs[i] = (*predicted_outputs)[i + offset_vars];
		}
		vector<Expression> score_vars(ex_scores.begin() + offset_vars,
		                              ex_scores.begin() + offset_vars
		                              + num_vars);
		Expression ex_p = input(cg, {num_vars}, float_predicted_outputs);
		ex_score = concatenate(score_vars);

		if (PROJECT) {
			y_pred = argmax_proj_sdp(ex_score, ex_p, instance, parts);
		} else {
			y_pred = argmax_ste(ex_score, ex_p);
		}
		return input(cg, 0.0);
	}

	vector<Expression> i_errs;
	double s_loss = 0.0, s_cost = 0.0;
	decoder_->DecodeCostAugmented(instance, parts, *scores, *gold_outputs,
	                              predicted_outputs, &s_cost, &s_loss);
	for (int r = 0; r < parts->size(); ++r) {
		if (!NEARLY_EQ_TOL((*gold_outputs)[r], (*predicted_outputs)[r], 1e-6)) {
			Expression i_err =
					((*predicted_outputs)[r] - (*gold_outputs)[r]) * ex_scores[r];
			i_errs.push_back(i_err);
		}
	}
	Expression loss = input(cg, s_cost);
	if (i_errs.size() > 0) {
		loss = loss + sum(i_errs);
	}
	return loss;
}