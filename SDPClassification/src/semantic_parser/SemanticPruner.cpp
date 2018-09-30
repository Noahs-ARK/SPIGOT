//
// Created by hpeng on 8/25/17.
//
#include <src/util/logval.h>
#include "SemanticPruner.h"


void SemanticPruner::InitParams(ParameterCollection *model) {
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
            {"unlab_w1_dist_", model->add_parameters({MLP_DIM, 1})},
            {"unlab_b1_", model->add_parameters({MLP_DIM})},
            {"unlab_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
            {"unlab_b2_", model->add_parameters({MLP_DIM})},
            {"unlab_w3_", model->add_parameters({1, MLP_DIM})},
            {"unlab_b3_", model->add_parameters({1})},
    };
}

Expression SemanticPruner::BuildGraph(Instance *instance, Parts *parts, vector<double> *scores,
                                      const vector<double> *gold_outputs, vector<double> *predicted_outputs,
                                      unordered_map<int, int> *form_count,
                                      const bool &is_train, ComputationGraph &cg) {
    auto sentence = dynamic_cast<SemanticInstanceNumeric *>(instance);
    const int slen = sentence->size() - 1;

    Expression pred_w1 = cg_params_.at("pred_w1_");
    Expression pred_b1 = cg_params_.at("pred_b1_");
    Expression pred_w2 = cg_params_.at("pred_w2_");
    Expression pred_b2 = cg_params_.at("pred_b2_");
    Expression pred_w3 = cg_params_.at("pred_w3_");
    Expression pred_b3 = cg_params_.at("pred_b3_");

    Expression unlab_w1_pred = cg_params_.at("unlab_w1_pred_");
    Expression unlab_w1_arg = cg_params_.at("unlab_w1_arg_");
    Expression unlab_w1_dist = cg_params_.at("unlab_w1_dist_");
    Expression unlab_b1 = cg_params_.at("unlab_b1_");
    Expression unlab_w2 = cg_params_.at("unlab_w2_");
    Expression unlab_b2 = cg_params_.at("unlab_b2_");
    Expression unlab_w3 = cg_params_.at("unlab_w3_");
    Expression unlab_b3 = cg_params_.at("unlab_b3_");

    vector<Expression> ex_lstm;
    RunLSTM(instance, l2rbuilder_, r2lbuilder_,
            ex_lstm, form_count, is_train, cg);

    vector<Expression> unlab_pred_exs(slen), unlab_arg_exs(slen);
    for (int i = 0; i < slen; ++i) {
        unlab_pred_exs[i] = (unlab_w1_pred * ex_lstm[i]);
        unlab_arg_exs[i] = (unlab_w1_arg * ex_lstm[i]);
    }

    vector<Expression> ex_scores(parts->size());
    scores->assign(parts->size(), 0.0);
    predicted_outputs->assign(parts->size(), 0.0);
    for (int r = 0; r < parts->size(); ++r) {
        if ((*parts)[r]->type() == SEMANTICPART_PREDICATE) {
            auto predicate = dynamic_cast<SemanticPartPredicate *>((*parts)[r]);
            int idx_pred = predicate->predicate();
	        Expression pred_MLP_in = tanh(affine_transform({pred_b1, pred_w1, ex_lstm[idx_pred]}));
	        Expression pred_phi = tanh(affine_transform({pred_b2, pred_w2, pred_MLP_in}));
	        ex_scores[r] = affine_transform({pred_b3, pred_w3, pred_phi});
            (*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
        } else if ((*parts)[r]->type() == SEMANTICPART_ARC) {
            auto arc = dynamic_cast<SemanticPartArc *>((*parts)[r]);
            int idx_pred = arc->predicate();
            int idx_arg = arc->argument();
	        float dist = idx_arg - idx_pred;
	        Expression ex_dist = unlab_w1_dist * input(cg, dist);
	        Expression unlab_MLP_in = tanh(unlab_pred_exs[idx_pred] + unlab_arg_exs[idx_arg] + unlab_b1
	                                       + ex_dist);
	        Expression unlab_phi = tanh(affine_transform({unlab_b2, unlab_w2, unlab_MLP_in}));
	        ex_scores[r] = affine_transform({unlab_b3, unlab_w3, unlab_phi});
            (*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
        }
    }
    vector<Expression> i_errs;
    if (!is_train) {
        decoder_->DecodePruner(instance, parts, *scores, predicted_outputs);
        return input(cg, 0.0);
    }

    Expression entropy = input(cg, 0.0);
    DecodeBasicMarginals(instance, parts, ex_scores,
                         predicted_outputs, entropy, cg);
    for (int r = 0; r < parts->size(); ++r) {
        if ((*gold_outputs)[r] != (*predicted_outputs)[r]) {
            Expression i_err = ((*predicted_outputs)[r] - (*gold_outputs)[r]) * ex_scores[r];
            i_errs.push_back(i_err);
        }
    }
    if (i_errs.size() > 0) {
        entropy = entropy + sum(i_errs);
    }
    return entropy;
}

void SemanticPruner::DecodeBasicMarginals(Instance *instance, Parts *parts,
                                  const vector<Expression> &scores,
                                  vector<double> *predicted_output,
                                  Expression &entropy, ComputationGraph &cg) {
    int slen = dynamic_cast<SemanticInstanceNumeric *>(instance)->size() - 1;
    auto semantic_parts = dynamic_cast<SemanticParts *>(parts);
    int offset_predicate_parts, num_predicate_parts;
    int offset_arcs, num_arcs;
    semantic_parts->GetOffsetPredicate(&offset_predicate_parts,
                                       &num_predicate_parts);
    semantic_parts->GetOffsetArc(&offset_arcs, &num_arcs);

    vector<SemanticPartArc *> arcs(num_arcs);
    vector<Expression> scores_arcs(num_arcs);
    for (int r = 0; r < num_arcs; ++r) {
        arcs[r] = static_cast<SemanticPartArc *>((*parts)[offset_arcs + r]);
        scores_arcs[r] = scores[offset_arcs + r];
    }

    vector<vector<vector<int> > > arcs_by_predicate;
    arcs_by_predicate.resize(slen);

    for (int r = 0; r < num_arcs; ++r) {
        int p = arcs[r]->predicate();
        int s = arcs[r]->sense();
        if (s >= arcs_by_predicate[p].size()) {
            arcs_by_predicate[p].resize(s + 1);
        }
        arcs_by_predicate[p][s].push_back(r);
    }

    vector<Expression> scores_predicates(num_predicate_parts);
    vector<vector<int> > index_predicates(slen);
    for (int r = 0; r < num_predicate_parts; ++r) {
        scores_predicates[r] = scores[offset_predicate_parts + r];
        auto predicate_part = dynamic_cast<SemanticPartPredicate *>((*parts)[offset_predicate_parts + r]);
        int p = predicate_part->predicate();
        int s = predicate_part->sense();
        if (s >= index_predicates[p].size()) {
            index_predicates[p].resize(s + 1, -1);
        }
        index_predicates[p][s] = r;
    }

    predicted_output->assign(parts->size(), 0.0);

    Expression log_partition_function = input(cg, 0.0);

    for (int p = 0; p < slen; ++p) {
        Expression log_partition_all_senses = input(cg, 0.0);
        vector<Expression> log_partition_senses(arcs_by_predicate[p].size());
        vector<vector<Expression> > log_partition_arcs(arcs_by_predicate[p].size());

        for (int s = 0; s < arcs_by_predicate[p].size(); ++s) {
            int r = index_predicates[p][s];
            Expression score = scores_predicates[r];
            log_partition_arcs[s].assign(arcs_by_predicate[p][s].size(),
                                         input(cg, 0.0));
            for (int k = 0; k < arcs_by_predicate[p][s].size(); ++k) {
                int r = arcs_by_predicate[p][s][k];
                log_partition_arcs[s][k] = logsumexp({log_partition_arcs[s][k], scores_arcs[r]});
                score = score + log_partition_arcs[s][k];
            }
            log_partition_senses[s] = score;
            log_partition_all_senses = logsumexp({log_partition_all_senses, log_partition_senses[s]});
        }

        if (arcs_by_predicate[p].size() > 0) {
            log_partition_function = log_partition_function + log_partition_all_senses;
        }

        for (int s = 0; s < arcs_by_predicate[p].size(); ++s) {
            int r = index_predicates[p][s];
            Expression predicate_marginal = exp(log_partition_senses[s] - log_partition_all_senses);

            (*predicted_output)[offset_predicate_parts + r]
                    = as_scalar(cg.incremental_forward(predicate_marginal));
            entropy = entropy - scores_predicates[r] * predicate_marginal;

            for (int k = 0; k < arcs_by_predicate[p][s].size(); ++k) {
                int r = arcs_by_predicate[p][s][k];
                Expression marginal = exp(scores_arcs[r] - log_partition_arcs[s][k]);
                marginal = marginal * predicate_marginal;
                (*predicted_output)[offset_arcs + r]
                        = as_scalar(cg.incremental_forward(marginal));
                entropy = entropy - scores_arcs[r] * marginal;
            }
        }
    }
    entropy = entropy + log_partition_function;
}