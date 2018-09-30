//
// Created by hpeng on 9/13/17.
//

#include "DependencyPruner.h"

void DependencyPruner::InitParams(ParameterCollection *model) {
	// shared
	BiLSTM::InitParams(model);
	// unlabeled arc
	params_ = {
			{"unlab_w1_head_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w1_mod_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w_dist_", model->add_parameters({MLP_DIM, 1})},
			{"unlab_b1_", model->add_parameters({MLP_DIM})},
			{"unlab_w2_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"unlab_b2_", model->add_parameters({MLP_DIM})},
			{"unlab_w3_", model->add_parameters({1, MLP_DIM})},
			{"unlab_b3_", model->add_parameters({1})},
	};
}

Expression DependencyPruner::BuildGraph(Instance *instance,
                                        Parts *parts, vector<double> *scores,
                                        const vector<double> *gold_outputs,
                                        vector<double> *predicted_outputs,
                                        unordered_map<int, int> *form_count,
                                        bool is_train, ComputationGraph &cg) {
	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	const int slen = sentence->size() - 1;

	Expression unlab_w1_head = cg_params_.at("unlab_w1_head_");
	Expression unlab_w1_mod = cg_params_.at("unlab_w1_mod_");
	Expression unlab_w_dist = cg_params_.at("unlab_w_dist_");
	Expression unlab_b1 = cg_params_.at("unlab_b1_");
	Expression unlab_w2 = cg_params_.at("unlab_w2_");
	Expression unlab_b2 = cg_params_.at("unlab_b2_");
	Expression unlab_w3 = cg_params_.at("unlab_w3_");
	Expression unlab_b3 = cg_params_.at("unlab_b3_");

	vector<Expression> ex_lstm;
	RunLSTM(instance, l2rbuilder_, r2lbuilder_,
	        ex_lstm, form_count, is_train, cg);

	vector<Expression> unlab_head_exs(slen), unlab_mod_exs(slen);
	for (int i = 0; i < slen; ++i) {
		unlab_head_exs[i] = (unlab_w1_head * ex_lstm[i]);
		unlab_mod_exs[i] = (unlab_w1_mod * ex_lstm[i]);
	}

	vector<Expression> ex_scores(parts->size());
	scores->assign(parts->size(), 0.0);
	predicted_outputs->assign(parts->size(), 0.0);
	for (int r = 0; r < parts->size(); ++r) {
		if ((*parts)[r]->type() == DEPENDENCYPART_ARC) {
			auto arc = static_cast<DependencyPartArc *>((*parts)[r]);
			int h = arc->head();
			int m = arc->modifier();
			float dist = m - h;
			Expression ex_dist = unlab_w_dist * input(cg, dist);
			Expression unlab_MLP_in = tanh(
					unlab_head_exs[h] + unlab_mod_exs[m] + unlab_b1 + ex_dist);
			Expression unlab_phi = tanh(
					affine_transform({unlab_b2, unlab_w2, unlab_MLP_in}));
			ex_scores[r] = affine_transform({unlab_b3, unlab_w3, unlab_phi});
			(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
		}
	}
	vector<Expression> i_errs;
	if (!is_train) {
		DecodePruner(instance, parts, *scores, predicted_outputs);
		return input(cg, 0.0);
	}

	Expression entropy = input(cg, 0.0);
	bool projective = true;
	if (projective) {
		DecodeInsideOutside(instance, parts, ex_scores,
		                 predicted_outputs, entropy, cg);
	} else {
		DecodeMatrixTree(instance, parts, ex_scores,
		                 predicted_outputs, entropy, cg);
	}
	for (int r = 0; r < parts->size(); ++r) {
		if ((*gold_outputs)[r] != (*predicted_outputs)[r]) {
			Expression i_err = ((*predicted_outputs)[r] - (*gold_outputs)[r]) *
			                   ex_scores[r];
			i_errs.push_back(i_err);
		}
	}
	if (i_errs.size() > 0) {
		entropy = entropy + sum(i_errs);
	}
	return entropy;
}

void DependencyPruner::DecodeMatrixTree(Instance *instance, Parts *parts,
                                        const vector<Expression> &scores,
                                        vector<double> *predicted_output,
                                        Expression &entropy,
                                        ComputationGraph &cg) {
	CHECK(false);
	int slen = static_cast<DependencyInstanceNumeric *>(instance)->size() - 1;
	DependencyParts *dependency_parts = static_cast<DependencyParts *>(parts);

	vector<Expression> potentials(slen * slen);
	vector<Expression> kirchhoff((slen - 1) * (slen - 1));

	int offset_arcs, num_arcs;
	dependency_parts->GetOffsetArc(&offset_arcs, &num_arcs);
	Expression constant = sum(scores) / static_cast<double>(num_arcs);

	for (int h = slen - 1; h >= 0; --h) {
		for (int m = slen - 1; m >= 0; --m) {
			int r = dependency_parts->FindArc(h, m);
			if (r >= 0) {
				potentials[m * slen + h] = scores[r] - constant;
				LOG(INFO) << potentials[m * slen + h].value();
				if (h > 0 && m > 0) {
					kirchhoff[(h - 1) * (slen - 1) + (m - 1)]
							= -exp(potentials[m * slen + h]);

				}
			} else {
				potentials[m * slen + h] = input(cg,
				                                 -std::numeric_limits<float>::infinity());
				if (h > 0 && m > 0) {
					kirchhoff[(h - 1) * (slen - 1) + (m - 1)]
							= input(cg, 0.0);
				}
			}
		}
	}

	// Set the Kirchhoff matrix.
	for (int m = 1; m < slen; ++m) {
		Expression sum = input(cg, 0.0);
		for (int h = 0; h < slen; ++h) {
			sum = sum + exp(potentials[m * slen + h]);
		}
		kirchhoff[(m - 1) * (slen - 1) + (m - 1)] = sum;
	}

	// Inverse of the Kirchoff matrix.
	Expression lu = reshape(concatenate(kirchhoff), {slen - 1, slen - 1});
	Expression inverted_kirchhoff = reshape(inverse(lu),
	                                        {(slen - 1) * (slen - 1)});
	Expression log_partition_function = logdet(lu) + constant * (slen - 1);
	double s = as_scalar(cg.incremental_forward(log_partition_function));

	vector<Expression> marginals(slen * slen);
	for (int h = 0; h < slen; ++h) {
		marginals[h] = input(cg, 0.0);
	}
	for (int m = 1; m < slen; ++m) {
		Expression tmp_ik = pick(inverted_kirchhoff,
		                         (m - 1) * (slen - 1) + (m - 1));
		marginals[m * slen] = exp(potentials[m * slen]) * tmp_ik;
		for (int h = 1; h < slen; ++h) {
			marginals[m * slen + h] = exp(potentials[m * slen + h])
			                          * (tmp_ik - pick(inverted_kirchhoff,
			                                           (m - 1) * (slen - 1) +
			                                           (h - 1)));
		}
	}

	// Compute the entropy.
	predicted_output->resize(parts->size());
	entropy = log_partition_function;
	for (int r = 0; r < num_arcs; ++r) {
		int h = static_cast<DependencyPartArc *>((*parts)[offset_arcs +
		                                                  r])->head();
		int m = static_cast<DependencyPartArc *>((*parts)[offset_arcs +
		                                                  r])->modifier();
		double marginal = as_scalar(
				cg.incremental_forward(marginals[m * slen + h]));
		if (marginal < 0) {
			if (!NEARLY_ZERO_TOL(marginal, 1e-6)) {
				marginals[m * slen + h] = input(cg, 0.0);
				marginal = 0.0;
			}
		} else if (marginal > 1.0) {
			if (!NEARLY_ZERO_TOL(marginal - 1.0, 1e-6)) {
				marginals[m * slen + h] = input(cg, 1.0);
				marginal = 1.0;
			}
		}

		(*predicted_output)[offset_arcs + r] = marginal;
		entropy = entropy - marginals[m * slen + h] * scores[offset_arcs + r];
	}


	double s_entropy = as_scalar(cg.incremental_forward(entropy));
	CHECK(!std::isnan(s_entropy));
	if (s_entropy < 0.0) {
		if (!NEARLY_ZERO_TOL(s_entropy, 1e-4)) {
			LOG(INFO) << "Entropy truncated to zero (" << s_entropy << ")";
		}
		entropy = input(cg, 0.0);
	}
}

void DependencyPruner::DecodePruner(Instance *instance, Parts *parts,
                                    const vector<double> &scores,
                                    vector<double> *predicted_output) {
	int slen = static_cast<DependencyInstanceNumeric *>(instance)->size() - 1;
	auto dependency_parts = static_cast<DependencyParts *>(parts);

	// TODO: change this
	double posterior_threshold = 0.0001;
	int max_heads = 10;
	if (max_heads < 0) max_heads = slen;
	predicted_output->clear();
	predicted_output->resize(parts->size(), 0.0);

	CHECK(dependency_parts->IsArcFactored());

	double entropy = 0.0, log_partition_function = 0.0;
	vector<double> posteriors;
	bool projective = true;
	if (projective) {
		decoder_->DecodeInsideOutside(instance, parts, scores, &posteriors,
		                              &log_partition_function, &entropy);
	} else {
		decoder_->DecodeMatrixTree(instance, parts, scores, &posteriors,
		                           &log_partition_function, &entropy);
	}


	int num_used_parts = 0;
	for (int m = 1; m < slen; ++m) {
		vector<pair<double, int> > scores_heads;
		for (int h = 0; h < slen; ++h) {
			int r = dependency_parts->FindArc(h, m);
			if (r < 0) continue;
			scores_heads.push_back(pair<double, int>(-posteriors[r], r));
		}
		if (scores_heads.size() == 0) continue;
		sort(scores_heads.begin(), scores_heads.end());
		double max_posterior = -scores_heads[0].first;
		for (int k = 0; k < max_heads && k < scores_heads.size(); ++k) {
			int r = scores_heads[k].second;
			if (k == 0 ||
			    -scores_heads[k].first >= posterior_threshold * max_posterior) {
				++num_used_parts;
				(*predicted_output)[r] = 1.0;
			} else {
				break;
			}
		}
	}

	VLOG(2) << "Pruning reduced to "
	        << static_cast<double>(num_used_parts) /
	           static_cast<double>(slen)
	        << " candidate heads per word.";
}


void DependencyPruner::RunEisnerInside(int slen,
                                       const vector<DependencyPartArc *> &arcs,
                                       const vector<Expression> &scores,
                                       vector<Expression> &inside_incomplete_spans,
                                       vector<vector<Expression> > &inside_complete_spans,
                                       vector<vector<bool> > &flag_ics,
                                       Expression &log_partition_function,
                                       ComputationGraph &cg) {
	vector<vector<int> > index_arcs(slen,
	                                vector<int>(slen, -1));
	int num_arcs = arcs.size();
	for (int r = 0; r < num_arcs; ++r) {
		int h = arcs[r]->head();
		int m = arcs[r]->modifier();
		index_arcs[h][m] = r;
	}

	// Initialize CKY table.
	inside_incomplete_spans.assign(num_arcs, input(cg, 0.0));
	inside_complete_spans.resize(slen);
	for (int s = 0; s < slen; ++s) {
		inside_complete_spans[s].assign(slen, input(cg, 0.0));
	}
	// set true if it's -inf
	flag_ics.assign(slen, vector<bool>(slen, false));


	// Loop from smaller items to larger items.
	for (int k = 1; k < slen; ++k) {
		for (int s = 1; s < slen - k; ++s) {
			int t = s + k;

			// First, create incomplete items.
			int left_arc_index = index_arcs[t][s];
			int right_arc_index = index_arcs[s][t];
			if (left_arc_index >= 0 || right_arc_index >= 0) {
				vector<Expression> list;
				bool neg_inf = false;
				bool number = false;
				for (int u = s; u < t; ++u) {
					// void push multiple neg_infs;
					if (neg_inf && (flag_ics[s][u] || flag_ics[t][u + 1])) {
						continue;
					}
					list.push_back(inside_complete_spans[s][u] + inside_complete_spans[t][u + 1]);
					if (flag_ics[s][u] || flag_ics[t][u + 1])
						neg_inf = true;
					else
						number = true;
				}
				Expression sum = logsumexp(list);
				if (left_arc_index >= 0) {
					inside_incomplete_spans[left_arc_index] =
							sum + scores[left_arc_index];
				}
				if (right_arc_index >= 0) {
					inside_incomplete_spans[right_arc_index] =
							sum + scores[right_arc_index];
				}
			}

			// Second, create complete items.
			// 1) Left complete item.
			vector<Expression> list;
			bool neg_inf = false;
			bool number = false;
			for (int u = s; u < t; ++u) {
				int left_arc_index = index_arcs[t][u];
				if (left_arc_index >= 0) {
					if (neg_inf && flag_ics[u][s])
						continue;
					list.push_back(inside_complete_spans[u][s] + inside_incomplete_spans[left_arc_index]);
					if (flag_ics[u][s])
						neg_inf = true;
					else
						number = true;
				}
			}
			if (list.size() > 0) {
				if (!number)
					flag_ics[t][s] = true;
				inside_complete_spans[t][s] = logsumexp(list);
			}
			else {
				flag_ics[t][s] = true;
				inside_complete_spans[t][s] = input(cg, -std::numeric_limits<float>::infinity());
			}

			// 2) Right complete item.
			list.clear();
			neg_inf = false;
			number = false;
			for (int u = s + 1; u <= t; ++u) {
				int right_arc_index = index_arcs[s][u];
				if (right_arc_index >= 0) {
					if (neg_inf && flag_ics[u][t])
						continue;
					list.push_back(inside_complete_spans[u][t] + inside_incomplete_spans[right_arc_index]);
					if (flag_ics[u][t])
						neg_inf = true;
					else
						number = true;
				}
			}
			if (list.size() > 0) {
				if (!number)
					flag_ics[s][t] = true;
				inside_complete_spans[s][t] = logsumexp(list);
			}
			else {
				flag_ics[s][t] = true;
				inside_complete_spans[s][t] = input(cg, -std::numeric_limits<float>::infinity());
			}
		}
	}

	// Handle the (single) root.
	vector<Expression> list;
	bool neg_inf = false;
	for (int s = 1; s < slen; ++s) {
		int arc_index = index_arcs[0][s];
		if (arc_index >= 0) {
			inside_incomplete_spans[arc_index] = logsumexp(
					{inside_complete_spans[s][1],
					 scores[arc_index]});
			inside_incomplete_spans[arc_index] = inside_complete_spans[s][1] + scores[arc_index];
			if (neg_inf && flag_ics[s][slen - 1])
				continue;
			list.push_back(inside_incomplete_spans[arc_index] + inside_complete_spans[s][slen - 1]);
			if (flag_ics[s][slen - 1])
				neg_inf = true;
		}
	}
	inside_complete_spans[0][slen - 1] = logsumexp(list);

	log_partition_function = inside_complete_spans[0][slen - 1];
}

void DependencyPruner::RunEisnerOutside(int slen, const vector<DependencyPartArc *> &arcs,
                                        const vector<Expression> &scores,
                                        const vector<Expression> &inside_incomplete_spans,
                                        const vector<vector<Expression> > &inside_complete_spans,
                                        const vector<vector<bool> > &flag_ics,
                                        vector<Expression> &outside_incomplete_spans,
                                        vector<vector<Expression> > &outside_complete_spans,
                                        ComputationGraph &cg) {
	vector<vector<int> > index_arcs(slen, vector<int>(slen, -1));
	int num_arcs = arcs.size();
	for (int r = 0; r < num_arcs; ++r) {
		int h = arcs[r]->head();
		int m = arcs[r]->modifier();
		index_arcs[h][m] = r;
	}

	// Initialize CKY table.
	outside_incomplete_spans.assign(num_arcs, input(cg, 0.0));
	outside_complete_spans.resize(slen);
	for (int s = 0; s < slen; ++s) {
		outside_complete_spans[s].assign(slen, input(cg, 0.0));
	}
	vector<vector<bool>> flag_ocs(slen, vector<bool>(slen, false));
	vector<bool> flag_ois(num_arcs, false);
	// set true if it's -inf;

	// Handle the root.
	for (int s = 1; s < slen; ++s) {
		int arc_index = index_arcs[0][s];
		if (arc_index >= 0) {
			outside_incomplete_spans[arc_index] =
					outside_complete_spans[0][slen - 1] +
					inside_complete_spans[s][slen - 1];
		}
	}

	// Loop from larger items to smaller items.
	for (int k = slen - 2; k > 0; --k) {
		for (int s = 1; s < slen - k; ++s) {
			int t = s + k;

			// First, create complete items.
			// 1) Left complete item.
			vector<Expression> list;
			bool neg_inf = false;
			bool number = false;
			for (int u = 0; u < s; ++u) {
				if (u == 0 && t < slen - 1) continue;
				int arc_index = index_arcs[u][s];
				if (arc_index >= 0) {
					if (neg_inf && flag_ocs[u][t])
						continue;
					list.push_back(outside_complete_spans[u][t] + inside_incomplete_spans[arc_index]);
					if (flag_ocs[u][t])
						neg_inf = true;
					else
						number = true;
				}
			}
			for (int u = t + 1; u < slen; ++u) {
				int left_arc_index = index_arcs[u][s];
				int right_arc_index = index_arcs[s][u];
				if (right_arc_index >= 0) {
					if (neg_inf && (flag_ics[u][t + 1] || flag_ois[right_arc_index]))
						continue;
					list.push_back(outside_incomplete_spans[right_arc_index]
								+ inside_complete_spans[u][t + 1]
								+ scores[right_arc_index]);
					if (flag_ics[u][t + 1] || flag_ois[right_arc_index])
						neg_inf = true;
					else
						number = true;
				}
				if (left_arc_index >= 0) {
					if (neg_inf && (flag_ics[u][t + 1] || flag_ois[left_arc_index]))
						continue;
					list.push_back(outside_incomplete_spans[left_arc_index]
					               + inside_complete_spans[u][t + 1]
					               + scores[left_arc_index]);
					if (flag_ics[u][t + 1] || flag_ois[left_arc_index])
						neg_inf = true;
					else
						number = true;
				}
			}
			if (list.size() > 0) {
				if (!number)
					flag_ocs[s][t] = true;
				outside_complete_spans[s][t] = logsumexp(list);
			}
			else {
				flag_ocs[s][t] = true;
				outside_complete_spans[s][t] = input(cg, -std::numeric_limits<float>::infinity());
			}

			// 2) Right complete item.
			list.clear();
			neg_inf = false;
			number = false;
			for (int u = t + 1; u < slen; ++u) {
				int arc_index = index_arcs[u][t];
				if (arc_index >= 0) {
					if (neg_inf && flag_ocs[u][s])
						continue;
					list.push_back(outside_complete_spans[u][s] + inside_incomplete_spans[arc_index]);
					if (flag_ocs[u][s])
						neg_inf = true;
					else
						number = true;
				}
			}
			for (int u = 0; u < s; ++u) {
				if (u == 0 && s > 1) continue;
				if (u == 0) {
					// Must have s = 1.
					int right_arc_index = index_arcs[u][t];
					if (right_arc_index >= 0) {
						if (neg_inf && flag_ois[right_arc_index])
							continue;
						list.push_back(outside_incomplete_spans[right_arc_index] + scores[right_arc_index]);
						if (flag_ois[right_arc_index])
							neg_inf = true;
						else
							number = true;
					}
				} else {
					int left_arc_index = index_arcs[t][u];
					int right_arc_index = index_arcs[u][t];
					if (right_arc_index >= 0) {
						if (neg_inf && (flag_ics[u][s - 1] || flag_ois[right_arc_index]))
							continue;
						list.push_back(outside_incomplete_spans[right_arc_index]
						+ inside_complete_spans[u][s - 1] + scores[right_arc_index]);
						if (flag_ics[u][s - 1] || flag_ois[right_arc_index])
							neg_inf = true;
						else
							number = true;
					}
					if (left_arc_index >= 0) {
						if (neg_inf && (flag_ics[u][s - 1] || flag_ois[left_arc_index]))
							continue;
						list.push_back(outside_incomplete_spans[left_arc_index]
						               + inside_complete_spans[u][s - 1]
						               + scores[left_arc_index]);
						if (flag_ics[u][s - 1] || flag_ois[left_arc_index])
							neg_inf = true;
						else
							number = true;
					}
				}
			}
			if (list.size() > 0) {
				if (!number)
					flag_ocs[t][s] = true;
				outside_complete_spans[t][s] = logsumexp(list);
			}
			else {
				flag_ocs[t][s] = true;
				outside_complete_spans[t][s] = input(cg, -std::numeric_limits<float>::infinity());
			}
			if (t == 2 && s == 4)
				LOG(INFO) << outside_complete_spans[t][s].value();

			// Second, create incomplete items.
			int left_arc_index = index_arcs[t][s];
			int right_arc_index = index_arcs[s][t];
			if (right_arc_index >= 0) {
				vector<Expression> list;
				bool neg_inf = false;
				bool number = false;
				for (int u = t; u < slen; ++u) {
					if (neg_inf && (flag_ocs[s][u] || flag_ics[t][u]))
						continue;
					list.push_back(outside_complete_spans[s][u] + inside_complete_spans[t][u]);
					if (flag_ocs[s][u] || flag_ics[t][u])
						neg_inf = true;
					else
						number = true;
				}
				if (!number)
					flag_ois[right_arc_index] = true;
				outside_incomplete_spans[right_arc_index] = logsumexp(list);
			}
			if (left_arc_index >= 0) {
				vector<Expression> list;
				bool neg_inf = false;
				bool number = false;
				for (int u = 1; u <= s; ++u) {
					if (neg_inf && (flag_ocs[t][u] || flag_ics[s][u]))
						continue;
					list.push_back(outside_complete_spans[t][u] + inside_complete_spans[s][u]);
					if (flag_ocs[t][u] || flag_ics[s][u])
						neg_inf = true;
					else
						number = true;
				}
				if (!number)
					flag_ois[left_arc_index] = true;
				outside_incomplete_spans[left_arc_index] = logsumexp(list);
			}
		}
	}
}

void DependencyPruner::DecodeInsideOutside(Instance *instance, Parts *parts,
                                           const vector<Expression> &scores,
                                           vector<double> *predicted_output,
                                           Expression &entropy,
                                           ComputationGraph &cg) {
	int slen = static_cast<DependencyInstanceNumeric *>(instance)->size() - 1;
	auto dependency_parts = static_cast<DependencyParts *>(parts);

	int offset_arcs, num_arcs;
	dependency_parts->GetOffsetArc(&offset_arcs, &num_arcs);
	vector<DependencyPartArc *> arcs(num_arcs);
	vector<Expression> scores_arcs(num_arcs);
	for (int r = 0; r < num_arcs; ++r) {
		arcs[r] = static_cast<DependencyPartArc *>((*parts)[offset_arcs + r]);
		scores_arcs[r] = scores[offset_arcs + r];
	}

	Expression log_partition_function;
	vector<Expression> inside_incomplete_spans;
	vector<vector<Expression> > inside_complete_spans;
	vector<vector<bool> > flag_ics;
	vector<Expression> outside_incomplete_spans;
	vector<vector<Expression> > outside_complete_spans;

	RunEisnerInside(slen, arcs, scores_arcs, inside_incomplete_spans,
	                inside_complete_spans, flag_ics, log_partition_function, cg);

	RunEisnerOutside(slen, arcs, scores_arcs, inside_incomplete_spans,
	                 inside_complete_spans, flag_ics, outside_incomplete_spans,
	                 outside_complete_spans, cg);

	// Compute the marginals and entropy.
	predicted_output->resize(parts->size());
	entropy = log_partition_function;

	for (int r = 0; r < num_arcs; ++r) {
		int h = static_cast<DependencyPartArc *>((*parts)[offset_arcs +
		                                                  r])->head();
		int m = static_cast<DependencyPartArc *>((*parts)[offset_arcs +
		                                                  r])->modifier();
		Expression ex_value = exp(inside_incomplete_spans[r] +
		                          outside_incomplete_spans[r] -
		                          log_partition_function);
		float value = as_scalar(cg.incremental_forward(ex_value));
		if (value > 1.0) {
			if (!NEARLY_ZERO_TOL(value - 1.0, 1e-4)) {
				LOG(INFO) << "Marginals truncated to one (" << value << ")";
			}
		}

		(*predicted_output)[offset_arcs + r] = value;
		entropy = entropy -
				(*predicted_output)[offset_arcs + r] * scores[offset_arcs + r];
	}
	float e = as_scalar(cg.incremental_forward(entropy));
	if (e < 0.0) {
		if (!NEARLY_ZERO_TOL(e, 1e-6)) {
			LOG(INFO) << "Entropy truncated to zero (" << e << ")";
		}
		e = 0.0;
		entropy = input(cg, 0.0);
	}
}
