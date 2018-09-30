//
// Created by hpeng on 1/20/18.
//

#include <src/parser/DependencyInstanceNumeric.h>
#include "ProjectSimplex.h"

int ProjectOntoKnapsackConstraint(vector<dynet::real> &x, vector<dynet::real> &costs,
                                  int d, dynet::real budget) {

	vector<dynet::real> lower_bounds(d);  // A.
	vector<dynet::real> upper_bounds(d);  // B.
	vector<dynet::real> weights(d);  // C.
	dynet::real total_weight;  // D.
	vector<dynet::real> solution(d);

	total_weight = budget;
	for (int i = 0; i < d; ++i) {
		lower_bounds[i] = -x[i] / costs[i];
		upper_bounds[i] = (1 - x[i]) / costs[i];
		weights[i] = costs[i] * costs[i];
		total_weight -= costs[i] * x[i];
	}

	SolveCanonicalQpKnapsack(lower_bounds, upper_bounds, weights,
	                         total_weight,
	                         solution);

	for (int i = 0; i < d; ++i) {
		x[i] += costs[i] * solution[i];
	}

	return 0;
}

int SolveCanonicalQpKnapsack(const vector<dynet::real> &lower_bounds,
                             const vector<dynet::real> &upper_bounds,
                             const vector<dynet::real> &weights,
                             dynet::real total_weight,
                             vector<dynet::real> &solution) {
	// Set dimension.
	int d = weights.size();

	// Sort lower and upper bounds and keep the sorted indices.
	vector<pair<dynet::real, int> > sorted_lower(d);
	vector<pair<dynet::real, int> > sorted_upper(d);
	for (int i = 0; i < d; ++i) {
		sorted_lower[i].first = lower_bounds[i];
		sorted_upper[i].first = upper_bounds[i];
		sorted_lower[i].second = i;
		sorted_upper[i].second = i;
	}
	sort(sorted_lower.begin(), sorted_lower.end());
	sort(sorted_upper.begin(), sorted_upper.end());

	dynet::real slackweight = 0.0;
	dynet::real tightsum = 0.0;
	for (int i = 0; i < d; ++i) {
		tightsum += lower_bounds[i] * weights[i];
	}

	int k = 0;
	int l = 0;
	int level = 0;
	dynet::real left, right = -std::numeric_limits<dynet::real>::infinity();
	bool found = false;
	dynet::real tau;
	int index_a, index_b;
	dynet::real val_a, val_b;

	while (k < d || l < d) {
		// Compute the estimate for tau.
		if (level != 0) {
			tau = (total_weight - tightsum) / slackweight;
			// cout << "tau = " << tau << endl;
		}

		if (k < d) {
			index_a = sorted_lower[k].second;
			val_a = sorted_lower[k].first;
		} else {
			val_a = std::numeric_limits<dynet::real>::infinity();
		}

		if (l < d) {
			index_b = sorted_upper[l].second;
			val_b = sorted_upper[l].first;
		} else {
			val_b = std::numeric_limits<dynet::real>::infinity();
		}

		left = right;
		if (val_a < val_b) {
			// Next value comes from the a-list.
			right = val_a;
		} else {
			// Next value comes from the b-list.
			left = right;
			right = val_b;
		}

		assert(level == 0 || tau >= left);
		if ((level == 0 && total_weight == tightsum) ||
		    (level != 0 && tau >= left && tau <= right)) {
			// Found the right split-point!
			found = true;
			break;
		}

		if (val_a < val_b) {
			tightsum -= lower_bounds[index_a] * weights[index_a];
			slackweight += weights[index_a];
			++level;
			++k;
		} else {
			tightsum += upper_bounds[index_b] * weights[index_b];
			slackweight -= weights[index_b];
			--level;
			++l;
		}
	}

	for (int i = 0; i < d; ++i) {
		solution[i] = 0.0;
	}
	if (!found) {
		left = right;
		right = std::numeric_limits<dynet::real>::infinity();
	}

	for (int i = 0; i < d; ++i) {
		if (lower_bounds[i] >= right) {
			solution[i] = lower_bounds[i];
		} else if (upper_bounds[i] <= left) {
			solution[i] = upper_bounds[i];
		} else {
			assert(found);
			assert(level != 0);
			solution[i] = tau;
		}
	}

	return 0;
}

void ProjectSimplex(int slen, unsigned num_variables,
                    dynet::real *pred, dynet::real *gradient,
                    dynet::real *projected_gradient) {
	vector<dynet::real> x(num_variables, 0.0);
	vector<dynet::real> costs(num_variables, 1.0);
	dynet::real l2_norm = 0.0;
	for (int i = 0;i < num_variables; ++ i) {
		l2_norm = l2_norm + (gradient[i] * gradient[i]);
	}
	l2_norm = sqrt(l2_norm);
	dynet::real gscale = 1.0;
	if (l2_norm > THRESHOLD) {
		gscale = THRESHOLD / l2_norm;
	}
	for (int i = 0;i < num_variables; ++ i) {
		x[i] = pred[i] - gradient[i] * gscale;
	}
	ProjectOntoKnapsackConstraint(x, costs, num_variables, slen - 1);
	for (int i = 0;i < num_variables; ++ i) {
		projected_gradient[i] = pred[i] - x[i];
	}
}

void ProjectSingleHeadSimplex(Instance *instance, Parts *parts,
                              unsigned num_variables,
                              dynet::real *pred, dynet::real *gradient,
                              dynet::real *projected_gradient) {
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	int slen = sentence->size() - 1;

	int offset_arcs, num_arcs;
	dependency_parts->GetOffsetArc(&offset_arcs, &num_arcs);

	vector<vector<int> > arcs_to_mod(slen);
	for (int i = 0; i < num_arcs; ++i) {
		int r = i + offset_arcs;
		auto arc = static_cast<DependencyPartArc *> ((*parts)[r]);
		int m = arc->modifier();
		arcs_to_mod[m].push_back(r);
	}

	dynet::real l2_norm = 0.0;
	for (int i = 0;i < num_variables; ++ i) {
		l2_norm = l2_norm + (gradient[i] * gradient[i]);
	}
	l2_norm = sqrt(l2_norm);
	dynet::real gscale = 1.0;
	if (l2_norm > THRESHOLD) {
		gscale = THRESHOLD / l2_norm;
	}
	vector<dynet::real> target(num_variables, 0.0);
	for (int i = 0;i < num_variables; ++ i) {
		target[i] = pred[i] - gradient[i] * gscale;
	}

	for (int i = 0;i < slen; ++ i) {
		vector<int> arcs_to_i = arcs_to_mod[i];
		int num_heads = arcs_to_i.size();
		if (num_heads == 0)
			continue;
		vector<dynet::real> x(num_heads, 0.0);
		for (int j = 0;j < num_heads; ++ j) {
			x[j] = target[arcs_to_i[j]];
		}
		vector<dynet::real> costs(num_heads, 1.0);
		ProjectOntoKnapsackConstraint(x, costs, num_heads, 1.0);
		for (int j = 0;j < num_heads; ++ j) {
			target[arcs_to_i[j]] = x[j];
		}
	}

	for (int i = 0;i < num_variables; ++ i) {
		projected_gradient[i] = pred[i] - target[i];
	}
}

void Project01(unsigned num_variables,
               dynet::real *pred, dynet::real *gradient,
               dynet::real *projected_gradient) {
	dynet::real l2_norm = 0.0;
	for (int i = 0;i < num_variables; ++ i) {
		l2_norm = l2_norm + (gradient[i] * gradient[i]);
	}
	l2_norm = sqrt(l2_norm);
	dynet::real gscale = 1.0;
	if (l2_norm > THRESHOLD) {
		gscale = THRESHOLD / l2_norm;
	}
	vector<dynet::real> x(num_variables, 0.0);
	for (int i = 0;i < num_variables; ++ i) {
		x[i] = pred[i] - gradient[i] * gscale;
		x[i] = min(x[i], dynet::real(1.0));
		x[i] = max(x[i], dynet::real(0.0));
	}

	for (int i = 0;i < num_variables; ++ i) {
		projected_gradient[i] = pred[i] - x[i];
	}
}