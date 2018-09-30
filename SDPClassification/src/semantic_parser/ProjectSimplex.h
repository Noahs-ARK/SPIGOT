//
// Created by hpeng on 1/20/18.
//

#ifndef PROJECTSIMPLEX_H
#define PROJECTSIMPLEX_H


#include <algorithm>
#include "ad3/GenericFactor.h"
#include "DependencyDecoder.h"
#include "dynet/tensor.h"

const dynet::real THRESHOLD = 5.0 / 32;

void ProjectSimplex(int slen, unsigned num_variables,
                    dynet::real *pred, dynet::real *gradient,
                    dynet::real *projected_gradient);

void ProjectSingleHeadSimplex(Instance *instance, Parts *parts,
                              unsigned num_variables,
                              dynet::real *pred, dynet::real *gradient,
                              dynet::real *projected_gradient);

int SolveCanonicalQpKnapsack(const vector<dynet::real> &lower_bounds,
                             const vector<dynet::real> &upper_bounds,
                             const vector<dynet::real> &weights,
                             dynet::real total_weight,
                             vector<dynet::real> &solution);

int ProjectOntoKnapsackConstraint(vector<dynet::real> &x,
                                  vector<dynet::real> &costs, int d,
                                  dynet::real budget);

void Project01(unsigned num_variables,
               dynet::real *pred, dynet::real *gradient,
               dynet::real *projected_gradient);

void ProjectSDP(Instance *instance, Parts *parts,
                unsigned num_variables,
                dynet::real *pred, dynet::real *gradient,
                dynet::real *projected_gradient);
#endif //PROJECTSIMPLEX_H
