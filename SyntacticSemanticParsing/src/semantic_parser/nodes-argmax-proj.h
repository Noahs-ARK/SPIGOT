//
// Created by hpeng on 1/12/18.
//

#ifndef ARGMAX_PROJ_H
#define ARGMAX_PROJ_H


#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/tensor-eigen.h"
#include "ProjectSimplex.h"

namespace dynet {

	struct ArgmaxProj : public Node {
		explicit ArgmaxProj(const std::initializer_list<VariableIndex> &a,
		                    int slen) :
				Node(a), slen_(slen) {}

		virtual bool supports_multibatch() const override { return true; }

		virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;

		virtual std::vector<int> autobatch_concat(const ComputationGraph &cg) const override;

		virtual void autobatch_reshape(const ComputationGraph &cg,
		                               const std::vector<VariableIndex> &batch_ids,
		                               const std::vector<int> &concat,
		                               std::vector<const Tensor *> &xs,
		                               Tensor &fx) const override {
			autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
		}

		int slen_;

		DYNET_NODE_DEFINE_DEV_IMPL()
	};

} // namespace dynet

#endif //ARGMAX_PROJ_H
