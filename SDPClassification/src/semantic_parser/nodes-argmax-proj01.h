//
// Created by hpeng on 1/17/18.
//

#ifndef ARGMAXPROJ01_H
#define ARGMAXPROJ01_H

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"
#include "ProjectSimplex.h"

namespace dynet {

	struct ArgmaxProj01 : public Node {
		explicit ArgmaxProj01(const std::initializer_list<VariableIndex> &a) : Node(a) {}

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

		DYNET_NODE_DEFINE_DEV_IMPL()
	};

} // namespace dynet

#endif //ARGMAXPROJ01_H
