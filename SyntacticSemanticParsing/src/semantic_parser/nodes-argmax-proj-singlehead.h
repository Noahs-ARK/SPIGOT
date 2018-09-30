//
// Created by hpeng on 1/25/18.
//

#ifndef ARGMAXPROJSINGLEHEAD_H
#define ARGMAXPROJSINGLEHEAD_H


#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/tensor-eigen.h"
#include "ProjectSimplex.h"
#include "DependencyInstanceNumeric.h"
#include "DependencyPart.h"

namespace dynet {

	struct ArgmaxProjSingleHead : public Node {
		explicit ArgmaxProjSingleHead(const std::initializer_list<VariableIndex> &a,
		                              Instance *instance, Parts *parts) :
				Node(a), instance_(instance), parts_(parts) { }

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

		Instance *instance_;
		Parts *parts_;

		DYNET_NODE_DEFINE_DEV_IMPL()
	};

} // namespace dynet

#endif //ARGMAXPROJSINGLEHEAD_H
