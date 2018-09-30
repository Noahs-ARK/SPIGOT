//
// Created by hpeng on 1/17/18.
//

#include "nodes-argmax-proj01.h"
#include <src/classifier/Part.h>
#include "dynet/nodes-impl-macros.h"
#include "dynet/tensor-eigen.h"

using namespace std;
namespace dynet {

// ************* ArgmaxProj01 *************
#ifndef __CUDACC__

	string ArgmaxProj01::as_string(const vector<string> &arg_names) const {
		ostringstream s;
		s << "ArgmaxProj01: (" << arg_names[0] << ")";
		return s.str();
	}

	Dim ArgmaxProj01::dim_forward(const vector<Dim> &xs) const {
		DYNET_ARG_CHECK(xs.size() == 3,
		                "Failed input count check in ArgmaxProj01");
		DYNET_ARG_CHECK(xs[0].nd <= 2,
		                "Bad input dimensions in ArgmaxProj01, must be 2 or fewer: "
				                << xs);
		return xs[0];
	}

	int
	ArgmaxProj01::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const {

		DYNET_ASSERT(false, "not implemented");
	}

	std::vector<int>
	ArgmaxProj01::autobatch_concat(const ComputationGraph &cg) const {
		return vector<int>(1, 1);
	}

#endif
	// x[0]: score
	// x[1]: pred
	// x[2]: 1 - pred
	template<class MyDevice>
	void ArgmaxProj01::forward_dev_impl(const MyDevice &dev,
	                                  const vector<const Tensor *> &xs,
	                                  Tensor &fx) const {
		// this is a fake foward prop.
		// just copying the results solved outside into the graph
		// this avoids potential mess caused by calling ArgmaxProj01 inside dynet cg
		DYNET_ASSERT(xs.size() == 3,
		             "Failed dimension check in ArgmaxProj01::forward");
		if (xs[0]->d.bd == 1) {
			tvec(fx).device(*dev.edevice) = tvec(*xs[1]);
		} else {
			DYNET_ASSERT(false, "Mini-batch support not implemented");
		}
	}

	// y_1 = relu(x + p) - p
	// y_2 = -relu(1-p-y_1) + 1-p
	template<class MyDevice>
	void ArgmaxProj01::backward_dev_impl(const MyDevice &dev,
	                                   const vector<const Tensor *> &xs,
	                                   const Tensor &fx,
	                                   const Tensor &dEdf,
	                                   unsigned i,
	                                   Tensor &dEdxi) const {
		DYNET_ASSERT(i == 0, "Failed dimension check in ArgmaxProj01::backward");
		DYNET_ASSERT(xs[1]->d.bd == xs[0]->d.bd,
		             "Failed dimension check in ArgmaxProj01::backward");
		DYNET_ASSERT(xs[0]->d.bd == 1,
		             "Failed dimension check in ArgmaxProj01::backward");
		if (dEdxi.d.bd == 1 && (dEdf.d.bd == xs[1]->d.bd)) {
			Project01(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
		} else {
			DYNET_ASSERT(false, "Mini-batch support not implemented");
		}
	}

	DYNET_NODE_INST_DEV_IMPL(ArgmaxProj01)
}

