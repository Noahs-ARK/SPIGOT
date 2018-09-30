//
// Created by hpeng on 1/25/18.
//

#include "nodes-argmax-proj-singlehead.h"

using namespace std;
namespace dynet {

// ************* ArgmaxProjSingleHead *************
#ifndef __CUDACC__

	string ArgmaxProjSingleHead::as_string(const vector<string> &arg_names) const {
		ostringstream s;
		s << "ArgmaxProjSingleHead: (" << arg_names[0] << ")";
		return s.str();
	}

	Dim ArgmaxProjSingleHead::dim_forward(const vector<Dim> &xs) const {
		DYNET_ARG_CHECK(xs.size() == 2,
		                "Failed input count check in ArgmaxProjSingleHead");
		DYNET_ARG_CHECK(xs[0].nd <= 2,
		                "Bad input dimensions in ArgmaxProjSingleHead, must be 2 or fewer: "
				                << xs);
		return xs[0];
	}

	int
	ArgmaxProjSingleHead::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const {

		DYNET_ASSERT(false, "not implemented");
	}

	std::vector<int>
	ArgmaxProjSingleHead::autobatch_concat(const ComputationGraph &cg) const {
		return vector<int>(1, 1);
	}

#endif
	// x[0]: score
	// x[1]: pred
	// x[2]: 1 - pred
	template<class MyDevice>
	void ArgmaxProjSingleHead::forward_dev_impl(const MyDevice &dev,
	                                  const vector<const Tensor *> &xs,
	                                  Tensor &fx) const {
		// this is a fake foward prop.
		// just copying the results solved outside into the graph
		// this avoids potential mess caused by calling ArgmaxProjSingleHead inside dynet cg
		DYNET_ASSERT(xs.size() == 3,
		             "Failed dimension check in ArgmaxProjSingleHead::forward");
		if (xs[0]->d.bd == 1) {
			tvec(fx).device(*dev.edevice) = tvec(*xs[1]);
		} else {
			DYNET_ASSERT(false, "Mini-batch support not implemented");
		}
	}

	template<class MyDevice>
	void ArgmaxProjSingleHead::backward_dev_impl(const MyDevice &dev,
	                                   const vector<const Tensor *> &xs,
	                                   const Tensor &fx,
	                                   const Tensor &dEdf,
	                                   unsigned i,
	                                   Tensor &dEdxi) const {
		DYNET_ASSERT(i == 0, "Failed dimension check in ArgmaxProjSingleHead::backward");
		DYNET_ASSERT(xs[1]->d.bd == xs[0]->d.bd,
		             "Failed dimension check in ArgmaxProjSingleHead::backward");
		DYNET_ASSERT(xs[0]->d.bd == 1,
		             "Failed dimension check in ArgmaxProjSingleHead::backward");

		if (dEdxi.d.bd == 1 && (dEdf.d.bd == xs[1]->d.bd)) {
			ProjectSingleHeadSimplex(instance_, parts_, fx.d.size(), fx.v,
			                         dEdf.v, dEdxi.v);
		} else {
			DYNET_ASSERT(false, "Mini-batch support not implemented");
		}
	}

	DYNET_NODE_INST_DEV_IMPL(ArgmaxProjSingleHead)
}
