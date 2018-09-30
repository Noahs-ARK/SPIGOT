#include "nodes-argmax-ste.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/tensor-eigen.h"

using namespace std;
namespace dynet {

// ************* ArgmaxSte *************
#ifndef __CUDACC__
    string ArgmaxSte::as_string(const vector<string> &arg_names) const {
        ostringstream s;
        s << "ArgmaxSte: ("<< arg_names[0] <<")";
        return s.str();
    }

    Dim ArgmaxSte::dim_forward(const vector<Dim> &xs) const {
        DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in ArgmaxSte");
        DYNET_ARG_CHECK(xs[0].nd <= 2, "Bad input dimensions in ArgmaxSte, must be 2 or fewer: " << xs);
        return xs[0];
    }

    int ArgmaxSte::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const {

        DYNET_ASSERT(false, "not implemented");
    }

    std::vector<int> ArgmaxSte::autobatch_concat(const ComputationGraph &cg) const {
        return vector<int>(1, 1);
    }

#endif

    // x[0]: score
	// x[1]: pred
    template<class MyDevice>
    void ArgmaxSte::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
        // this is a fake foward prop.
        // just copying the results solved outside into the graph
        // this avoids potential mess caused by calling ArgmaxSte inside dynet cg
        DYNET_ASSERT(xs.size() == 2, "Failed dimension check in ArgmaxSte::forward");
        if(xs[0]->d.bd == 1) {
            tvec(fx).device(*dev.edevice) = tvec(*xs[1]);
        } else {
            DYNET_ASSERT(false, "Mini-batch support not implemented");
        }
    }

    template<class MyDevice>
    void ArgmaxSte::backward_dev_impl(const MyDevice &dev,
                                const vector<const Tensor *> &xs,
                                const Tensor &fx,
                                const Tensor &dEdf,
                                unsigned i,
                                Tensor &dEdxi) const {
        DYNET_ASSERT(i == 0, "Failed dimension check in ArgmaxSte::backward");
        DYNET_ASSERT(xs[1]->d.bd == xs[0]->d.bd, "Failed dimension check in ArgmaxSte::backward");
        DYNET_ASSERT(xs[0]->d.bd == 1, "Failed dimension check in ArgmaxSte::backward");
        if(dEdxi.d.bd == 1 && (dEdf.d.bd == xs[1]->d.bd)) {
            tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
        } else {
            DYNET_ASSERT(false, "Mini-batch support not implemented");
        }
    }

    DYNET_NODE_INST_DEV_IMPL(ArgmaxSte)

}
