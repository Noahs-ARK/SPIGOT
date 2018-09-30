#include "expr.h"

namespace dynet {

    using std::vector;

    Expression argmax_ste(const Expression &x, const Expression &p) {
        return Expression(x.pg, x.pg->add_function<ArgmaxSte>({x.i, p.i}));
    }

    Expression argmax_proj(const Expression &x, const Expression &p,
                           int slen) {
	    return Expression(x.pg, x.pg->add_function<ArgmaxProj> ({x.i, p.i}, slen));
    }

	Expression argmax_proj_singlehead(const Expression &x, const Expression &p,
	                                  Instance *instance, Parts *parts) {
		return Expression(x.pg,
		                  x.pg->add_function<ArgmaxProjSingleHead> ({x.i, p.i},
		                                                        instance, parts));
	}

    Expression argmax_proj01(const Expression &x, const Expression &p, const Expression &one_minus_p) {
        return Expression(x.pg, x.pg->add_function<ArgmaxProj01>({x.i, p.i, one_minus_p.i}));
    }
}  // namespace dynet
