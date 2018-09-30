#ifndef EXPR_H
#define EXPR_H

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "nodes-argmax-ste.h"
#include "nodes-argmax-proj.h"
#include "nodes-argmax-proj-singlehead.h"
#include "nodes-argmax-proj01.h"
#include <stdexcept>

namespace dynet {
    // argmax_ste(x, y) = y
    Expression argmax_ste(const Expression &x, const Expression &p);

	Expression argmax_proj(const Expression &x, const Expression &p,
	                       int slen);

	Expression argmax_proj_singlehead(const Expression &x, const Expression &p,
	                                  Instance *instance, Parts *parts);

	Expression argmax_proj01(const Expression &x, const Expression &p, const Expression &one_minus_p);
}  // namespace dynet

#endif
