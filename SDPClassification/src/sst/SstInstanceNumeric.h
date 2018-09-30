// Copyright (c) 2012-2015 Andre Martins
// All Rights Reserved.
//
// This file is part of TurboParser 2.3.
//
// TurboParser 2.3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.3.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SSTINSTANCENUMERIC_H_
#define SSTINSTANCENUMERIC_H_

#include <vector>
#include <string>
#include "Dictionary.h"
#include "SstInstance.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDictionary.h"
#include "DependencyInstanceNumeric.h"
#include "SstDictionary.h"

using namespace std;

class SstInstanceNumeric : public SemanticInstanceNumeric {
public:
    SstInstanceNumeric() {};

    virtual ~SstInstanceNumeric() { Clear(); };

    Instance *Copy() {
	    Instance *ret = new SemanticInstanceNumeric;
	    static_cast<SemanticInstanceNumeric *> (ret)
			    ->Set(form_ids_, form_lower_ids_,
			          lemma_ids_, pos_ids_, cpos_ids_, label_id_);
	    return ret;
    };

    void Clear() {
        form_ids_.clear();
        form_lower_ids_.clear();
        lemma_ids_.clear();
        pos_ids_.clear();
        cpos_ids_.clear();
    }

	int size() { return form_ids_.size(); };

    int GetLabel() { return label_id_; }

	int GetFormId(int i) { return form_ids_[i]; };

	int GetFormLowerId(int i) { return form_lower_ids_[i]; };

	int GetLemmaId(int i) { return lemma_ids_[i]; };

	int GetPosId(int i) { return pos_ids_[i]; };

	int GetCoarsePosId(int i) { return cpos_ids_[i]; };

    void Initialize(const SstDictionary &dictionary,
                    SstInstance *instance);

private:
    vector<int> form_ids_;
    vector<int> form_lower_ids_;
    vector<int> lemma_ids_;
    vector<int> pos_ids_;
    vector<int> cpos_ids_;
    int label_id_;
};

#endif /* SSTINSTANCENUMERIC_H_ */
