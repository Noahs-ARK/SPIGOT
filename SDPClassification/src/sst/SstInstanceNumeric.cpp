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

#include <algorithm>
#include "SstInstanceNumeric.h"

using namespace std;

void SstInstanceNumeric::Initialize(const SstDictionary &dictionary,
                                    SstInstance *instance) {

    bool form_case_sensitive = FLAGS_form_case_sensitive;
    int slen = instance->size();
    int i, id;

    Clear();
    form_ids_.resize(slen);
    form_lower_ids_.resize(slen);
    lemma_ids_.resize(slen);
    pos_ids_.resize(slen);
    cpos_ids_.resize(slen);

    for (i = 0; i < slen; i++) {
        std::string form = instance->GetForm(i);
        std::string form_lower(form);
        transform(form_lower.begin(), form_lower.end(), form_lower.begin(),
                  ::tolower);
        if (!form_case_sensitive) form = form_lower;
        id = dictionary.GetFormId(form);
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        form_ids_[i] = id;

        id = dictionary.GetFormLowerId(form_lower);
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        form_lower_ids_[i] = id;

        id = dictionary.GetLemmaId(instance->GetLemma(i));
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        lemma_ids_[i] = id;

        id = dictionary.GetPosTagId(instance->GetPosTag(i));
        CHECK_LT(id, 0xff);
        if (id < 0) id = TOKEN_UNKNOWN;
        pos_ids_[i] = id;

        id = dictionary.GetCoarsePosTagId(instance->GetCoarsePosTag(i));
        CHECK_LT(id, 0xff);
        if (id < 0) id = TOKEN_UNKNOWN;
        cpos_ids_[i] = id;
    }
    string label = instance->GetLabel();
	if (label == "0") {
		label_id_ = 0;
	} else if (label == "1") {
		label_id_ = 1;
	} else {
		CHECK(false);
	}
}
