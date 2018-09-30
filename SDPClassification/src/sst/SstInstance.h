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

#ifndef SSTINSTANCE_H_
#define SSTINSTANCE_H_

#include <string>
#include <vector>
#include <iostream>
#include "glog/logging.h"
#include "Instance.h"

using namespace std;

class SstInstance : public Instance {
public:
    SstInstance() {};

    virtual ~SstInstance() {};

    Instance *Copy() {
        SstInstance *instance = new SstInstance();
        instance->Initialize(name_, forms_,
                             lemmas_, cpostags_,
                             postags_, label_);
        return static_cast<Instance *>(instance);
    }

    void Initialize(const string &name,
                    const vector<string> &forms,
                    const vector<string> &lemmas,
                    const vector<string> &cpos,
                    const vector<string> &pos,
                    const string &label);

    const string &GetName() { return name_; }

	int size() {
		return forms_.size();
	};

    const string &GetForm(int i) {
	    return forms_[i];
    }

    const string &GetLemma(int i) {
	    return lemmas_[i];
    }

    const string &GetCoarsePosTag(int i) {
	    return cpostags_[i];
    };

    const string &GetPosTag(int i) {
	    return postags_[i];
    };

    const string &GetLabel() { return label_; }

protected:
    string name_;
    vector<string> forms_;
    vector<string> lemmas_;
    vector<string> cpostags_;
    vector<string> postags_;
    string label_;
};

#endif /* SSTINSTANCE_H_*/
