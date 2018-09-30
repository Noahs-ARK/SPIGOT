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

#ifndef SSTREADER_H_
#define SSTREADER_H_

#include "SstInstance.h"
#include "Reader.h"
#include "Options.h"
#include <fstream>

using namespace std;

// Note: this is made to derive from DependencyReader so that
// we don't need to change TokenDictionary.h which already
// builds all necessary dictionaries given a set of dependency
// instances.
class SstReader : public Reader {
public:
    SstReader() {
        options_ = NULL;
    }

    SstReader(Options *options) {
        options_ = options;
    }

    virtual ~SstReader() {}

public:

    Instance *GetNext();

protected:
    Options *options_;
};

#endif /* SSTREADER_H_ */
