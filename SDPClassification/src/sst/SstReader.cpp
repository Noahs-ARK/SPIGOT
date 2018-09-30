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

#include <SemanticReader.h>
#include "SstReader.h"
#include "Utils.h"

using namespace std;

Instance *SstReader::GetNext() {
    string name = "";
    vector<vector<string> > sent_fields;
    string line;
    if (is_.is_open()) {
        while (!is_.eof()) {
            getline(is_, line);
            if (line.length() <= 0) break;
            if (0 == line.substr(0, 1).compare("#")) {
                //LOG(INFO) << line;
                if (name != "") {
                    name += "\n" + line;
                } else {
                    name = line;
                }
                continue; // Sentence ID.
            }
            vector<string> fields;
            StringSplit(line, "\t", &fields, true);
            sent_fields.push_back(fields);
        }
    }

    bool read_next_sentence = false;
    if (!is_.eof()) read_next_sentence = true;

    int slen = sent_fields.size();

    // Convert to array of forms, lemmas2, etc.
    // Note: the first token is the root symbol.
    vector<string> forms(slen + 2);
    vector<string> lemmas(slen + 2);
    vector<string> cpos(slen + 2);
    vector<string> pos(slen + 2);
    string label;
    forms[0] = kStart; forms[slen + 1] = kEnd;
    lemmas[0] = kStart; lemmas[slen + 1] = kEnd;
    cpos[0] = kStart; cpos[slen + 1] = kEnd;
    pos[0] = kStart; pos[slen + 1] = kEnd;

    for (int i = 0;i < slen; ++ i) {
        const vector<string> &info = sent_fields[i];
        forms[i + 1] = info[1];
        lemmas[i + 1] = info[2];
        cpos[i + 1] = pos[i + 1] = info[3];
        label = info[4];
    }

    SstInstance *instance = nullptr;
    if (read_next_sentence && slen > 0) {
        instance = new SstInstance;
        instance->Initialize(name, forms, lemmas, cpos, pos, label);
    }

    return static_cast<Instance *>(instance);
}
