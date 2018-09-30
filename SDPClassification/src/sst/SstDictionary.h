//
// Created by hpeng on 2/8/18.
//

#ifndef DIFF_ARGMAX_SSTDICTIONARY_H
#define DIFF_ARGMAX_SSTDICTIONARY_H

#include <unordered_map>
#include "SstReader.h"
#include "TokenDictionary.h"

class SstDictionary : public TokenDictionary {

public:

	SstDictionary() {}

	SstDictionary(Pipe *pipe) : pipe_(pipe) {}

	virtual ~SstDictionary() {};

	void Initialize(SstReader *reader, unordered_map<int, int> *form_count_);

protected:
	Pipe *pipe_;
};


#endif //DIFF_ARGMAX_SSTDICTIONARY_H
