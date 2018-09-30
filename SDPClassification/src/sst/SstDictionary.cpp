//
// Created by hpeng on 2/8/18.
//

#include "SstDictionary.h"
#include "SemanticPipe.h"

void SstDictionary::Initialize(SstReader *reader, unordered_map<int, int> *form_count_) {
	SetTokenDictionaryFlagValues();
	auto semantic_options = dynamic_cast<SemanticOptions *>(pipe_->GetOptions());
	LOG(INFO) << "Creating token dictionary...";
	unordered_map<int, int> *form_count = new unordered_map<int, int>();
	vector<int> form_freqs;
	vector<int> form_lower_freqs;
	vector<int> lemma_freqs;
	vector<int> pos_freqs;
	vector<int> cpos_freqs;

	Alphabet form_alphabet;
	Alphabet form_lower_alphabet;
	Alphabet lemma_alphabet;
	Alphabet pos_alphabet;
	Alphabet cpos_alphabet;

	string special_symbols[NUM_SPECIAL_TOKENS];
	special_symbols[TOKEN_UNKNOWN] = kTokenUnknown;
	special_symbols[TOKEN_START] = kTokenStart;
	special_symbols[TOKEN_STOP] = kTokenStop;

	for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
		prefix_alphabet_.Insert(special_symbols[i]);
		suffix_alphabet_.Insert(special_symbols[i]);
		int id = form_alphabet.Insert(special_symbols[i]);
		form_lower_alphabet.Insert(special_symbols[i]);
		lemma_alphabet.Insert(special_symbols[i]);
		pos_alphabet.Insert(special_symbols[i]);
		cpos_alphabet.Insert(special_symbols[i]);

		// Counts of special symbols are set to -1:
		form_freqs.push_back(-1);
		form_lower_freqs.push_back(-1);
		lemma_freqs.push_back(-1);
		pos_freqs.push_back(-1);
		cpos_freqs.push_back(-1);
		(*form_count)[id] = 0;
	}

	reader->Open(semantic_options->GetTrainingFilePath("classification"));
	auto classification_instance =
			dynamic_cast<SstInstance *>(reader->GetNext());
	while (classification_instance != nullptr) {
		int instance_length = classification_instance->size();
		for (int i = 0; i < instance_length; ++i) {
			int id;

			// Add form to alphabet.
			std::string form = classification_instance->GetForm(i);
			std::string form_lower(form);
			transform(form_lower.begin(), form_lower.end(),
			          form_lower.begin(), ::tolower);
			if (!form_case_sensitive) form = form_lower;
			id = form_alphabet.Insert(form);
			if (id >= form_freqs.size()) {
				CHECK_EQ(id, form_freqs.size());
				form_freqs.push_back(0);
			}
			++form_freqs[id];
			(*form_count)[id] = 0;

//			 Add lower-case form to alphabet.
			id = form_lower_alphabet.Insert(form_lower);
			if (id >= form_lower_freqs.size()) {
				CHECK_EQ(id, form_lower_freqs.size());
				form_lower_freqs.push_back(0);
			}
			++form_lower_freqs[id];

			// Add lemma to alphabet.
			std::string lemma = classification_instance->GetLemma(i);
			transform(lemma.begin(), lemma.end(),
			          lemma.begin(), ::tolower);
			id = lemma_alphabet.Insert(lemma);
			if (id >= lemma_freqs.size()) {
				CHECK_EQ(id, lemma_freqs.size());
				lemma_freqs.push_back(0);
			}
			++lemma_freqs[id];

			// Add prefix/suffix to alphabet.
			// TODO: add varying lengths.
			string prefix = form.substr(0, prefix_length);
			id = prefix_alphabet_.Insert(prefix);
			int start = form.length() - suffix_length;
			if (start < 0) start = 0;
			string suffix = form.substr(start, suffix_length);
			id = suffix_alphabet_.Insert(suffix);

			// Add POS to alphabet.
			id = pos_alphabet.Insert(classification_instance->GetPosTag(i));
			if (id >= pos_freqs.size()) {
				CHECK_EQ(id, pos_freqs.size());
				pos_freqs.push_back(0);
			}
			++pos_freqs[id];

			// Add CPOS to alphabet.
			id = cpos_alphabet.Insert(classification_instance->GetCoarsePosTag(i));
			if (id >= cpos_freqs.size()) {
				CHECK_EQ(id, cpos_freqs.size());
				cpos_freqs.push_back(0);
			}
			++cpos_freqs[id];
		}
		delete classification_instance;
		classification_instance = dynamic_cast<SstInstance *>(reader->GetNext());
	}
	reader->Close();

	{
		ifstream in(semantic_options->GetEmbeddingFilePath("classification"));

		if (!in.is_open()) {
			cerr << "Pretrained embeddings FILE NOT FOUND!" << endl;
		}
		string line;
		getline(in, line);
		string form;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> form;
			std::string form_lower(form);
			transform(form_lower.begin(), form_lower.end(),
			          form_lower.begin(), ::tolower);
			if (!form_case_sensitive) form = form_lower;
			int id = form_alphabet.Insert(form);
			if (id >= form_freqs.size()) {
				CHECK_EQ(id, form_freqs.size());
				form_freqs.push_back(0);
			}
			++form_freqs[id];
			(*form_count)[id] = 1;

			id = form_lower_alphabet.Insert(form);
			if (id >= form_lower_freqs.size()) {
				CHECK_EQ(id, form_lower_freqs.size());
				form_lower_freqs.push_back(0);
			}
			++form_lower_freqs[id];
		}
		in.close();
	}


	// Now adjust the cutoffs if necessary.
	while (true) {
		form_alphabet_.clear();
		for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
			int id = form_alphabet_.Insert(special_symbols[i]);
			(*form_count_)[id] = 0;
		}
		for (Alphabet::iterator iter = form_alphabet.begin();
		     iter != form_alphabet.end();
		     ++iter) {
			if (form_freqs[iter->second] > form_cutoff) {
				int id = form_alphabet_.Insert(iter->first);
				(*form_count_)[id] = (*form_count)[iter->second];
			}
		}
		if (form_alphabet_.size() < kMaxFormAlphabetSize) break;
		++form_cutoff;
		LOG(INFO) << "Incrementing form cutoff to " << form_cutoff << "...";
	}

	while (true) {
		form_lower_alphabet_.clear();
		for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
			form_lower_alphabet_.Insert(special_symbols[i]);
		}
		for (Alphabet::iterator iter = form_lower_alphabet.begin();
		     iter != form_lower_alphabet.end();
		     ++iter) {
			if (form_lower_freqs[iter->second] > form_lower_cutoff) {
				form_lower_alphabet_.Insert(iter->first);
			}
		}
		if (form_lower_alphabet_.size() < kMaxFormAlphabetSize) break;
		++form_lower_cutoff;
		LOG(INFO) << "Incrementing lower-case form cutoff to "
		          << form_lower_cutoff << "...";
	}

	while (true) {
		lemma_alphabet_.clear();
		for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
			lemma_alphabet_.Insert(special_symbols[i]);
		}
		for (Alphabet::iterator iter = lemma_alphabet.begin();
		     iter != lemma_alphabet.end();
		     ++iter) {
			if (lemma_freqs[iter->second] > lemma_cutoff) {
				lemma_alphabet_.Insert(iter->first);
			}
		}
		if (lemma_alphabet_.size() < kMaxLemmaAlphabetSize) break;
		++lemma_cutoff;
		LOG(INFO) << "Incrementing lemma cutoff to " << lemma_cutoff << "...";
	}

	while (true) {
		pos_alphabet_.clear();
		for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
			pos_alphabet_.Insert(special_symbols[i]);
		}
		for (Alphabet::iterator iter = pos_alphabet.begin();
		     iter != pos_alphabet.end();
		     ++iter) {
			if (pos_freqs[iter->second] > pos_cutoff) {
				pos_alphabet_.Insert(iter->first);
			}
		}
		if (pos_alphabet_.size() < kMaxPosAlphabetSize) break;
		++pos_cutoff;
		LOG(INFO) << "Incrementing POS cutoff to " << pos_cutoff << "...";
	}

	while (true) {
		cpos_alphabet_.clear();
		for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
			cpos_alphabet_.Insert(special_symbols[i]);
		}
		for (Alphabet::iterator iter = cpos_alphabet.begin();
		     iter != cpos_alphabet.end();
		     ++iter) {
			if (cpos_freqs[iter->second] > cpos_cutoff) {
				cpos_alphabet_.Insert(iter->first);
			}
		}
		if (cpos_alphabet_.size() < kMaxCoarsePosAlphabetSize) break;
		++cpos_cutoff;
		LOG(INFO) << "Incrementing CPOS cutoff to " << cpos_cutoff << "...";
	}

	form_alphabet_.StopGrowth();
	form_lower_alphabet_.StopGrowth();
	lemma_alphabet_.StopGrowth();
	prefix_alphabet_.StopGrowth();
	suffix_alphabet_.StopGrowth();
	pos_alphabet_.StopGrowth();
	cpos_alphabet_.StopGrowth();
	delete form_count;

	LOG(INFO) << "Number of forms: " << form_alphabet_.size() << endl
	          << "Number of lower-case forms: " << form_lower_alphabet_.size()
	          << endl
	          << "Number of lemmas: " << lemma_alphabet_.size() << endl
	          << "Number of prefixes: " << prefix_alphabet_.size() << endl
	          << "Number of suffixes: " << suffix_alphabet_.size() << endl
	          << "Number of pos: " << pos_alphabet_.size() << endl
	          << "Number of cpos: " << cpos_alphabet_.size();

	CHECK_LT(form_alphabet_.size(), 0xfffff);
	CHECK_LT(form_lower_alphabet_.size(), 0xfffff);
	CHECK_LT(lemma_alphabet_.size(), 0xfffff);
	CHECK_LT(prefix_alphabet_.size(), 0xffff);
	CHECK_LT(suffix_alphabet_.size(), 0xffff);
	CHECK_LT(pos_alphabet_.size(), 0xff);
	CHECK_LT(cpos_alphabet_.size(), 0xff);

	// TODO: Remove this (only for debugging purposes).
	BuildNames();
}