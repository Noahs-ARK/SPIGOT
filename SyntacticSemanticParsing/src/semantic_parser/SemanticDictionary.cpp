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

#include "SemanticDictionary.h"
#include "SemanticPipe.h"

// Special symbols.
//const string kPathUnknown = "_UNKNOWN_"; // Unknown path.

// Maximum alphabet sizes.
DEFINE_int32(role_cutoff, 30,
             "Ignore roles whose frequency is less than this.");

DEFINE_int32(relation_path_cutoff, 0,
             "Ignore relation paths whose frequency is less than this.");
DEFINE_int32(pos_path_cutoff, 0,
             "Ignore relation paths whose frequency is less than this.");
DEFINE_int32(num_frequent_role_pairs, 50,
             "Number of frequent role pairs to use in labeled sibling features");

void SemanticDictionary::CreatePredicateRoleDictionaries(SemanticReader *reader) {
	LOG(INFO) << "Creating predicate and role dictionaries...";

	// Initialize lemma predicates.
	int num_lemmas = token_dictionary_->GetNumLemmas();
	lemma_predicates_.resize(num_lemmas);

	vector<int> role_freqs;
	vector<int> predicate_freqs;

	string special_symbols[NUM_SPECIAL_PREDICATES];
	special_symbols[PREDICATE_UNKNOWN] = kPredicateUnknown;
	for (int i = 0; i < NUM_SPECIAL_PREDICATES; ++i) {
		predicate_alphabet_.Insert(special_symbols[i]);

		// Counts of special symbols are set to -1:
		predicate_freqs.push_back(-1);
	}
	int index_role_unk = role_alphabet_.Insert(kRoleUnknown);

	// Go through the corpus and build the predicate/roles dictionaries,
	// counting the frequencies.
	reader->Open(static_cast<SemanticPipe *> (pipe_)->GetSemanticOptions()->GetTrainingFilePath("semantic"));
	SemanticInstance *instance =
			static_cast<SemanticInstance *>(reader->GetNext());
	int instance_length = instance->size();
	Alphabet role_alphabet_tmp;
	while (instance != NULL) {
		for (int k = 0; k < instance->GetNumPredicates(); ++k) {
			int i = instance->GetPredicateIndex(k);
			const std::string lemma = instance->GetLemma(i);
			const std::string predicate_name = instance->GetPredicateName(k);

			// Get the lemma integer representation.
			int lemma_id = token_dictionary_->GetLemmaId(lemma);

			// If the lemma does not exist, the predicate will not be added.
			SemanticPredicate *predicate = NULL;
			if (lemma_id >= 0) {
				// Add predicate name to alphabet.
				int predicate_id =
						predicate_alphabet_.Insert(predicate_name);
				if (predicate_id >= predicate_freqs.size()) {
					CHECK_EQ(predicate_id, predicate_freqs.size());
					predicate_freqs.push_back(0);
				}
				++predicate_freqs[predicate_id];

				// Add predicate to the list of lemma predicates.
				std::vector<SemanticPredicate *> *predicates =
						&lemma_predicates_[lemma_id];
				for (int j = 0; j < predicates->size(); ++j) {
					if ((*predicates)[j]->id() == predicate_id) {
						predicate = (*predicates)[j];
					}
				}
				if (!predicate) {
					predicate = new SemanticPredicate(predicate_id);
					predicates->push_back(predicate);
				}
			}

			// Add semantic roles to alphabet.
			for (int l = 0; l < instance->GetNumArgumentsPredicate(k); ++l) {
				int role_id = role_alphabet_tmp.Insert(instance->GetArgumentRole(k, l));

				if (role_id >= role_freqs.size()) {
					CHECK_EQ(role_id, role_freqs.size());
					role_freqs.push_back(0);
				}
				++role_freqs[role_id];
				// Add this role to the predicate.
				if (predicate && !predicate->HasRole(role_id)) {
					predicate->InsertRole(role_id);
				}
			}
		}
		delete instance;
		instance = static_cast<SemanticInstance *>(reader->GetNext());
	}
	reader->Close();
	role_alphabet_tmp.StopGrowth();
	int role_cutoff = FLAGS_role_cutoff;
	role_alphabet_.AllowGrowth();
	for (Alphabet::iterator it = role_alphabet_tmp.begin();
	     it != role_alphabet_tmp.end(); ++it) {
		string role = it->first;
		int role_id = it->second;
		if (role_freqs[role_id] < role_cutoff) {
			LOG(INFO) << "Role "
			          << role
			          << "appears less than " << role_cutoff
			          << " times ("<<role_freqs[role_id]
			          << "). Mapping to unk."<<endl;
			continue;
		}
		role_id = role_alphabet_.Insert(role);
	}
	LOG(INFO) << "Role alphabet after pruning: " <<role_alphabet_.size()<<endl;
	//deterministic_roles_.assign(GetNumRoles(), true);
	role_alphabet_.StopGrowth();

	// Take care of the special "unknown" predicate.
	bool allow_unseen_predicates =
			static_cast<SemanticPipe *>(pipe_)->GetSemanticOptions()->
					allow_unseen_predicates();
	bool use_predicate_senses =
			static_cast<SemanticPipe *>(pipe_)->GetSemanticOptions()->
					use_predicate_senses();
	if (allow_unseen_predicates || !use_predicate_senses) {
		// 1) Add the predicate as the singleton list of lemma predicates for the
		// "unknown" lemma.
		std::vector<SemanticPredicate *> *predicates =
				&lemma_predicates_[TOKEN_UNKNOWN];
		CHECK_EQ(predicates->size(), 0);
		SemanticPredicate *predicate = new SemanticPredicate(PREDICATE_UNKNOWN);
		predicates->push_back(predicate);

		// 2) Add all possible roles to the special "unknown" predicate.
		for (int role_id = 0; role_id < role_alphabet_.size(); ++role_id) {
			if (!predicate->HasRole(role_id)) predicate->InsertRole(role_id);
		}
	}

	predicate_alphabet_.StopGrowth();

	CHECK_LT(predicate_alphabet_.size(), kMaxPredicateAlphabetSize);
	CHECK_LT(role_alphabet_.size(), kMaxRoleAlphabetSize);

	// Prepare alphabets for dependency paths (relations and POS).
	vector<int> relation_path_freqs;
	Alphabet relation_path_alphabet;
	vector<int> pos_path_freqs;
	Alphabet pos_path_alphabet;

	string special_path_symbols[NUM_SPECIAL_PATHS];
	special_path_symbols[PATH_UNKNOWN] = kPathUnknown;
	for (int i = 0; i < NUM_SPECIAL_PATHS; ++i) {
		relation_path_alphabet.Insert(special_path_symbols[i]);
		pos_path_alphabet.Insert(special_path_symbols[i]);

		// Counts of special symbols are set to -1:
		relation_path_freqs.push_back(-1);
		pos_path_freqs.push_back(-1);
	}

	// Go through the corpus and build the existing labels for:
	// - each head-modifier POS pair,
	// - each syntactic path (if available).
	// Keep also the maximum left/right arc lengths for each pair of POS tags.
	existing_roles_.clear();
	existing_roles_.resize(token_dictionary_->GetNumPosTags(),
	                       vector<vector<int> >(
			                       token_dictionary_->GetNumPosTags()));

	vector<vector<int> > existing_roles_with_relation_path(NUM_SPECIAL_PATHS,
	                                                       std::vector<int>(0));
	vector<int> role_pair_freqs(GetNumRoleBigramLabels(), 0);
	// Initialize every label as deterministic.
	deterministic_roles_.assign(GetNumRoles(), true);

	maximum_left_distances_.clear();
	maximum_left_distances_.resize(token_dictionary_->GetNumPosTags(),
	                               vector<int>(
			                               token_dictionary_->GetNumPosTags(), 0));

	maximum_right_distances_.clear();
	maximum_right_distances_.resize(token_dictionary_->GetNumPosTags(),
	                                vector<int>(
			                                token_dictionary_->GetNumPosTags(), 0));

	reader->Open(static_cast<SemanticPipe *> (pipe_)->GetSemanticOptions()->GetTrainingFilePath("semantic"));
	instance = static_cast<SemanticInstance *>(reader->GetNext());
	while (instance != NULL) {
		int instance_length = instance->size();
		for (int k = 0; k < instance->GetNumPredicates(); ++k) {
			int p = instance->GetPredicateIndex(k);
			const string &predicate_pos = instance->GetPosTag(p);
			int predicate_pos_id = token_dictionary_->GetPosTagId(predicate_pos);
			if (predicate_pos_id < 0) predicate_pos_id = TOKEN_UNKNOWN;

			// Add semantic roles to alphabet.
			for (int l = 0; l < instance->GetNumArgumentsPredicate(k); ++l) {
				int a = instance->GetArgumentIndex(k, l);
				const string &argument_pos = instance->GetPosTag(a);
				int argument_pos_id = token_dictionary_->GetPosTagId(argument_pos);
				if (argument_pos_id < 0) argument_pos_id = TOKEN_UNKNOWN;
				int role_id = role_alphabet_.Lookup(instance->GetArgumentRole(k, l));
				if (role_id < 0)
					role_id = index_role_unk;;
				//CHECK_GE(role_id, 0);

				// Look for possible role pairs.
				for (int m = l + 1; m < instance->GetNumArgumentsPredicate(k); ++m) {
					int other_role_id =
							role_alphabet_.Lookup(instance->GetArgumentRole(k, m));
					//CHECK_GE(other_role_id, 0);
					if (other_role_id < 0)
						other_role_id = index_role_unk;
					int bigram_label = GetRoleBigramLabel(role_id, other_role_id);
					CHECK_GE(bigram_label, 0);
					CHECK_LT(bigram_label, GetNumRoleBigramLabels());
					++role_pair_freqs[bigram_label];
					if (role_id == other_role_id) {
						// Role label is not deterministic.
						deterministic_roles_[role_id] = false;
					}
				}

				// Insert new role in the set of existing labels, if it is not there
				// already. NOTE: this is inefficient, maybe we should be using a
				// different data structure.
				vector<int> &roles = existing_roles_[predicate_pos_id][argument_pos_id];
				int j;
				for (j = 0; j < roles.size(); ++j) {
					if (roles[j] == role_id) break;
				}
				if (j == roles.size()) roles.push_back(role_id);

				// Update the maximum distances if necessary.
				if (p < a) {
					// Right attachment.
					if (a - p >
					    maximum_right_distances_[predicate_pos_id][argument_pos_id]) {
						maximum_right_distances_[predicate_pos_id][argument_pos_id] = a - p;
					}
				} else {
					// Left attachment (or self-loop). TODO(atm): treat self-loops differently?
					if (p - a >
					    maximum_left_distances_[predicate_pos_id][argument_pos_id]) {
						maximum_left_distances_[predicate_pos_id][argument_pos_id] = p - a;
					}
				}

				// Compute the syntactic path between the predicate and the argument and
				// add it to the dictionary.
				string relation_path;
				string pos_path;
				ComputeDependencyPath(instance, p, a, &relation_path, &pos_path);
				int relation_path_id = relation_path_alphabet.Insert(relation_path);
				if (relation_path_id >= relation_path_freqs.size()) {
					CHECK_EQ(relation_path_id, relation_path_freqs.size());
					relation_path_freqs.push_back(0);
				}
				++relation_path_freqs[relation_path_id];
				int pos_path_id = pos_path_alphabet.Insert(pos_path);
				if (pos_path_id >= pos_path_freqs.size()) {
					CHECK_EQ(pos_path_id, pos_path_freqs.size());
					pos_path_freqs.push_back(0);
				}
				++pos_path_freqs[pos_path_id];

				// Insert new role in the set of existing labels with this relation
				// path, if it is not there already. NOTE: this is inefficient, maybe we
				// should be using a different data structure.
				if (relation_path_id >= existing_roles_with_relation_path.size()) {
					existing_roles_with_relation_path.resize(relation_path_id + 1);
				}
				vector<int> &path_roles =
						existing_roles_with_relation_path[relation_path_id];
				for (j = 0; j < path_roles.size(); ++j) {
					if (path_roles[j] == role_id) break;
				}
				if (j == path_roles.size()) path_roles.push_back(role_id);
			}
		}
		delete instance;
		instance = static_cast<SemanticInstance *>(reader->GetNext());
	}
	reader->Close();

	// Now adjust the cutoffs if necessary.
	int relation_path_cutoff = FLAGS_relation_path_cutoff;
	while (true) {
		relation_path_alphabet_.clear();
		existing_roles_with_relation_path_.clear();
		CHECK_GE(existing_roles_with_relation_path.size(), NUM_SPECIAL_PATHS);
		for (int i = 0; i < NUM_SPECIAL_PATHS; ++i) {
			int relation_path_id =
					relation_path_alphabet_.Insert(special_path_symbols[i]);
			vector<int> &roles = existing_roles_with_relation_path[i];
			CHECK_EQ(roles.size(), 0);
			existing_roles_with_relation_path_.push_back(roles);
			//existing_roles_with_relation_path_.push_back(vector<int>(0));
		}
		for (Alphabet::iterator iter = relation_path_alphabet.begin();
		     iter != relation_path_alphabet.end();
		     ++iter) {
			if (relation_path_freqs[iter->second] > relation_path_cutoff) {
				int relation_path_id = relation_path_alphabet_.Insert(iter->first);
				vector<int> &roles = existing_roles_with_relation_path[iter->second];
				existing_roles_with_relation_path_.push_back(roles);
			}
		}
		CHECK_EQ(relation_path_alphabet_.size(),
		         existing_roles_with_relation_path_.size());
		if (relation_path_alphabet_.size() < kMaxRelationPathAlphabetSize) break;
		++relation_path_cutoff;
		CHECK(false); // For now, disallowed: this would mess up the relation path filter.
		LOG(INFO) << "Incrementing relation path cutoff to "
		          << relation_path_cutoff << "...";
	}

	int pos_path_cutoff = FLAGS_pos_path_cutoff;
	while (true) {
		pos_path_alphabet_.clear();
		for (int i = 0; i < NUM_SPECIAL_PATHS; ++i) {
			pos_path_alphabet_.Insert(special_path_symbols[i]);
		}
		for (Alphabet::iterator iter = pos_path_alphabet.begin();
		     iter != pos_path_alphabet.end();
		     ++iter) {
			if (pos_path_freqs[iter->second] > pos_path_cutoff) {
				pos_path_alphabet_.Insert(iter->first);
			}
		}
		if (pos_path_alphabet_.size() < kMaxPosPathAlphabetSize) break;
		++pos_path_cutoff;
		LOG(INFO) << "Incrementing pos path cutoff to "
		          << pos_path_cutoff << "...";
	}

	relation_path_alphabet_.StopGrowth();
	pos_path_alphabet_.StopGrowth();

	CHECK_LT(relation_path_alphabet_.size(), kMaxRelationPathAlphabetSize);
	CHECK_LT(pos_path_alphabet_.size(), kMaxPosPathAlphabetSize);

	// Compute the set of most frequent role pairs.
	vector<pair<int, int> > freqs_pairs;
	for (int k = 0; k < role_pair_freqs.size(); ++k) {
		freqs_pairs.push_back(pair<int, int>(-role_pair_freqs[k], k));
	}
	sort(freqs_pairs.begin(), freqs_pairs.end());
	frequent_role_pairs_.clear();
	for (int k = 0;
	     k < FLAGS_num_frequent_role_pairs && k < freqs_pairs.size();
	     ++k) {
		frequent_role_pairs_.insert(freqs_pairs[k].second);
	}

	// Display information about frequent role pairs.
	for (Alphabet::iterator it = role_alphabet_.begin();
	     it != role_alphabet_.end(); ++it) {
		string first_role = it->first;
		int first_role_id = it->second;
		for (Alphabet::iterator it2 = role_alphabet_.begin();
		     it2 != role_alphabet_.end(); ++it2) {
			string second_role = it2->first;
			int second_role_id = it2->second;
			if (IsFrequentRolePair(first_role_id, second_role_id)) {
				LOG(INFO) << "Frequent role pair: "
				          << first_role << " " << second_role;
			}
		}
	}

	// Display information about deterministic roles.
	int num_deterministic_roles = 0;
	for (Alphabet::iterator it = role_alphabet_.begin();
	     it != role_alphabet_.end(); ++it) {
		string role = it->first;
		int role_id = it->second;
		if (IsRoleDeterministic(role_id)) {
			LOG(INFO) << "Deterministic role: "
			          << role;
			++num_deterministic_roles;
		}
	}
	LOG(INFO) << num_deterministic_roles << " out of "
	          << GetNumRoles() << " roles are deterministic.";

#if 0
	// Go again through the corpus to build the existing labels for:
    // - each syntactic path (if available).
    reader->Open(pipe_->GetOptions()->GetTrainingFilePath());
    instance = static_cast<SemanticInstance*>(reader->GetNext());
    while (instance != NULL) {
      int instance_length = instance->size();
      for (int k = 0; k < instance->GetNumPredicates(); ++k) {
        int p = instance->GetPredicateIndex(k);
        const string &predicate_pos = instance->GetPosTag(p);
        int predicate_pos_id = token_dictionary_->GetPosTagId(predicate_pos);
        if (predicate_pos_id < 0) predicate_pos_id = TOKEN_UNKNOWN;

        // Add semantic roles to alphabet.
        for (int l = 0; l < instance->GetNumArgumentsPredicate(k); ++l) {
          int a = instance->GetArgumentIndex(k, l);
          const string &argument_pos = instance->GetPosTag(a);
          int argument_pos_id = token_dictionary_->GetPosTagId(argument_pos);
          if (argument_pos_id < 0) argument_pos_id = TOKEN_UNKNOWN;
          int role_id = role_alphabet_.Lookup(instance->GetArgumentRole(k, l));
          CHECK_GE(role_id, 0);

          // Compute the syntactic path between the predicate and the argument and
          // add it to the dictionary.
          string relation_path;
          ComputeDependencyPath(instance, p, a, &relation_path, &pos_path);
          int relation_path_id = relation_path_alphabet_.Get(relation_path);
          CHECK_GE(relation_path_id, 0);

          // Insert new role in the set of existing labels with this relation
          // path, if it is not there already. NOTE: this is inefficient, maybe we
          // should be using a different data structure.
          if (relation_path_id >= existing_roles_with_relation_path_.size()) {
            existing_roles_with_relation_path_.resize(relation_path_id + 1);
          }
          vector<int> &path_roles = existing_roles_with_relation_path_[relation_path_id];
          for (j = 0; j < path_roles.size(); ++j) {
            if (path_roles[j] == role_id) break;
          }
          if (j == path_roles.size()) path_roles.push_back(role_id);
        }
      }
      delete instance;
      instance = static_cast<SemanticInstance*>(reader->GetNext());
    }
    reader->Close();
#endif
	BuildPredicateRoleNames();
	// Show corpus statistics.
	LOG(INFO) << "Number of predicates: " << predicate_alphabet_.size();
	LOG(INFO) << "Number of roles: " << role_alphabet_.size();
	LOG(INFO) << "Number of relation paths: " << relation_path_alphabet_.size();
	LOG(INFO) << "Number of POS paths: " << pos_path_alphabet_.size();
}

void SemanticDictionary::ComputeDependencyPath(SemanticInstance *instance,
                                               int p, int a,
                                               string *relation_path,
                                               string *pos_path) const {
	const vector<int> &heads = instance->GetHeads();
	vector<string> relations_up;
	vector<string> relations_down;
	vector<string> pos_up;
	vector<string> pos_down;

	int ancestor = FindLowestCommonAncestor(heads, p, a);
	int h = p;
	while (ancestor != h) {
		relations_up.push_back(instance->GetDependencyRelation(h));
		pos_up.push_back(instance->GetPosTag(h));
		h = heads[h];
	}
	h = a;
	while (ancestor != h) {
		relations_down.push_back(instance->GetDependencyRelation(h));
		pos_down.push_back(instance->GetPosTag(h));
		h = heads[h];
	}

	relation_path->clear();
	pos_path->clear();
	for (int i = 0; i < relations_up.size(); ++i) {
		*relation_path += relations_up[i] + "^";
		*pos_path += pos_up[i] + "^";
	}
	*pos_path += instance->GetPosTag(ancestor);
	for (int i = relations_down.size() - 1; i >= 0; --i) {
		*relation_path += relations_down[i] + "!";
		*pos_path += pos_down[i] + "!";
	}
}

int SemanticDictionary::FindLowestCommonAncestor(const vector<int> &heads,
                                                 int p, int a) const {
	vector<bool> is_ancestor(heads.size(), false);
	int h = p;
	// 0 is the root and is a common ancestor.
	while (h != 0) {
		is_ancestor[h] = true;
		h = heads[h];
	}
	h = a;
	while (h != 0) {
		if (is_ancestor[h]) return h;
		h = heads[h];
	}
	return 0;
}

void SemanticTokenDictionary::Initialize(SemanticReader *semantic_reader) {
	SetTokenDictionaryFlagValues();
	LOG(INFO) << "Creating token dictionary...";

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
		form_alphabet.Insert(special_symbols[i]);
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
	}


	semantic_reader->Open(static_cast<SemanticOptions *>(
			                      pipe_->GetOptions())->GetTrainingFilePath("semantic"));
	auto semantic_instance =
			static_cast<SemanticInstance *>(semantic_reader->GetNext());
	while (semantic_instance != NULL) {
		int instance_length = semantic_instance->size();
		for (int i = 0; i < instance_length; ++i) {
			int id;

			// Add form to alphabet.
			std::string form = semantic_instance->GetForm(i);
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

			// Add lower-case form to alphabet.
			id = form_lower_alphabet.Insert(form_lower);
			if (id >= form_lower_freqs.size()) {
				CHECK_EQ(id, form_lower_freqs.size());
				form_lower_freqs.push_back(0);
			}
			++form_lower_freqs[id];

			// Add lemma to alphabet.
			std::string lemma = semantic_instance->GetLemma(i);
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
			id = pos_alphabet.Insert(semantic_instance->GetPosTag(i));
			if (id >= pos_freqs.size()) {
				CHECK_EQ(id, pos_freqs.size());
				pos_freqs.push_back(0);
//				LOG(INFO) << semantic_instance->GetPosTag(i);
			}
			++pos_freqs[id];

			// Add CPOS to alphabet.
			id = cpos_alphabet.Insert(semantic_instance->GetCoarsePosTag(i));
			if (id >= cpos_freqs.size()) {
				CHECK_EQ(id, cpos_freqs.size());
				cpos_freqs.push_back(0);
			}
			++cpos_freqs[id];
		}
		delete semantic_instance;
		semantic_instance = static_cast<SemanticInstance *>(semantic_reader->GetNext());
	}
	semantic_reader->Close();

	{
		ifstream in(static_cast<SemanticPipe *> (pipe_)->GetSemanticOptions()
				            ->GetPretrainedEmbeddingFilePath());

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
			++ form_freqs[id];

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
			form_alphabet_.Insert(special_symbols[i]);
		}
		for (Alphabet::iterator iter = form_alphabet.begin();
		     iter != form_alphabet.end();
		     ++iter) {
			if (form_freqs[iter->second] > form_cutoff) {
				form_alphabet_.Insert(iter->first);
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

	LOG(INFO) << "Number of forms: " << form_alphabet_.size() << endl
	          << "Number of lower-case forms: " << form_lower_alphabet_.size() << endl
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
