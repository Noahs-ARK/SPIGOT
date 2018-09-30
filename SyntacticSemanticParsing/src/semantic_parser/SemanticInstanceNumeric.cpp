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

#include "SemanticInstanceNumeric.h"
#include "SemanticPipe.h"

using namespace std;

const int kUnknownPredicate = 0xffff;
const int kUnknownRole = 0xffff;
const int kUnknownRelationPath = 0xffff;
const int kUnknownPosPath = 0xffff;

void SemanticInstanceNumeric::Initialize(
        const SemanticDictionary &dictionary,
        SemanticInstance *instance) {

	SemanticOptions *options =
			static_cast<SemanticPipe *>(dictionary.GetPipe())->GetSemanticOptions();
    TokenDictionary *token_dictionary = dictionary.GetTokenDictionary();
    int length = instance->size();
    int i;
    int id;

    int prefix_length = FLAGS_prefix_length;
    int suffix_length = FLAGS_suffix_length;
    bool form_case_sensitive = FLAGS_form_case_sensitive;

    Clear();

    form_ids_.resize(length);
    form_lower_ids_.resize(length);
    lemma_ids_.resize(length);
    prefix_ids_.resize(length);
    suffix_ids_.resize(length);
    feats_ids_.resize(length);
    pos_ids_.resize(length);
    cpos_ids_.resize(length);
    is_noun_.resize(length);
    is_verb_.resize(length);
    is_punc_.resize(length);
    is_coord_.resize(length);
    heads_.resize(length);
    relations_.resize(length);

    for (i = 0; i < length; i++) {
        std::string form = instance->GetForm(i);
        std::string form_lower(form);
        transform(form_lower.begin(), form_lower.end(), form_lower.begin(),
                  ::tolower);
        if (!form_case_sensitive) form = form_lower;
        id = token_dictionary->GetFormId(form);
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        form_ids_[i] = id;

        id = token_dictionary->GetFormLowerId(form_lower);
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        form_lower_ids_[i] = id;

        id = token_dictionary->GetLemmaId(instance->GetLemma(i));
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        lemma_ids_[i] = id;

        std::string prefix = form.substr(0, prefix_length);
        id = token_dictionary->GetPrefixId(prefix);
        CHECK_LT(id, 0xffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        prefix_ids_[i] = id;

        int start = form.length() - suffix_length;
        if (start < 0) start = 0;
        std::string suffix = form.substr(start, suffix_length);
        id = token_dictionary->GetSuffixId(suffix);
        CHECK_LT(id, 0xffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        suffix_ids_[i] = id;

        id = token_dictionary->GetPosTagId(instance->GetPosTag(i));
        CHECK_LT(id, 0xff);
        if (id < 0) id = TOKEN_UNKNOWN;
        pos_ids_[i] = id;

        id = token_dictionary->GetCoarsePosTagId(instance->GetCoarsePosTag(i));
        CHECK_LT(id, 0xff);
        if (id < 0) id = TOKEN_UNKNOWN;
        cpos_ids_[i] = id;

        feats_ids_[i].resize(instance->GetNumMorphFeatures(i));
        for (int j = 0; j < instance->GetNumMorphFeatures(i); ++j) {
            id = token_dictionary->GetMorphFeatureId(instance->GetMorphFeature(i, j));
            CHECK_LT(id, 0xffff);
            if (id < 0) id = TOKEN_UNKNOWN;
            feats_ids_[i][j] = id;
        }

        //GetWordShape(instance->GetForm(i), &shapes_[i]);

        // Check whether the word is a noun, verb, punctuation or coordination.
        // Note: this depends on the POS tag string.
        // This procedure is taken from EGSTRA
        // (http://groups.csail.mit.edu/nlp/egstra/).
        is_noun_[i] = false;
        is_verb_[i] = false;
        is_punc_[i] = false;
        is_coord_[i] = false;

        const char *tag = instance->GetPosTag(i).c_str();
        if (tag[0] == 'v' || tag[0] == 'V') {
            is_verb_[i] = true;
        } else if (tag[0] == 'n' || tag[0] == 'N') {
            is_noun_[i] = true;
        } else if (strcmp(tag, "Punc") == 0 ||
                   strcmp(tag, "$,") == 0 ||
                   strcmp(tag, "$.") == 0 ||
                   strcmp(tag, "PUNC") == 0 ||
                   strcmp(tag, "punc") == 0 ||
                   strcmp(tag, "F") == 0 ||
                   strcmp(tag, "IK") == 0 ||
                   strcmp(tag, "XP") == 0 ||
                   strcmp(tag, ",") == 0 ||
                   strcmp(tag, ";") == 0) {
            is_punc_[i] = true;
        } else if (strcmp(tag, "Conj") == 0 ||
                   strcmp(tag, "KON") == 0 ||
                   strcmp(tag, "conj") == 0 ||
                   strcmp(tag, "Conjunction") == 0 ||
                   strcmp(tag, "CC") == 0 ||
                   strcmp(tag, "cc") == 0) {
            is_coord_[i] = true;
        }

        heads_[i] = instance->GetHead(i);
        relations_[i] = dictionary.GetDependencyDictionary()->GetLabelAlphabet().Lookup(
                instance->GetDependencyRelation(i));
    }

    int num_predicates = instance->GetNumPredicates();
    predicate_ids_.resize(num_predicates);
    predicate_indices_.resize(num_predicates);
    argument_role_ids_.resize(num_predicates);
    argument_indices_.resize(num_predicates);
    for (int k = 0; k < instance->GetNumPredicates(); k++) {
        int id = -1;
        if (options->use_predicate_senses()) {
            const string &name = instance->GetPredicateName(k);
            id = dictionary.GetPredicateAlphabet().Lookup(name);
            CHECK_LT(id, 0xffff);
            if (id < 0) id = kUnknownPredicate;
        }
        predicate_ids_[k] = id;
        predicate_indices_[k] = instance->GetPredicateIndex(k);

        int num_arguments = instance->GetNumArgumentsPredicate(k);
        argument_role_ids_[k].resize(num_arguments);
        argument_indices_[k].resize(num_arguments);
        for (int l = 0; l < num_arguments; ++l) {
            const string &name = instance->GetArgumentRole(k, l);
            id = dictionary.GetRoleAlphabet().Lookup(name);
            CHECK_LT(id, 0xffff);
            if (id < 0) {
                id = dictionary.GetRoleAlphabet().Lookup(kRoleUnknown);
            }
            CHECK_GE(id, 0);
            argument_role_ids_[k][l] = id;
            argument_indices_[k][l] = instance->GetArgumentIndex(k, l);
        }
    }

    ComputeDependencyInformation(dictionary, instance);

    BuildIndices();
}

void SemanticInstanceNumeric::ComputeDependencyInformation(
        const SemanticDictionary &dictionary,
        SemanticInstance *instance) {

    int instance_length = instance->size() - 1;
    modifiers_.resize(instance_length);
    left_siblings_.resize(instance_length);
    right_siblings_.resize(instance_length);

    // List of dependents, left and right siblings.
    for (int h = 0; h < instance_length; ++h) {
        modifiers_[h].clear();
        left_siblings_[h] = -1;
        right_siblings_[h] = -1;
    }
    for (int m = 1; m < instance_length; ++m) {
        int h = instance->GetHead(m);
        modifiers_[h].push_back(m);
    }
    for (int h = 0; h < instance_length; ++h) {
        for (int k = 0; k < modifiers_[h].size(); ++k) {
            int m = modifiers_[h][k];
            if (k > 0) left_siblings_[m] = modifiers_[h][k - 1];
            if (k + 1 < modifiers_[h].size()) right_siblings_[m] = modifiers_[h][k + 1];
        }
    }

    // Select passive/active voice.
    is_passive_voice_.assign(instance_length, false);
    for (int i = 0; i < instance_length; ++i) {
        is_passive_voice_[i] = ComputePassiveVoice(instance, i);
    }

    // Compute relation/pos paths.
    relation_path_ids_.resize(instance_length);
    pos_path_ids_.resize(instance_length);
    for (int p = 0; p < instance_length; ++p) {
        relation_path_ids_[p].assign(instance_length, -1);
        pos_path_ids_[p].assign(instance_length, -1);
        for (int a = 1; a < instance_length; ++a) {
            string relation_path;
            string pos_path;
            dictionary.ComputeDependencyPath(instance, p, a,
                                             &relation_path, &pos_path);
            int relation_path_id =
                    dictionary.GetRelationPathAlphabet().Lookup(relation_path);
            CHECK_LT(relation_path_id, 0xffff);
            if (relation_path_id < 0) relation_path_id = kUnknownRelationPath;
            int pos_path_id = dictionary.GetPosPathAlphabet().Lookup(pos_path);
            CHECK_LT(pos_path_id, 0xffff);
            if (pos_path_id < 0) pos_path_id = kUnknownPosPath;
            relation_path_ids_[p][a] = relation_path_id;
            pos_path_ids_[p][a] = pos_path_id;
        }
    }
}

bool SemanticInstanceNumeric::ComputePassiveVoice(
        SemanticInstance *instance,
        int index) {
    const string &form = instance->GetForm(index);
    const string &tag = instance->GetPosTag(index);
    if (!IsVerb(index)) {
        return false; // Not even a verb.
    }

    std::string form_lower = form;
    std::transform(form_lower.begin(),
                   form_lower.end(),
                   form_lower.begin(),
                   ::tolower);

    if (0 == form_lower.compare("been")) return false;
    if (0 != tag.compare("VBN")) return false;

    // Find passive in parents.
    int head = instance->GetHead(index);
    while (true) {
        if (head <= 0) return true;

        const string &head_form = instance->GetForm(head);
        const string &head_tag = instance->GetPosTag(head);

        if (0 == head_tag.compare(0, 2, "NN")) return true;

        std::string head_form_lower = form;
        std::transform(head_form_lower.begin(),
                       head_form_lower.end(),
                       head_form_lower.begin(),
                       ::tolower);

        if (0 == head_form_lower.compare("am") ||
            0 == head_form_lower.compare("are") ||
            0 == head_form_lower.compare("is") ||
            0 == head_form_lower.compare("was") ||
            0 == head_form_lower.compare("were") ||
            0 == head_form_lower.compare("be") ||
            0 == head_form_lower.compare("been") ||
            0 == head_form_lower.compare("being")) {
            return true;
        }

        if (0 == head_form_lower.compare("have") ||
            0 == head_form_lower.compare("has") ||
            0 == head_form_lower.compare("had") ||
            0 == head_form_lower.compare("having")) {
            return false;
        }

        if (0 == head_tag.compare("VBZ") ||
            0 == head_tag.compare("VBD") ||
            0 == head_tag.compare("VBP") ||
            0 == head_tag.compare("MD")) {
            return true;
        }

        head = instance->GetHead(head);
    }

    return false;
}
