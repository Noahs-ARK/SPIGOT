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

#include <queue>
#include "SemanticPipe.h"

#ifndef _WIN32

#else
#include <time.h>
#endif


using namespace std;

// Define the current model version and the oldest back-compatible version.
// The format is AAAA.BBBB.CCCC, e.g., 2 0003 0000 means "2.3.0".
const uint64_t kSemanticParserModelVersion = 200030000;
const uint64_t kOldestCompatibleSemanticParserModelVersion = 200030000;
const uint64_t kSemanticParserModelCheck = 1234567890;

DEFINE_bool(use_only_labeled_arc_features, true,
            "True for not using unlabeled arc features in addition to labeled ones.");
DEFINE_bool(use_only_labeled_sibling_features, false, //true,
            "True for not using unlabeled sibling features in addition to labeled ones.");
DEFINE_bool(use_labeled_sibling_features, false, //true,
            "True for using labels in sibling features.");

void SemanticPipe::Initialize() {
	Pipe::Initialize();
	PreprocessData();
	classifier_model_ = new ParameterCollection();
	parser_model_ = new ParameterCollection();
	pruner_model_ = new ParameterCollection();

	SemanticOptions *semantic_options = GetSemanticOptions();

	if (semantic_options->trainer("parser") == "adadelta") {
		parser_trainer_ = new AdadeltaTrainer(*parser_model_);
		pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
	} else if (semantic_options->trainer("parser") == "adam") {
		parser_trainer_ = new AdamTrainer(*parser_model_,
		                                    semantic_options->eta0("parser"));
		pruner_trainer_ = new AdamTrainer(*pruner_model_,
		                                  semantic_options->eta0("parser"));
	} else if (semantic_options->trainer("parser") == "sgd") {
		parser_trainer_ = new SimpleSGDTrainer(*parser_model_,
		                                         semantic_options->eta0("parser"));
		pruner_trainer_ = new SimpleSGDTrainer(*pruner_model_,
		                                       semantic_options->eta0("parser"));
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	if (semantic_options->trainer("classification") == "adadelta") {
		classifier_trainer_ = new AdadeltaTrainer(*classifier_model_);
	} else if (semantic_options->trainer("classification") == "adam") {
		classifier_trainer_ = new AdamTrainer(*classifier_model_,
		                                      semantic_options->eta0("classification"));
	} else if (semantic_options->trainer("classification") == "sgd") {
		classifier_trainer_ = new SimpleSGDTrainer(*classifier_model_,
		                                           semantic_options->eta0("classification"));
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	if (semantic_options->batch_size() == 1) {
		classifier_trainer_->clip_threshold = 1.0;
		pruner_trainer_->clip_threshold = 1.0;
	}
	classifier_ = new Classifier(semantic_options, 2, classifier_model_);
	classifier_->InitParams(classifier_model_);

	int num_roles = GetSemanticDictionary()->GetNumRoles();
	parser_ = new SemanticParser(semantic_options, num_roles, GetSemanticDecoder(), parser_model_);
	static_cast<SemanticParser *> (parser_)->InitParams(parser_model_);
	if (semantic_options->pretrain_parser()) {
		const string file_path
				= semantic_options->pretrained_parser_model();
		load_dynet_model(file_path, parser_model_);
		semantic_options->parser_num_updates_ = 7434;
		for (int i = 0; i < 7434; ++i) {
			parser_model_->get_weight_decay().update_weight_decay();
			if (parser_model_->get_weight_decay().parameters_need_rescaled())
				parser_model_->get_weight_decay().reset_weight_decay();
		}
	}
	pruner_ = new SemanticPruner(semantic_options, GetSemanticDecoder(),
	                             pruner_model_);
	static_cast<SemanticPruner *> (pruner_)->InitParams(pruner_model_);
}

void SemanticPipe::SaveModel(FILE *fs) {
	bool success;
	success = WriteUINT64(fs, kSemanticParserModelCheck);
	CHECK(success);
	success = WriteUINT64(fs, kSemanticParserModelVersion);
	CHECK(success);
	token_dictionary_->Save(fs);
	classification_token_dictionary_->Save(fs);
	dependency_dictionary_->Save(fs);
	Pipe::SaveModel(fs);
	return;
}

void SemanticPipe::SaveNeuralModel() {
	string file_path = options_->GetModelFilePath() + ".dynet.classifier";
	save_dynet_model(file_path, classifier_model_);
	file_path = options_->GetModelFilePath() + ".dynet.parser";
	save_dynet_model(file_path, parser_model_);
}

void SemanticPipe::SavePruner() {
	SemanticOptions *semantic_options = GetSemanticOptions();
	const string file_path
			= semantic_options->GetPrunerModelFilePath();
	save_dynet_model(file_path, pruner_model_);
}

void SemanticPipe::LoadModel(FILE *fs) {
	bool success;
	uint64_t model_check;
	uint64_t model_version;
	success = ReadUINT64(fs, &model_check);
	CHECK(success);
	CHECK_EQ(model_check, kSemanticParserModelCheck)
		<< "The model file is too old and not supported anymore.";
	success = ReadUINT64(fs, &model_version);
	CHECK(success);
	CHECK_GE(model_version, kOldestCompatibleSemanticParserModelVersion)
		<< "The model file is too old and not supported anymore.";
	delete token_dictionary_;
	CreateTokenDictionary();
	static_cast<SemanticDictionary *>(dictionary_)->
			SetTokenDictionary(token_dictionary_);
	token_dictionary_->Load(fs);
	classification_token_dictionary_->Load(fs);
	CreateDependencyDictionary();
	dependency_dictionary_->SetTokenDictionary(token_dictionary_);
	static_cast<SemanticDictionary *>(dictionary_)->
			SetDependencyDictionary(dependency_dictionary_);
	dependency_dictionary_->Load(fs);

	options_->Load(fs);
	dictionary_->Load(fs);
}

void SemanticPipe::LoadNeuralModel() {
	if (classifier_trainer_) delete classifier_trainer_;
	if (classifier_model_) delete classifier_model_;
	if (classifier_) delete classifier_;
	if (parser_trainer_) delete parser_trainer_;
	if (parser_model_) delete parser_model_;
	if (parser_) delete parser_;

	classifier_model_ = new ParameterCollection();
	parser_model_ = new ParameterCollection();
	SemanticOptions *semantic_options = GetSemanticOptions();

	int num_roles = GetSemanticDictionary()->GetNumRoles();
	if (semantic_options->trainer("parser") == "adadelta") {
		parser_trainer_ = new AdadeltaTrainer(*parser_model_);
		pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
	} else if (semantic_options->trainer("parser") == "adam") {
		parser_trainer_ = new AdamTrainer(*parser_model_,
		                                  semantic_options->eta0("parser"));
		pruner_trainer_ = new AdamTrainer(*pruner_model_,
		                                  semantic_options->eta0("parser"));
	} else if (semantic_options->trainer("parser") == "sgd") {
		parser_trainer_ = new SimpleSGDTrainer(*parser_model_,
		                                       semantic_options->eta0("parser"));
		pruner_trainer_ = new SimpleSGDTrainer(*pruner_model_,
		                                       semantic_options->eta0("parser"));
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	if (semantic_options->trainer("classification") == "adadelta") {
		classifier_trainer_ = new AdadeltaTrainer(*classifier_model_);
	} else if (semantic_options->trainer("classification") == "adam") {
		classifier_trainer_ = new AdamTrainer(*classifier_model_,
		                                      semantic_options->eta0("classification"));
	} else if (semantic_options->trainer("classification") == "sgd") {
		classifier_trainer_ = new SimpleSGDTrainer(*classifier_model_,
		                                           semantic_options->eta0("classification"));
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	if (semantic_options->batch_size() == 1) {
		classifier_trainer_->clip_threshold = 1.0;
		pruner_trainer_->clip_threshold = 1.0;
	}
	classifier_ = new Classifier(semantic_options, 2, classifier_model_);
	classifier_->InitParams(classifier_model_);

	parser_ = new SemanticParser(semantic_options, num_roles, GetSemanticDecoder(), parser_model_);
	static_cast<SemanticParser *> (parser_)->InitParams(parser_model_);

	string file_path = options_->GetModelFilePath() + ".dynet.parser";
	load_dynet_model(file_path, parser_model_);
	// temporary solution to weight_decay issue in dynet
	// TODO: save the weight_decay along with the model.
	for (int i = 0; i < semantic_options->parser_num_updates_; ++i) {
		parser_model_->get_weight_decay().update_weight_decay();
		if (parser_model_->get_weight_decay().parameters_need_rescaled())
			parser_model_->get_weight_decay().reset_weight_decay();
	}

	file_path = options_->GetModelFilePath() + ".dynet.classifier";
	load_dynet_model(file_path, classifier_model_);
	// temporary solution to weight_decay issue in dynet
	// TODO: save the weight_decay along with the model.
	for (int i = 0; i < semantic_options->classification_num_updates_; ++i) {
		classifier_model_->get_weight_decay().update_weight_decay();
		if (classifier_model_->get_weight_decay().parameters_need_rescaled())
			classifier_model_->get_weight_decay().reset_weight_decay();
	}
}

void SemanticPipe::LoadPruner() {
	SemanticOptions *semantic_options = GetSemanticOptions();
	const string file_path = semantic_options->GetPrunerModelFilePath();
	if (pruner_model_) delete pruner_model_;
	if (pruner_) delete pruner_;
	if (pruner_trainer_) delete pruner_trainer_;

	pruner_model_ = new ParameterCollection();

	if (semantic_options->trainer("parser") == "adadelta") {
		pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
	} else if (semantic_options->trainer("parser") == "adam") {
		pruner_trainer_ = new AdamTrainer(*pruner_model_,
		                                  semantic_options->eta0("parser"));
	} else if (semantic_options->trainer("parser") == "sgd") {
		pruner_trainer_ = new SimpleSGDTrainer(*pruner_model_,
		                                       semantic_options->eta0("parser"));
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	pruner_ = new SemanticPruner(semantic_options, GetSemanticDecoder(),
	                             pruner_model_);
	static_cast<SemanticPruner *> (pruner_)->InitParams(pruner_model_);

	load_dynet_model(file_path, pruner_model_);
	// temporary solution to weight_decay issue in dynet
	// TODO: save the weight_decay along with the model.
	for (int i = 0; i < semantic_options->pruner_num_updates_; ++i) {
		pruner_model_->get_weight_decay().update_weight_decay();
		if (pruner_model_->get_weight_decay().parameters_need_rescaled())
			pruner_model_->get_weight_decay().reset_weight_decay();
	}
}

void SemanticPipe::PreprocessData() {
	if (token_dictionary_)
		delete token_dictionary_;
	if (classification_token_dictionary_)
		delete  classification_token_dictionary_;
	CreateTokenDictionary();
	if (dependency_dictionary_)
		delete dependency_dictionary_;
	CreateDependencyDictionary();
	dependency_dictionary_->SetTokenDictionary(token_dictionary_);
	static_cast<SemanticDictionary *>(dictionary_)->SetTokenDictionary(
			token_dictionary_);
	static_cast<SemanticTokenDictionary *>(token_dictionary_)->Initialize(
			GetSemanticReader());

	classification_form_count_ = new unordered_map<int, int>();
	classification_token_dictionary_->Initialize(GetSstReader(),
	                                             classification_form_count_);

	static_cast<SemanticDictionary *>(dictionary_)->SetDependencyDictionary(
			dependency_dictionary_);
	dependency_dictionary_->CreateLabelDictionary(GetDependencyReader());

	static_cast<SemanticDictionary *>(dictionary_)->CreatePredicateRoleDictionaries(
			GetSemanticReader());
}

void SemanticPipe::MakeParts(Instance *instance,
                             Parts *parts, vector<double> *gold_outputs) {
	int slen = static_cast<SemanticInstanceNumeric *>(instance)->size() - 1;
	auto semantic_parts = static_cast<SemanticParts *>(parts);
	semantic_parts->Initialize();
	bool make_gold = (gold_outputs != NULL);
	if (make_gold) gold_outputs->clear();
	parts->clear();
	if (GetSemanticOptions()->train_pruner()) {
		SemanticMakePartsBasic(instance, false, parts, gold_outputs);
		semantic_parts->BuildOffsets();
		semantic_parts->BuildIndices(slen, false);
	} else {
		SemanticMakePartsBasic(instance, parts, gold_outputs);
		semantic_parts->BuildOffsets();
		semantic_parts->BuildIndices(slen, GetSemanticOptions()->labeled());
	}
}

void SemanticPipe::SemanticMakePartsBasic(Instance *instance, Parts *parts,
                                          vector<double> *gold_outputs) {
	int slen = static_cast<SemanticInstanceNumeric *>(instance)->size() - 1;
	auto semantic_parts = static_cast<SemanticParts *>(parts);

	SemanticMakePartsBasic(instance, false, parts, gold_outputs);
	semantic_parts->BuildOffsets();
	semantic_parts->BuildIndices(slen, false);

	// Prune using a basic first-order model.
	if (GetSemanticOptions()->prune_basic()) {
		Prune(instance, parts, gold_outputs, options_->train());
		semantic_parts->BuildOffsets();
		semantic_parts->BuildIndices(slen, false);
	}

	if (GetSemanticOptions()->labeled()) {
		SemanticMakePartsBasic(instance, true, parts, gold_outputs);
	}
}

void SemanticPipe::SemanticMakePartsBasic(Instance *instance, bool add_labeled_parts,
                                          Parts *parts,
                                          vector<double> *gold_outputs) {
	auto sentence = static_cast<SemanticInstanceNumeric *>(instance);
	auto semantic_parts = static_cast<SemanticParts *>(parts);
	auto semantic_dictionary = GetSemanticDictionary();
	auto semantic_options = GetSemanticOptions();
	int slen = sentence->size() - 1;
	bool make_gold = (gold_outputs != NULL);
	bool prune_labels = semantic_options->prune_labels();
	bool prune_labels_with_relation_paths =
			semantic_options->prune_labels_with_relation_paths();
	bool prune_labels_with_senses = semantic_options->prune_labels_with_senses();
	bool prune_distances = semantic_options->prune_distances();
	bool allow_self_loops = semantic_options->allow_self_loops();
	bool allow_root_predicate = semantic_options->allow_root_predicate();
	bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
	bool use_predicate_senses = semantic_options->use_predicate_senses();
	vector<int> allowed_labels;

	if (add_labeled_parts && !prune_labels) {
		allowed_labels.resize(semantic_dictionary->GetRoleAlphabet().size());
		for (int i = 0; i < allowed_labels.size(); ++i) {
			allowed_labels[i] = i;
		}
	}

	// Add predicate parts.
	int num_parts_initial = semantic_parts->size();
	if (!add_labeled_parts) {
		for (int p = 0; p < slen; ++p) {
			if (p == 0 && !allow_root_predicate) continue;
			int lemma_id = TOKEN_UNKNOWN;
			if (use_predicate_senses) {
				lemma_id = sentence->GetLemmaId(p);
				CHECK_GE(lemma_id, 0);
			}
			const vector<SemanticPredicate *> *predicates =
					&semantic_dictionary->GetLemmaPredicates(lemma_id);
			if (predicates->size() == 0 && allow_unseen_predicates) {
				predicates = &semantic_dictionary->GetLemmaPredicates(
						TOKEN_UNKNOWN);
			}
			for (int s = 0; s < predicates->size(); ++s) {
				Part *part = semantic_parts->CreatePartPredicate(p, s);
				semantic_parts->AddPart(part);
				if (make_gold) {
					bool is_gold = false;
					int k = sentence->FindPredicate(p);
					if (k >= 0) {
						int predicate_id = sentence->GetPredicateId(k);
						if (!use_predicate_senses) {
							CHECK_EQ((*predicates)[s]->id(), PREDICATE_UNKNOWN);
						}
						if (predicate_id < 0 ||
						    (*predicates)[s]->id() == predicate_id) {
							is_gold = true;
						}
					}
					if (is_gold) {
						gold_outputs->push_back(1.0);
					} else {
						gold_outputs->push_back(0.0);
					}
				}
			}
		}

		// Compute offsets for predicate parts.
		semantic_parts->SetOffsetPredicate(num_parts_initial,
		                                   semantic_parts->size() -
		                                   num_parts_initial);
	}

	// Add unlabeled/labeled arc parts.
	num_parts_initial = semantic_parts->size();
	for (int p = 0; p < slen; ++p) {
		if (p == 0 && !allow_root_predicate) continue;
		int lemma_id = TOKEN_UNKNOWN;
		if (use_predicate_senses) {
			lemma_id = sentence->GetLemmaId(p);
			CHECK_GE(lemma_id, 0);
		}
		const vector<SemanticPredicate *> *predicates =
				&semantic_dictionary->GetLemmaPredicates(lemma_id);
		if (predicates->size() == 0 && allow_unseen_predicates) {
			predicates = &semantic_dictionary->GetLemmaPredicates(
					TOKEN_UNKNOWN);
		}
		for (int a = 1; a < slen; ++a) {
			if (!allow_self_loops && p == a) continue;
			for (int s = 0; s < predicates->size(); ++s) {
				int arc_index = -1;
				if (add_labeled_parts) {
					// If no unlabeled arc is there, just skip it.
					// This happens if that arc was pruned out.
					arc_index = semantic_parts->FindArc(p, a, s);
					if (0 > arc_index) {
						continue;
					}
				} else {
					if (prune_distances) {
						int predicate_pos_id = sentence->GetPosId(p);
						int argument_pos_id = sentence->GetPosId(a);
						if (p < a) {
							// Right attachment.
							if (a - p >
							    semantic_dictionary->GetMaximumRightDistance
									    (predicate_pos_id, argument_pos_id))
								continue;
						} else {
							// Left attachment.
							if (p - a >
							    semantic_dictionary->GetMaximumLeftDistance
									    (predicate_pos_id, argument_pos_id))
								continue;
						}
					}
				}

				if (prune_labels_with_relation_paths) {
					int relation_path_id = sentence->GetRelationPathId(p, a);
					allowed_labels.clear();
					if (relation_path_id >= 0 &&
					    relation_path_id < semantic_dictionary->
							    GetRelationPathAlphabet().size()) {
						allowed_labels = semantic_dictionary->
								GetExistingRolesWithRelationPath(
								relation_path_id);
						//LOG(INFO) << "Path: " << relation_path_id << " Roles: " << allowed_labels.size();
					}
					set<int> label_set;
					for (int m = 0; m < allowed_labels.size(); ++m) {
						if (!prune_labels_with_senses ||
						    (*predicates)[s]->HasRole(allowed_labels[m])) {
							label_set.insert(allowed_labels[m]);
						}
					}
					allowed_labels.clear();
					for (set<int>::iterator it = label_set.begin();
					     it != label_set.end(); ++it) {
						allowed_labels.push_back(*it);
					}
					if (!add_labeled_parts && allowed_labels.empty()) {
						continue;
					}
				} else if (prune_labels) {
					// TODO: allow both kinds of label pruning simultaneously?
					int predicate_pos_id = sentence->GetPosId(p);
					int argument_pos_id = sentence->GetPosId(a);
					allowed_labels.clear();
					allowed_labels = semantic_dictionary->
							GetExistingRoles(predicate_pos_id, argument_pos_id);
					set<int> label_set;
					for (int m = 0; m < allowed_labels.size(); ++m) {
						if (!prune_labels_with_senses ||
						    (*predicates)[s]->HasRole(allowed_labels[m])) {
							label_set.insert(allowed_labels[m]);
						}
					}
					allowed_labels.clear();
					for (set<int>::iterator it = label_set.begin();
					     it != label_set.end(); ++it) {
						allowed_labels.push_back(*it);
					}
					if (!add_labeled_parts && allowed_labels.empty()) {
						continue;
					}
				}

				// Add parts for labeled/unlabeled arcs.
				if (add_labeled_parts) {
					// If there is no allowed label for this arc, but the unlabeled arc was added,
					// then it was forced to be present for some reason (e.g. to maintain connectivity of the
					// graph). In that case (which should be pretty rare) consider all the
					// possible labels.
					if (allowed_labels.empty()) {
						allowed_labels.resize(
								semantic_dictionary->GetRoleAlphabet().size());
						for (int role = 0;
						     role < allowed_labels.size(); ++role) {
							allowed_labels[role] = role;
						}
					}

					for (int m = 0; m < allowed_labels.size(); ++m) {
						int role = allowed_labels[m];
						if (prune_labels && prune_labels_with_senses) {
							CHECK((*predicates)[s]->HasRole(role));
						}

						Part *part = semantic_parts->CreatePartLabeledArc(p, a, s, role);
						CHECK_GE(arc_index, 0);
						semantic_parts->AddLabeledPart(part, arc_index);
						if (make_gold) {
							int k = sentence->FindPredicate(p);
							int l = sentence->FindArc(p, a);
							bool is_gold = false;

							if (k >= 0 && l >= 0) {
								int predicate_id = sentence->GetPredicateId(k);
								int argument_id = sentence->GetArgumentRoleId(k, l);
								if (!use_predicate_senses) {
									CHECK_EQ((*predicates)[s]->id(),
									         PREDICATE_UNKNOWN);
								}
								//if (use_predicate_senses) CHECK_LT(predicate_id, 0);
								if ((predicate_id < 0 ||
								     (*predicates)[s]->id() == predicate_id) &&
								    role == argument_id) {
									is_gold = true;
								}
							}
							if (is_gold) {
								gold_outputs->push_back(1.0);
							} else {
								gold_outputs->push_back(0.0);
							}
						}
					}
				} else {
					Part *part = semantic_parts->CreatePartArc(p, a, s);
					semantic_parts->AddPart(part);
					if (make_gold) {
						int k = sentence->FindPredicate(p);
						int l = sentence->FindArc(p, a);
						bool is_gold = false;
						if (k >= 0 && l >= 0) {
							int predicate_id = sentence->GetPredicateId(k);
							if (!use_predicate_senses) {
								CHECK_EQ((*predicates)[s]->id(),
								         PREDICATE_UNKNOWN);
							}
							if (predicate_id < 0 ||
							    (*predicates)[s]->id() == predicate_id) {
								is_gold = true;
							}
						}
						if (is_gold) {
							gold_outputs->push_back(1.0);
						} else {
							gold_outputs->push_back(0.0);
						}
					}
				}
			}
		}
	}

	// Compute offsets for labeled/unlabeled arcs.
	if (!add_labeled_parts) {
		semantic_parts->SetOffsetArc(num_parts_initial, semantic_parts->size() - num_parts_initial);
	} else {
		semantic_parts->SetOffsetLabeledArc(num_parts_initial,
		                                    semantic_parts->size() - num_parts_initial);
	}
}

void SemanticPipe::Prune(Instance *instance, Parts *parts,
                                 vector<double> *gold_outputs,
                                 bool preserve_gold) {
	auto semantic_parts = static_cast<SemanticParts *>(parts);
	vector<double> scores;
	vector<double> predicted_outputs;

	// Make sure gold parts are only preserved at training time.
	CHECK(!preserve_gold || options_->train());
	if (!gold_outputs) preserve_gold = false;

	ComputationGraph cg;
	pruner_->StartGraph(cg, false);
	Expression ex_loss = static_cast<SemanticPruner *> (pruner_)
					->BuildGraph(instance, parts, &scores,
					             gold_outputs, &predicted_outputs,
					             parser_form_count_, false, cg);
	double loss = as_scalar(cg.forward(ex_loss));

	int offset_predicate_parts, num_predicate_parts;
	int offset_arcs, num_arcs;
	semantic_parts->GetOffsetPredicate(&offset_predicate_parts,
	                                   &num_predicate_parts);
	semantic_parts->GetOffsetArc(&offset_arcs, &num_arcs);

	double threshold = 0.5;
	int r0 = offset_arcs; // Preserve all the predicate parts.
	semantic_parts->ClearOffsets();
	semantic_parts->SetOffsetPredicate(offset_predicate_parts,
	                                   num_predicate_parts);
	for (int r = 0; r < num_arcs; ++r) {
		// Preserve gold parts (at training time).
		if (predicted_outputs[offset_arcs + r] >= threshold ||
		    (preserve_gold && (*gold_outputs)[offset_arcs + r] >= threshold)) {
			(*parts)[r0] = (*parts)[offset_arcs + r];
			semantic_parts->
					SetLabeledParts(r0, semantic_parts->GetLabeledParts(
					offset_arcs + r));
			if (gold_outputs) {
				(*gold_outputs)[r0] = (*gold_outputs)[offset_arcs + r];
			}
			++r0;
		} else {
			delete (*parts)[offset_arcs + r];
		}
	}
	if (gold_outputs) gold_outputs->resize(r0);
	semantic_parts->Resize(r0);
	semantic_parts->DeleteIndices();
	semantic_parts->SetOffsetArc(offset_arcs,
	                             parts->size() - offset_arcs);
}

void SemanticPipe::LabelInstance(Parts *parts, const vector<double> &output,
                                         Instance *instance) {
	auto semantic_parts = static_cast<SemanticParts *>(parts);
	auto semantic_instance = static_cast<SemanticInstance *>(instance);
	auto semantic_dictionary = GetSemanticDictionary();

	int slen = semantic_instance->size() - 1;
	double threshold = 0.5;
	semantic_instance->ClearPredicates();
	for (int p = 0; p < slen; ++p) {
		//if (p == 0 && !allow_root_predicate) continue;
		const vector<int> &senses = semantic_parts->GetSenses(p);
		vector<int> argument_indices;
		vector<string> argument_roles;
		int predicted_sense = -1;
		for (int k = 0; k < senses.size(); k++) {
			int s = senses[k];
			for (int a = 1; a < slen; ++a) {
				if (GetSemanticOptions()->labeled()) {
					int r = semantic_parts->FindArc(p, a, s);
					if (r < 0) continue;
					const vector<int> &labeled_arcs =
							semantic_parts->FindLabeledArcs(p, a, s);
					for (int l = 0; l < labeled_arcs.size(); ++l) {
						int r = labeled_arcs[l];
						CHECK_GE(r, 0);
						CHECK_LT(r, parts->size());
						if (output[r] > threshold) {
							if (predicted_sense != s) {
								CHECK_LT(predicted_sense, 0);
								predicted_sense = s;
							}
							argument_indices.push_back(a);
							SemanticPartLabeledArc *labeled_arc =
									static_cast<SemanticPartLabeledArc *>((*parts)[r]);
							string role =
									semantic_dictionary->GetRoleName(
											labeled_arc->role());
							argument_roles.push_back(role);
						}
					}
				} else {
					int r = semantic_parts->FindArc(p, a, s);
					if (r < 0) continue;
					if (output[r] > threshold) {
						if (predicted_sense != s) {
							CHECK_LT(predicted_sense, 0);
							predicted_sense = s;
						}
						argument_indices.push_back(a);
						argument_roles.push_back("ARG");
					}
				}
			}
		}

		if (predicted_sense >= 0) {
			int s = predicted_sense;
			// Get the predicate id for this part.
			// TODO(atm): store this somewhere, so that we don't need to recompute this
			// all the time. Maybe store this directly in arc->sense()?
			int lemma_id = TOKEN_UNKNOWN;
			if (GetSemanticOptions()->use_predicate_senses()) {
				lemma_id = semantic_dictionary->GetTokenDictionary()->
						GetLemmaId(semantic_instance->GetLemma(p));
				if (lemma_id < 0) lemma_id = TOKEN_UNKNOWN;
			}
			const vector<SemanticPredicate *> *predicates =
					&GetSemanticDictionary()->GetLemmaPredicates(lemma_id);
			if (predicates->size() == 0 &&
			    GetSemanticOptions()->allow_unseen_predicates()) {
				predicates = &GetSemanticDictionary()->GetLemmaPredicates(
						TOKEN_UNKNOWN);
			}
			int predicate_id = (*predicates)[s]->id();
			string predicate_name =
					semantic_dictionary->GetPredicateName(predicate_id);
			semantic_instance->AddPredicate(predicate_name, p, argument_roles,
			                                argument_indices);
		}
	}
}

void SemanticPipe::EvaluateInstance(
		Instance *instance,
		Instance *output_instance,
		Parts *parts,
		const vector<double> &gold_outputs,
		const vector<double> &predicted_outputs) {
	int num_possible_unlabeled_arcs = 0;
	int num_possible_labeled_arcs = 0;
	int num_gold_unlabeled_arcs = 0;
	int num_gold_labeled_arcs = 0;
	auto semantic_instance = static_cast<SemanticInstance *>(instance);
	auto semantic_parts = static_cast<SemanticParts *>(parts);
	int slen = semantic_instance->size() - 1;
	for (int p = 0; p < slen; ++p) {
		const vector<int> &senses = semantic_parts->GetSenses(p);
		for (int a = 1; a < slen; ++a) {
			bool unlab_gold = false, unlab_predicted = false;
			int lab_gold = -1, lab_predicted = -1;
			for (int k = 0; k < senses.size(); ++k) {
				int s = senses[k];
				int r = semantic_parts->FindArc(p, a, s);
				if (r < 0) continue;
				++num_possible_unlabeled_arcs;
				if (gold_outputs[r] >= 0.5) {
					CHECK_EQ(gold_outputs[r], 1.0);
					unlab_gold = true;
					++num_gold_unlabeled_arcs;
				}
				if (predicted_outputs[r] >= 0.5) {
					CHECK_EQ(predicted_outputs[r], 1.0);
					unlab_predicted = true;
				}
				if (GetSemanticOptions()->labeled()) {
					const vector<int> &labeled_arcs =
							semantic_parts->FindLabeledArcs(p, a, s);
					for (int k = 0; k < labeled_arcs.size(); ++k) {
						int r = labeled_arcs[k];
						if (r < 0) continue;
						int role = static_cast<SemanticPartLabeledArc *>((*parts)[r])->role();
						++num_possible_labeled_arcs;
						if (gold_outputs[r] >= 0.5) {
							CHECK_EQ(gold_outputs[r], 1.0);
							CHECK_EQ(lab_gold, -1);
							lab_gold = role;

							++num_gold_labeled_arcs;
						}
						if (predicted_outputs[r] >= 0.5) {
							CHECK_EQ(predicted_outputs[r], 1.0);
							CHECK_EQ(lab_predicted, -1);
							lab_predicted = role;
						}
					}
				}
			}
			num_matched_unlabeled_arcs_ += (unlab_gold && unlab_predicted);
			num_predicted_unlabeled_arcs_ += (unlab_predicted);
			num_matched_labeled_arcs_ += (lab_gold == lab_predicted && lab_gold >= 0);
			num_predicted_labeled_arcs_ += (lab_predicted >= 0);
		}

		++num_tokens_;
		num_unlabeled_arcs_after_pruning_ += num_possible_unlabeled_arcs;
		num_labeled_arcs_after_pruning_ += num_possible_labeled_arcs;
	}

	int num_actual_gold_arcs = 0;
	for (int k = 0; k < semantic_instance->GetNumPredicates(); ++k) {
		num_actual_gold_arcs +=
				semantic_instance->GetNumArgumentsPredicate(k);
	}
	num_gold_unlabeled_arcs_ += num_actual_gold_arcs;
	num_gold_labeled_arcs_ += num_actual_gold_arcs;
	int missed_unlabeled = num_actual_gold_arcs - num_gold_unlabeled_arcs;
	int missed_labeled = num_actual_gold_arcs - num_gold_labeled_arcs;
	int missed = missed_unlabeled + missed_labeled;
	num_pruned_gold_unlabeled_arcs_ += missed_unlabeled;
	num_possible_unlabeled_arcs_ += num_possible_unlabeled_arcs;
	num_pruned_gold_labeled_arcs_ += missed_labeled;
	num_possible_labeled_arcs_ += num_possible_labeled_arcs;
}

void SemanticPipe::Train() {
	CreateInstances();
	SemanticOptions *semantic_options = GetSemanticOptions();
	LoadPruner();
	if (semantic_options->use_pretrained_embedding()) {
		// if use pretrained parser, avoid initialiaing its word embedding
		LoadPretrainedEmbedding(true, !semantic_options->pretrain_parser());
	}
	BuildFormCount();
	vector<int> parser_idxs, classifier_idxs;
	for (int i = 0; i < parser_instances_.size(); ++i) {
		parser_idxs.push_back(i);
	}
	for (int i = 0; i < classification_instances_.size(); ++i) {
		classifier_idxs.push_back(i);
	}
	double unlabeled_F1 = 0, labeled_F1 = 0, best_accuracy = -1, accuracy = 0;

	for (int i = 0; i < options_->GetNumEpochs(); ++i) {
		semantic_options->train_on();
		random_shuffle(parser_idxs.begin(), parser_idxs.end());
		random_shuffle(classifier_idxs.begin(), classifier_idxs.end());

		TrainEpoch(parser_idxs, classifier_idxs, i);
		semantic_options->train_off();
		Run(unlabeled_F1, labeled_F1, accuracy);

		if (accuracy > best_accuracy && accuracy > 0.8){
			SaveNeuralModel();
			SaveModelFile();
			LOG(INFO) << semantic_options->parser_num_updates_
			          <<" " << semantic_options->classification_num_updates_;
			best_accuracy = accuracy;
		}
		LOG(INFO) <<"Current best: " << best_accuracy;
	}
}

void SemanticPipe::TrainPruner() {
	CreateInstances();
	SemanticOptions *semantic_options = GetSemanticOptions();

	BuildFormCount();
	vector<int> idxs;
	for (int i = 0; i < parser_instances_.size(); ++i) {
		idxs.push_back(i);
	}

	for (int i = 0; i < 3; ++i) {
		semantic_options->train_on();
		random_shuffle(idxs.begin(), idxs.end());
		TrainPrunerEpoch(idxs, i);
		semantic_options->train_off();
		LOG(INFO) << semantic_options->pruner_num_updates_;
		SavePruner();
		SaveModelFile();
	}
}

double SemanticPipe::TrainEpoch(const vector<int> &parser_idxs,
                                const vector<int> &classifier_idxs, int epoch) {
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();
	vector<Instance *> instance(batch_size, nullptr);
	vector<Instance *> classification_instance(batch_size, nullptr);

	vector<Parts *> parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) {
		parts[i] = CreateParts();
	}
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());

	vector<int> predicted_labels(batch_size);
	vector<int> gold_labels(batch_size);

	float forward_loss = 0.0;
	float parser_fraction = semantic_options->parser_fraction();
	int num_parser_instances = parser_idxs.size() * parser_fraction;
	int num_classifier_instances = classifier_idxs.size();

	int num_instances = num_parser_instances + num_classifier_instances;
	string feature = semantic_options->feature();
	LOG(INFO) << " Iteration #" << epoch + 1
	          << "; # parser instances: " << num_parser_instances
	          << "; # classifier instances: " << num_classifier_instances << endl;

	int parser_ite = 0, classifier_ite = 0;
	int n_batch = 0;
	for (int i = 0;i < num_instances; i += n_batch) {
		float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
		bool choose_parser = rand_float > num_parser_instances * 1.0 / num_instances;
		if (parser_ite >= num_parser_instances) choose_parser = false;
		if (classifier_ite >= num_classifier_instances) choose_parser = true;

		if (choose_parser) {
			n_batch = min(batch_size, num_parser_instances - parser_ite);
			for (int j = 0; j < n_batch; ++j) {
				instance[j] = parser_instances_[parser_idxs[parser_ite++]];
				MakeParts(instance[j], parts[j], &gold_outputs[j]);
			}
			ComputationGraph cg;
			parser_->StartGraph(cg, true);
			vector <Expression> ex_losses;

			for (int j = 0; j < n_batch; ++j) {
				Expression y_pred, ex_score, i_loss;
				i_loss = static_cast<SemanticParser *> (parser_)->BuildGraph(
						instance[j], parts[j], &scores[j],
						&gold_outputs[j], &predicted_outputs[j],
						ex_score, y_pred, parser_form_count_,
						true, cg);

				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
			if (loss > 0) {
				cg.backward(ex_loss);
				parser_trainer_->update();
				++semantic_options->parser_num_updates_;
			}
		} else{
			n_batch = min(batch_size, num_classifier_instances - classifier_ite);
			for (int j = 0; j < n_batch; ++j) {
				classification_instance[j] = classification_instances_[
						classifier_idxs[classifier_ite]];
				instance[j] = classification_parser_instances_[
						classifier_idxs[classifier_ite++]];
				gold_labels[j] = static_cast<SemanticInstanceNumeric *> (
						classification_instance[j])->GetLabel();
				if (feature == "headword") {
					MakeParts(instance[j], parts[j], nullptr);
				}
			}
			ComputationGraph cg;
			classifier_->StartGraph(cg);
			if (feature == "headword") {
				parser_->StartGraph(cg, false);
			}
			vector<Expression> ex_losses, ex_scores(n_batch), y_preds(n_batch);
			for (int j = 0; j < n_batch; ++j) {
				if (feature == "headword") {
					static_cast<SemanticParser *> (parser_)->BuildGraph(
							instance[j], parts[j], &scores[j],
							&gold_outputs[j], &predicted_outputs[j],
							ex_scores[j], y_preds[j], parser_form_count_,
							false, cg);
				}
				Expression i_loss = classifier_->Sst(
						classification_instance[j], parts[j],
						y_preds[j], gold_labels[j], predicted_labels[j],
						classification_form_count_, true, cg);
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = as_scalar(cg.forward(ex_loss));

			forward_loss += loss;
			cg.backward(ex_loss);
			classifier_trainer_->update();

			++semantic_options->classification_num_updates_;
			if (semantic_options->update_parser() && feature == "headword") {
				parser_trainer_->update();
				++semantic_options->parser_num_updates_;
			}
		}
	}
	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}
	parts.clear();
	parser_trainer_->status();
	LOG(INFO) << endl;
	classifier_trainer_->status();
	LOG(INFO) << endl;
	LOG(INFO) << "Training loss: " << forward_loss / num_classifier_instances << endl;
	if (semantic_options->halve("classification") > 0
	    && (epoch + 1 - semantic_options->parser_epochs())
	       % semantic_options->halve("classification") == 0) {
		classifier_trainer_->learning_rate /= 2;
	}

	if (semantic_options->halve("parser") > 0
	    && (epoch + 1) % semantic_options->halve("parser") == 0) {
		parser_trainer_->learning_rate /= 2;
	}
	return forward_loss;
}

double SemanticPipe::TrainPrunerEpoch(const vector<int> &idxs, int epoch) {
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();
	vector<Instance *> instance(batch_size, nullptr);
	vector<Parts *> parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) parts[i] = CreateParts();
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());
	double loss = 0.0;
	int num_instances = idxs.size();
	if (epoch == 0) {
		LOG(INFO) << "Number of instances: " << num_instances << endl;
	}
	LOG(INFO) << " Iteration #" << epoch + 1;
	for (int i = 0;i < num_instances; i += batch_size) {
		int n_batch = min(batch_size, num_instances - i);
		for (int j = 0; j < n_batch; ++j) {
			instance[j] = parser_instances_[idxs[i + j]];
			MakeParts(instance[j], parts[j], &gold_outputs[j]);
		}
		ComputationGraph cg;
		pruner_->StartGraph(cg, true);
		vector<Expression> ex_losses;
		for (int j = 0;j < n_batch; ++ j) {
			Expression i_loss;
			i_loss = static_cast<SemanticPruner *> (pruner_)
					->BuildGraph(instance[j], parts[j], &scores[j],
					             &gold_outputs[j], &predicted_outputs[j],
					             parser_form_count_, true, cg);
			ex_losses.push_back(i_loss);
		}
		Expression ex_loss = sum(ex_losses);
		loss += as_scalar(cg.forward(ex_loss));
		cg.backward(ex_loss);
		pruner_trainer_->update();
		++semantic_options->pruner_num_updates_;
	}
	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}
	parts.clear();
	LOG(INFO) << "training loss: " << loss / num_instances << endl;
	return loss;
}

void SemanticPipe::Test() {
	CreateInstances();
	SemanticOptions *semantic_options = GetSemanticOptions();
	LoadNeuralModel();
	LoadPruner();
	semantic_options->train_off();
	double unlabeled_F1 = 0, labeled_F1 = 0, accuracy = 0;
	Run(unlabeled_F1, labeled_F1, accuracy);
}

void SemanticPipe::Run(
		double &unlabeled_F1, double &labeled_F1, double &accuracy) {
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();
	vector<Instance *> instance(batch_size, nullptr);
	vector<Instance *> classification_instance(batch_size, nullptr);

	vector<Parts *> parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) {
		parts[i] = CreateParts();
	}
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());

	vector<int> predicted_labels(batch_size);
	vector<int> gold_labels(batch_size);

	timeval start, end;
	gettimeofday(&start, nullptr);

	double forward_loss = 0.0;
	int n_instances = 0;
	string feature = semantic_options->feature();
	if (feature == "headword") {
		if (options_->evaluate()) BeginEvaluation();
		int num_instances = parser_dev_instances_.size();

		writer_->Open(options_->GetOutputFilePath());
		for (int i = 0;i < num_instances; i += batch_size) {
			int n_batch = min(batch_size, num_instances - i);
			for (int j = 0;j < n_batch; ++ j) {
				instance[j] = GetFormattedInstance(parser_dev_instances_[i + j]);
				MakeParts(instance[j], parts[j], &gold_outputs[j]);
			}
			ComputationGraph cg;
			parser_->StartGraph(cg, false);
			vector<Expression> ex_losses, y_preds(n_batch), ex_scores(n_batch);
			for (int j = 0;j < n_batch; ++ j) {
				Expression i_loss;
				i_loss = static_cast<SemanticParser *> (parser_)->BuildGraph(
						instance[j], parts[j], &scores[j],
						&gold_outputs[j], &predicted_outputs[j],
						ex_scores[j], y_preds[j], parser_form_count_,
						false, cg);
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
			for (int j = 0;j < n_batch; ++ j) {
				Instance *output_instance = parser_dev_instances_[i + j]->Copy();
				LabelInstance(parts[j], predicted_outputs[j], output_instance);
				if (options_->evaluate()) {
					EvaluateInstance(parser_dev_instances_[i + j], output_instance,
					                 parts[j], gold_outputs[j], predicted_outputs[j]);
				}
				writer_->Write(output_instance);
				if (instance[j] != parser_dev_instances_[i + j]) delete instance[j];
				delete output_instance;
			}
		}
		writer_->Close();
		gettimeofday(&end, nullptr);
		if (options_->evaluate()) EndEvaluation(unlabeled_F1, labeled_F1);
	}
	{
		int num_instances = classification_dev_instances_.size();
		int total = 0, corr = 0;
		for (int i = 0;i < num_instances; i += batch_size) {
			int n_batch = min(batch_size, num_instances - i);

			for (int j = 0;j < n_batch; ++ j) {
				classification_instance[j] = GetSstFormattedInstance(
						classification_dev_instances_[i + j]);
				instance[j] = GetSstParserInstance(
						classification_dev_instances_[i + j]);
				gold_labels[j] = static_cast<SemanticInstanceNumeric *> (
						classification_instance[j])->GetLabel();
				if (feature == "headword") {
					MakeParts(instance[j], parts[j], nullptr);
				}
			}
			ComputationGraph cg;
			classifier_->StartGraph(cg);
			if (feature == "headword") {
				parser_->StartGraph(cg, false);
			}
			vector<Expression> ex_losses, y_preds(n_batch), ex_scores(n_batch);
			for (int j = 0;j < n_batch; ++ j) {
				if (feature == "headword") {
					static_cast<SemanticParser *> (parser_)->BuildGraph(
							instance[j], parts[j], &scores[j],
							&gold_outputs[j], &predicted_outputs[j],
							ex_scores[j], y_preds[j], parser_form_count_,
							false, cg);
				}
				Expression i_loss = classifier_->Sst(
						classification_instance[j], parts[j], y_preds[j],
						gold_labels[j], predicted_labels[j],
						classification_form_count_, false, cg);
				ex_losses.push_back(i_loss);

			}
			Expression ex_loss = sum(ex_losses);
			forward_loss += as_scalar(cg.forward(ex_loss));
			for (int j = 0;j < n_batch; ++ j) {
				if (classification_instance[j] != classification_dev_instances_[i + j])
					delete classification_instance[j];
				delete instance[j];
				corr += (predicted_labels[j] == gold_labels[j]);
				++total;
			}
		}
		n_instances += num_instances;
		accuracy = corr * 1.0 / total;
		LOG(INFO) << "Sst Acc: " << accuracy << endl;
	}
	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}
	forward_loss /= n_instances;
	LOG(INFO) << "Dev loss: " << forward_loss << endl;
}

void SemanticPipe::LoadPretrainedEmbedding(bool load_classifier_embedding,
                                           bool load_parser_embedding) {
	if (load_parser_embedding) {
		SemanticOptions *semantic_option = GetSemanticOptions();
		embedding_ = new unordered_map<int, vector<float>>();
		unsigned dim = semantic_option->word_dim("parser");
		ifstream in(semantic_option->GetEmbeddingFilePath("parser"));
		CHECK(in.is_open()) << "Pretrained embeddings FILE NOT FOUND!" << endl;
		string line;
		getline(in, line);
		vector<float> v(dim, 0);
		string word;
		int found = 0;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> word;
			for (unsigned i = 0; i < dim; ++i)
				lin >> v[i];
			int form_id = token_dictionary_->GetFormId(word);
			if (form_id < 0)
				continue;
			found += 1;
			(*embedding_)[form_id] = v;
		}
		in.close();
		LOG(INFO) << found << "/" << token_dictionary_->GetNumForms()
		          << " words found in the pretrained embedding";
		if (load_parser_embedding) parser_->LoadEmbedding(embedding_);
		delete embedding_;
	}
	if (load_classifier_embedding) {
		SemanticOptions *semantic_option = GetSemanticOptions();
		embedding_ = new unordered_map<int, vector<float>>();
		unsigned dim = semantic_option->word_dim("classification");
		ifstream in(semantic_option->GetEmbeddingFilePath("classification"));
		CHECK(in.is_open()) << "Pretrained embeddings FILE NOT FOUND!" << endl;
		string line;
		getline(in, line);
		vector<float> v(dim, 0);
		string word;
		int found = 0;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> word;
			for (unsigned i = 0; i < dim; ++i)
				lin >> v[i];
			int form_id = classification_token_dictionary_->GetFormId(word);
			if (form_id < 0)
				continue;
			found += 1;
			(*embedding_)[form_id] = v;
		}
		in.close();
		LOG(INFO) << found << "/" << classification_token_dictionary_->GetNumForms()
		          << " words found in the pretrained embedding";
		classifier_->LoadEmbedding(embedding_);
		delete embedding_;
	}
}

void SemanticPipe::BuildFormCount() {
	parser_form_count_ = new unordered_map<int, int>();
	for (int i = 0; i < parser_instances_.size(); i++) {
		Instance *instance = parser_instances_[i];
		auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
		const vector<int> form_ids = sentence->GetFormIds();
		for (int r = 0; r < form_ids.size(); ++r) {
			int form_id = form_ids[r];
			CHECK_NE(form_id, UNK_ID);
			if (parser_form_count_->find(form_id) == parser_form_count_->end()) {
				(*parser_form_count_)[form_id] = 1;
			} else {
				(*parser_form_count_)[form_id] += 1;
			}
		}
	}
}