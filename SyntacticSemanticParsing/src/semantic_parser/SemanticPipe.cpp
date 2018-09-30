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
	semantic_model_ = new ParameterCollection();
	dependency_model_ = new ParameterCollection();
	semantic_pruner_model_ = new ParameterCollection();
	dependency_pruner_model_ = new ParameterCollection();
	SemanticOptions *semantic_options = GetSemanticOptions();
	if (semantic_options->trainer("semantic") == "adadelta") {
		semantic_trainer_ = new AdadeltaTrainer(*semantic_model_);
		semantic_pruner_trainer_ = new AdadeltaTrainer(*semantic_pruner_model_);
	} else if (semantic_options->trainer("semantic") == "adam") {
		semantic_trainer_ = new AdamTrainer(*semantic_model_, semantic_options->semantic_eta0_);
		semantic_pruner_trainer_ = new AdamTrainer(*semantic_pruner_model_,semantic_options->semantic_eta0_);
	} else if (semantic_options->trainer("semantic") == "sgd") {
		semantic_trainer_ = new SimpleSGDTrainer(*semantic_model_, semantic_options->semantic_eta0_);
		semantic_pruner_trainer_ = new SimpleSGDTrainer(*semantic_pruner_model_, semantic_options->semantic_eta0_);
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	if (semantic_options->trainer("dependency") == "adadelta") {
		dependency_trainer_
				= new AdadeltaTrainer(*dependency_model_);
		dependency_pruner_trainer_
				= new AdadeltaTrainer(*dependency_pruner_model_);
	} else if (semantic_options->trainer("dependency") == "adam") {
		dependency_trainer_
				= new AdamTrainer(*dependency_model_, semantic_options->dependency_eta0_);
		dependency_pruner_trainer_
				= new AdamTrainer(*dependency_pruner_model_, semantic_options->dependency_eta0_);
	} else if (semantic_options->trainer("dependency") == "sgd") {
		dependency_trainer_
				= new SimpleSGDTrainer(*dependency_model_, semantic_options->dependency_eta0_);
		dependency_pruner_trainer_
				= new SimpleSGDTrainer(*dependency_pruner_model_, semantic_options->dependency_eta0_);
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}
	if (semantic_options->batch_size() == 1) {
		semantic_trainer_->clip_threshold = 1.0;
		dependency_trainer_->clip_threshold = 1.0;
		semantic_pruner_trainer_->clip_threshold = 1.0;
		dependency_pruner_trainer_->clip_threshold = 1.0;
	}
	int semantic_num_roles = GetSemanticDictionary()->GetNumRoles();

	semantic_parser_ = new SemanticParser(semantic_options, semantic_num_roles,
	                               GetSemanticDecoder(), semantic_model_);
	semantic_parser_->InitParams(semantic_model_);

	if (semantic_options->struct_att()) {
		parser_ = new StructuredAttention(semantic_options, GetDepdendencyDecoder(),
		                                  dependency_model_);
		static_cast<StructuredAttention *> (parser_)->InitParams(dependency_model_);
	} else {
		parser_ = new Dependency(semantic_options,
		                         GetDepdendencyDecoder(), dependency_model_);
		static_cast<Dependency *> (parser_)->InitParams(dependency_model_);
	}

	semantic_pruner_ = new SemanticPruner(semantic_options,
	                            GetSemanticDecoder(), semantic_pruner_model_);
	semantic_pruner_->InitParams(semantic_pruner_model_);
	dependency_pruner_ = new DependencyPruner(semantic_options,
	                                          GetDepdendencyDecoder(),
	                                          dependency_pruner_model_);
	dependency_pruner_->InitParams(dependency_pruner_model_);
}

void SemanticPipe::SaveModel(FILE *fs) {
	bool success;
	success = WriteUINT64(fs, kSemanticParserModelCheck);
	CHECK(success);
	success = WriteUINT64(fs, kSemanticParserModelVersion);
	CHECK(success);
	dependency_token_dictionary_->Save(fs);
	semantic_token_dictionary_->Save(fs);
	dependency_dictionary_->Save(fs);
	semantic_dictionary_->Save(fs);
	options_->Save(fs);
	return;
}

void SemanticPipe::SaveNeuralModel() {
	string file_path = options_->GetModelFilePath() + ".semantic.dynet";
	save_dynet_model(file_path, semantic_model_);
	file_path = options_->GetModelFilePath() + ".dependency.dynet";
	save_dynet_model(file_path, dependency_model_);
}

void SemanticPipe::SavePruner(const string &formalism) {
	if (formalism != "dependency" && formalism != "semantic") {
		CHECK(false)
		<< "Unsupported formalism: " << formalism << ". Giving up..." << endl;
	}
	SemanticOptions *semantic_options = GetSemanticOptions();
	const string file_path
			= semantic_options->GetPrunerModelFilePath(formalism);
	if (formalism == "dependency") {
		save_dynet_model(file_path, dependency_pruner_model_);
	} else if (formalism == "semantic") {
		save_dynet_model(file_path, semantic_pruner_model_);
	}
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
	delete dependency_token_dictionary_;
	delete semantic_token_dictionary_;

	CreateTokenDictionary();
	static_cast<DependencyDictionary *>(dependency_dictionary_)->
			SetTokenDictionary(dependency_token_dictionary_);
	static_cast<SemanticDictionary *>(semantic_dictionary_)->
			SetTokenDictionary(semantic_token_dictionary_);
	static_cast<SemanticDictionary *>(semantic_dictionary_)->SetDependencyDictionary(
			static_cast<DependencyDictionary *> (dependency_dictionary_));
	dependency_token_dictionary_->Load(fs);
	semantic_token_dictionary_->Load(fs);
	dependency_dictionary_->Load(fs);
	semantic_dictionary_->Load(fs);
	options_->Load(fs);
	return;
}

void SemanticPipe::LoadNeuralModel() {
	if (semantic_model_) delete semantic_model_;
	if (semantic_trainer_) delete semantic_trainer_;
	if (semantic_parser_) delete semantic_parser_;
	if (dependency_model_) delete dependency_model_;
	if (dependency_trainer_) delete dependency_trainer_;
	if (parser_) delete parser_;

	SemanticOptions *semantic_options = GetSemanticOptions();

	semantic_model_ = new ParameterCollection();
	dependency_model_ = new ParameterCollection();

	int semantic_num_roles = GetSemanticDictionary()->GetNumRoles();

	if (semantic_options->trainer("semantic") == "adadelta") {
		semantic_trainer_ = new AdadeltaTrainer(*semantic_model_);
		semantic_pruner_trainer_ = new AdadeltaTrainer(*semantic_pruner_model_);
	} else if (semantic_options->trainer("semantic") == "adam") {
		semantic_trainer_ = new AdamTrainer(*semantic_model_, semantic_options->semantic_eta0_);
		semantic_pruner_trainer_ = new AdamTrainer(*semantic_pruner_model_,semantic_options->semantic_eta0_);
	} else if (semantic_options->trainer("semantic") == "sgd") {
		semantic_trainer_ = new SimpleSGDTrainer(*semantic_model_, semantic_options->semantic_eta0_);
		semantic_pruner_trainer_ = new SimpleSGDTrainer(*semantic_pruner_model_, semantic_options->semantic_eta0_);
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}


	if (semantic_options->trainer("dependency") == "adadelta") {
		dependency_trainer_
				= new AdadeltaTrainer(*dependency_model_);
		dependency_pruner_trainer_
				= new AdadeltaTrainer(*dependency_pruner_model_);
	} else if (semantic_options->trainer("dependency") == "adam") {
		dependency_trainer_
				= new AdamTrainer(*dependency_model_, semantic_options->dependency_eta0_);
		dependency_pruner_trainer_
				= new AdamTrainer(*dependency_pruner_model_, semantic_options->dependency_eta0_);
	} else if (semantic_options->trainer("dependency") == "sgd") {
		dependency_trainer_
				= new SimpleSGDTrainer(*dependency_model_, semantic_options->dependency_eta0_);
		dependency_pruner_trainer_
				= new SimpleSGDTrainer(*dependency_pruner_model_, semantic_options->dependency_eta0_);
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	if (semantic_options->batch_size() == 1) {
		semantic_trainer_->clip_threshold = 1.0;
		dependency_trainer_->clip_threshold = 1.0;
	}
	semantic_parser_ = new SemanticParser(semantic_options, semantic_num_roles, GetSemanticDecoder(), semantic_model_);
	semantic_parser_->InitParams(semantic_model_);
	if (semantic_options->struct_att()) {
		parser_ = new StructuredAttention(semantic_options, GetDepdendencyDecoder(),
		                                  dependency_model_);
		static_cast<StructuredAttention *> (parser_)->InitParams(dependency_model_);
	} else {
		parser_ = new Dependency(semantic_options,
		                         GetDepdendencyDecoder(), dependency_model_);
		static_cast<Dependency *> (parser_)->InitParams(dependency_model_);
	}


	string file_path = options_->GetModelFilePath() + ".semantic.dynet";
	load_dynet_model(file_path, semantic_model_);

	for (int i = 0; i < semantic_options->semantic_num_updates_; ++i) {
		semantic_model_->get_weight_decay().update_weight_decay();
		if (semantic_model_->get_weight_decay().parameters_need_rescaled())
			semantic_model_->get_weight_decay().reset_weight_decay();
	}


	file_path = options_->GetModelFilePath() + ".dependency.dynet";
	load_dynet_model(file_path, dependency_model_);
	for (uint64_t i = 0; i < semantic_options->dependency_num_updates_; ++i) {
		dependency_model_->get_weight_decay().update_weight_decay();
		if (dependency_model_->get_weight_decay().parameters_need_rescaled())
			dependency_model_->get_weight_decay().reset_weight_decay();
	}
}

void SemanticPipe::LoadPruner(const string &formalism) {
	if (formalism != "dependency" && formalism != "semantic") {
		CHECK(false)
		<< "Unsupported formalism: " << formalism << ". Giving up..." << endl;
	}

	SemanticOptions *semantic_options = GetSemanticOptions();
	const string file_path = semantic_options->GetPrunerModelFilePath(
			formalism);

	if (formalism == "dependency") {
		dependency_pruner_model_ = new ParameterCollection();
		if (semantic_options->trainer("dependency") == "adadelta")
			dependency_pruner_trainer_ = new AdadeltaTrainer(
					*dependency_pruner_model_);
		else if (semantic_options->trainer("dependency") == "adam") {
			dependency_pruner_trainer_ = new AdamTrainer(
					*dependency_pruner_model_, semantic_options->dependency_eta0_);
		} else if (semantic_options->trainer("dependency") == "sgd") {
			dependency_pruner_trainer_ = new SimpleSGDTrainer(
					*dependency_pruner_model_, semantic_options->dependency_eta0_);
		} else {
			CHECK(false) << "Unsupported trainer. Giving up..." << endl;
		}
		dependency_pruner_trainer_->clip_threshold = 1.0;
		dependency_pruner_ = new DependencyPruner(semantic_options,
		                                          GetDepdendencyDecoder(),
		                                          dependency_pruner_model_);

		dependency_pruner_->InitParams(dependency_pruner_model_);
		load_dynet_model(file_path, dependency_pruner_model_);
		dependency_pruner_model_->get_weight_decay().update_weight_decay(
				semantic_options->dependency_pruner_num_updates_);
	} else if (formalism == "semantic") {
		semantic_pruner_model_ = new ParameterCollection();
		if (semantic_options->trainer("semantic") == "adadelta")
			semantic_pruner_trainer_ = new AdadeltaTrainer(*semantic_pruner_model_);
		else if (semantic_options->trainer("semantic") == "adam") {
			semantic_pruner_trainer_ = new AdamTrainer(*semantic_pruner_model_,
			                                      semantic_options->semantic_eta0_);
		} else if (semantic_options->trainer("semantic") == "sgd") {
			semantic_pruner_trainer_ = new SimpleSGDTrainer(*semantic_pruner_model_,
			                                           semantic_options->semantic_eta0_);
		} else {
			CHECK(false) << "Unsupported trainer. Giving up..." << endl;
		}
		semantic_pruner_trainer_->clip_threshold = 1.0;
		semantic_pruner_ = new SemanticPruner(semantic_options,
		                            GetSemanticDecoder(), semantic_pruner_model_);
		semantic_pruner_->InitParams(semantic_pruner_model_);
		load_dynet_model(file_path, semantic_pruner_model_);
		semantic_pruner_model_->get_weight_decay().update_weight_decay(
				semantic_options->semantic_pruner_num_updates_);
	}
}

void SemanticPipe::PreprocessData() {
	delete dependency_token_dictionary_;
	delete semantic_token_dictionary_;
	CreateTokenDictionary();
	delete dependency_dictionary_;
	CreateDependencyDictionary();
	static_cast<DependencyDictionary *>(dependency_dictionary_)->SetTokenDictionary(
			dependency_token_dictionary_);
	static_cast<SemanticDictionary *>(semantic_dictionary_)->SetTokenDictionary(
			semantic_token_dictionary_);

	static_cast<SemanticTokenDictionary *>(semantic_token_dictionary_)
			->Initialize(GetSemanticReader());
	static_cast<DependencyTokenDictionary *>(dependency_token_dictionary_)
			->Initialize(GetDependencyReader());

	static_cast<SemanticDictionary *>(semantic_dictionary_)->SetDependencyDictionary(
			static_cast<DependencyDictionary *> (dependency_dictionary_));
	static_cast<DependencyDictionary *>(dependency_dictionary_)->CreateLabelDictionary(
			GetDependencyReader());
	static_cast<SemanticDictionary *>(semantic_dictionary_)->CreatePredicateRoleDictionaries(
			static_cast<SemanticReader *> (semantic_reader_));
}

void SemanticPipe::EnforceWellFormedGraph(Instance *instance,
                                          const vector<Part *> &arcs,
                                          vector<int> *inserted_heads,
                                          vector<int> *inserted_modifiers) {
	SemanticOptions *dependency_options = GetSemanticOptions();
	if (dependency_options->projective()) {
		EnforceProjectiveGraph(instance, arcs, inserted_heads,
		                       inserted_modifiers);
	} else {
		EnforceConnectedGraph(instance, arcs, inserted_heads,
		                      inserted_modifiers);
	}
}

// Make sure the graph formed by the unlabeled arc parts is connected,
// otherwise there is no feasible solution.
// If necessary, root nodes are added and passed back through the last
// argument.
void SemanticPipe::EnforceConnectedGraph(Instance *instance,
                                         const vector<Part *> &arcs,
                                         vector<int> *inserted_heads,
                                         vector<int> *inserted_modifiers) {
	DependencyInstanceNumeric *sentence =
			static_cast<DependencyInstanceNumeric *>(instance);
	inserted_heads->clear();
	inserted_modifiers->clear();
	int slen = sentence->size() - 1;

	// Create a list of children for each node.
	vector<vector<int> > children(slen);
	for (int r = 0; r < arcs.size(); ++r) {
		CHECK_EQ(arcs[r]->type(), DEPENDENCYPART_ARC);
		DependencyPartArc *arc = static_cast<DependencyPartArc *>(arcs[r]);
		int h = arc->head();
		int m = arc->modifier();
		children[h].push_back(m);
	}

	// Check if the root is connected to every node.
	vector<bool> visited(slen, false);
	queue<int> nodes_to_explore;
	nodes_to_explore.push(0);
	while (!nodes_to_explore.empty()) {
		int h = nodes_to_explore.front();
		nodes_to_explore.pop();
		visited[h] = true;
		for (int k = 0; k < children[h].size(); ++k) {
			int m = children[h][k];
			if (visited[m]) continue;
			nodes_to_explore.push(m);
		}

		// If there are no more nodes to explore, check if all nodes
		// were visited and, if not, add a new edge from the node to
		// the first node that was not visited yet.
		if (nodes_to_explore.empty()) {
			for (int m = 1; m < slen; ++m) {
				if (!visited[m]) {
					LOG(INFO) << "Inserted root node 0 -> " << m << ".";
					inserted_heads->push_back(0);
					inserted_modifiers->push_back(m);
					nodes_to_explore.push(m);
					break;
				}
			}
		}
	}
}

// Make sure the graph formed by the unlabeled arc parts admits a projective
// tree, otherwise there is no feasible solution when --projective=true.
// If necessary, we add arcs of the form m-1 -> m to make sure the sentence
// has a projective parse.
void SemanticPipe::EnforceProjectiveGraph(Instance *instance,
                                          const vector<Part *> &arcs,
                                          vector<int> *inserted_heads,
                                          vector<int> *inserted_modifiers) {
	DependencyInstanceNumeric *sentence =
			static_cast<DependencyInstanceNumeric *>(instance);
	inserted_heads->clear();
	inserted_modifiers->clear();
	int slen = sentence->size() - 1;
	// Create an index of existing arcs.
	vector<vector<int> > index(slen, vector<int>(slen, -1));
	for (int r = 0; r < arcs.size(); ++r) {
		CHECK_EQ(arcs[r]->type(), DEPENDENCYPART_ARC);
		DependencyPartArc *arc = static_cast<DependencyPartArc *>(arcs[r]);
		int h = arc->head();
		int m = arc->modifier();
		index[h][m] = r;
	}

	// Insert consecutive right arcs if necessary.
	for (int m = 1; m < slen; ++m) {
		int h = m - 1;
		if (index[h][m] < 0) {
			inserted_heads->push_back(h);
			inserted_modifiers->push_back(m);
		}
	}
}

void SemanticPipe::MakeParts(const string &formalism, Instance *instance,
                             Parts *parts, vector<double> *gold_outputs) {
	if (formalism != "dependency" && formalism != "semantic") {
		CHECK(false)
		<< "Unsupported formalism: " << formalism << ". Giving up...";
	}
	if (formalism == "dependency") {
		int slen = static_cast<DependencyInstanceNumeric *>(instance)->size() - 1;
		auto dependency_parts = static_cast<DependencyParts *>(parts);
		dependency_parts->Initialize();
		bool make_gold = (gold_outputs != nullptr);
		if (make_gold) gold_outputs->clear();

		if (GetSemanticOptions()->train_pruner()) {
			// For the pruner, make only unlabeled arc-factored and predicate parts and
			// compute indices.
			DependencyMakePartsBasic(instance, false, parts, gold_outputs);
			dependency_parts->BuildOffsets();
			dependency_parts->BuildIndices(slen, false);
		} else {
			// Make arc-factored and predicate parts and compute indices.
			DependencyMakePartsBasic(instance, parts, gold_outputs);
			dependency_parts->BuildOffsets();
			dependency_parts->BuildIndices(slen, false);
		}
	} else if (formalism == "semantic") {
		int slen = static_cast<SemanticInstanceNumeric *>(instance)->size() - 1;
		auto semantic_parts = static_cast<SemanticParts *>(parts);
		semantic_parts->Initialize();
		bool make_gold = (gold_outputs != nullptr);
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
}

void SemanticPipe::DependencyMakePartsBasic(Instance *instance, Parts *parts,
                                            vector<double> *gold_outputs) {
	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	int slen = sentence->size() - 1;
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	bool make_gold = (gold_outputs != nullptr);

	DependencyMakePartsBasic(instance, false, parts, gold_outputs);
	dependency_parts->BuildOffsets();
	dependency_parts->BuildIndices(slen, false);

	if (GetSemanticOptions()->prune_basic()) {
		if (options_->train()) {
			DependencyPrune(instance, parts, gold_outputs, true);
		} else {
			DependencyPrune(instance, parts, gold_outputs, false);
		}
		// In principle, the pruner should never make the graph
		// ill-formed, but this seems to happen sometimes...
		int num_parts_initial = 0;
		vector<Part *> arcs(dependency_parts->begin() +
		                    num_parts_initial,
		                    dependency_parts->end());
		vector<int> inserted_heads;
		vector<int> inserted_modifiers;
		EnforceWellFormedGraph(sentence, arcs, &inserted_heads,
		                       &inserted_modifiers);
		if (inserted_modifiers.size() > 0) {
//			LOG(INFO) << "The pruner made the graph ill-formed!";
		}
		for (int k = 0; k < inserted_modifiers.size(); ++k) {
			int m = inserted_modifiers[k];
			int h = inserted_heads[k];
			Part *part = dependency_parts->CreatePartArc(h, m);
			dependency_parts->push_back(part);
			if (make_gold) {
				if (sentence->GetHead(m) == h) {
					gold_outputs->push_back(1.0);
				} else {
					gold_outputs->push_back(0.0);
				}
			}
		}

		dependency_parts->SetOffsetArc(num_parts_initial,
				             dependency_parts->size() - num_parts_initial);

		dependency_parts->BuildOffsets();
		dependency_parts->BuildIndices(slen, false);
	}
//	LOG(INFO) << parts->size();
}

void SemanticPipe::DependencyMakePartsBasic(Instance *instance,
                                            bool add_labeled_parts,
                                            Parts *parts,
                                            vector<double> *gold_outputs) {
	auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	auto dependency_dictionary = GetDependencyDictionary();
	auto semantic_options = GetSemanticOptions();
	int slen = sentence->size() - 1;
	bool make_gold = (gold_outputs != nullptr);
	bool prune_labels = semantic_options->prune_labels();
	bool prune_distances = false;
	vector<int> allowed_labels;

	if (add_labeled_parts && !prune_labels) {
		allowed_labels.resize(dependency_dictionary->GetLabelAlphabet().size());
		for (int i = 0; i < allowed_labels.size(); ++i) {
			allowed_labels[i] = i;
		}
	}

	int num_parts_initial = dependency_parts->size();

	for (int h = 0; h < slen; ++h) {
		for (int m = 1; m < slen; ++m) {
			if (h == m) continue;
			if (add_labeled_parts) {
				// If no unlabeled arc is there, just skip it.
				// This happens if that arc was pruned out.
				if (0 > dependency_parts->FindArc(h, m)) continue;
			} else {
				if (h != 0 && prune_distances) {
					int modifier_pos_id = sentence->GetPosId(m);
					int head_pos_id = sentence->GetPosId(h);
					if (h < m) {
						// Right attachment.
						if (m - h >
						    dependency_dictionary->GetMaximumRightDistance
								    (modifier_pos_id, head_pos_id))
							continue;
					} else {
						// Left attachment.
						if (h - m >
						    dependency_dictionary->GetMaximumLeftDistance
								    (modifier_pos_id, head_pos_id))
							continue;
					}
				}
			}

			if (prune_labels) {
				int modifier_pos_id = sentence->GetPosId(m);
				int head_pos_id = sentence->GetPosId(h);
				allowed_labels.clear();
				allowed_labels = dependency_dictionary->
						GetExistingLabels(modifier_pos_id, head_pos_id);
				if (!add_labeled_parts && allowed_labels.empty()) {
					VLOG_IF(2, h == 0) << "No allowed labels between "
					                   << dependency_token_dictionary_->GetPosTagName(
							                   head_pos_id)
					                   << " and "
					                   << dependency_token_dictionary_->GetPosTagName(
							                   modifier_pos_id);
					continue;
				}
			}

			// Add parts for labeled/unlabeled arcs.
			if (add_labeled_parts) {
				// If there is no allowed label for this arc, but the unlabeled arc was added,
				// then it was forced to be present to maintain connectivity of the
				// graph. In that case (which should be pretty rare) consider all the
				// possible labels.
				if (allowed_labels.empty()) {
					allowed_labels.resize(
							dependency_dictionary->GetLabelAlphabet().size());
					for (int l = 0; l < allowed_labels.size(); ++l) {
						allowed_labels[l] = l;
					}
				}
				for (int k = 0; k < allowed_labels.size(); ++k) {
					int l = allowed_labels[k];
					Part *part = dependency_parts->CreatePartLabeledArc(h, m, l);
					dependency_parts->push_back(part);
					if (make_gold) {
						if (sentence->GetHead(m) == h &&
						    sentence->GetRelationId(m) == l) {
							gold_outputs->push_back(1.0);
						} else {
							gold_outputs->push_back(0.0);
						}
					}
				}
			} else {
				Part *part = dependency_parts->CreatePartArc(h, m);
				dependency_parts->push_back(part);
				if (make_gold) {
					if (sentence->GetHead(m) == h) {
						gold_outputs->push_back(1.0);
					} else {
						gold_outputs->push_back(0.0);
					}
				}
			}
		}
	}

	// When adding unlabeled arcs, make sure the graph stays connected.
	// Otherwise, enforce connectedness by adding some extra arcs
	// that connect words to the root.
	// NOTE: if --projective, enforcing connectedness is not enough,
	// so we add arcs of the form m-1 -> m to make sure the sentence
	// has a projective parse.
	if (!add_labeled_parts) {
		vector<Part *> arcs(dependency_parts->begin() +
		                    num_parts_initial,
		                    dependency_parts->end());
		vector<int> inserted_heads;
		vector<int> inserted_modifiers;
		EnforceWellFormedGraph(sentence, arcs, &inserted_heads,
		                       &inserted_modifiers);
		for (int k = 0; k < inserted_modifiers.size(); ++k) {
			int m = inserted_modifiers[k];
			int h = inserted_heads[k];
			Part *part = dependency_parts->CreatePartArc(h, m);
			dependency_parts->push_back(part);
			if (make_gold) {
				if (sentence->GetHead(m) == h) {
					gold_outputs->push_back(1.0);
				} else {
					gold_outputs->push_back(0.0);
				}
			}
		}

		dependency_parts->
				SetOffsetArc(num_parts_initial,
				             dependency_parts->size() - num_parts_initial);
	} else {
		dependency_parts->
				SetOffsetLabeledArc(num_parts_initial,
				                    dependency_parts->size() - num_parts_initial);
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
		SemanticPrune(instance, parts, gold_outputs, options_->train());
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
	bool make_gold = (gold_outputs != nullptr);
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
		semantic_parts->SetOffsetArc(
				num_parts_initial, semantic_parts->size() - num_parts_initial);
	} else {
		semantic_parts->SetOffsetLabeledArc(num_parts_initial,
		                                    semantic_parts->size() - num_parts_initial);
	}
}

void SemanticPipe::DependencyPrune(Instance *instance, Parts *parts,
                                   vector<double> *gold_outputs,
                                   bool preserve_gold) {
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	vector<double> scores;
	vector<double> predicted_outputs;

	// Make sure gold parts are only preserved at training time.
	CHECK(!preserve_gold || options_->train());
	if (!gold_outputs) preserve_gold = false;

	ComputationGraph cg;
	dependency_pruner_->StartGraph(cg, false);
	Expression ex_loss
			= dependency_pruner_->BuildGraph(instance, parts, &scores,
			                                 nullptr, &predicted_outputs,
			                                 dependency_form_count_, false, cg);
	double loss = as_scalar(cg.forward(ex_loss));

	double threshold = 0.5;
	int r0 = 0;
	for (int r = 0; r < parts->size(); ++r) {
		// Preserve gold parts (at training time).
		if (predicted_outputs[r] >= threshold ||
		    (preserve_gold && (*gold_outputs)[r] >= threshold)) {
			(*parts)[r0] = (*parts)[r];
			if (gold_outputs) (*gold_outputs)[r0] = (*gold_outputs)[r];
			++r0;
		} else {
			delete (*parts)[r];
		}
	}

	if (gold_outputs) gold_outputs->resize(r0);
	parts->resize(r0);
	dependency_parts->DeleteIndices();
	dependency_parts->SetOffsetArc(0, parts->size());
}

void SemanticPipe::SemanticPrune(Instance *instance, Parts *parts,
                         vector<double> *gold_outputs,
                         bool preserve_gold) {
	SemanticParts *semantic_parts
			= static_cast<SemanticParts *>(parts);
	vector<double> scores;
	vector<double> predicted_outputs;

	// Make sure gold parts are only preserved at training time.
	CHECK(!preserve_gold || options_->train());
	if (!gold_outputs) preserve_gold = false;

	ComputationGraph cg;
	semantic_pruner_->StartGraph(cg, false);
	Expression ex_loss
			= semantic_pruner_->BuildGraph(instance, parts, &scores,
			                      gold_outputs, &predicted_outputs,
			                               semantic_form_count_, false, cg);
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

void SemanticPipe::DependencyLabelInstance(Parts *parts,
                                           const vector<double> &output,
                                           Instance *instance) {
	auto dependency_parts = static_cast<DependencyParts *>(parts);
	auto dependency_instance = static_cast<DependencyInstance *>(instance);
	int instance_length = dependency_instance->size() - 1;

	int offset_labeled_arcs, num_labeled_arcs;
	dependency_parts->GetOffsetLabeledArc(&offset_labeled_arcs, &num_labeled_arcs);
	bool labeled = num_labeled_arcs > 0;
	for (int m = 0; m < instance_length; ++m) {
		dependency_instance->SetHead(m, -1);
		if (labeled) {
			dependency_instance->SetDependencyRelation(m, "nullptr");
		}
	}
	double threshold = 0.5;

	if (labeled) {
		int offset, num_labeled_arcs;
		dependency_parts->GetOffsetLabeledArc(&offset, &num_labeled_arcs);
		for (int r = 0; r < num_labeled_arcs; ++r) {
			auto arc = static_cast<DependencyPartLabeledArc *>((*dependency_parts)[
							offset + r]);
			if (output[offset + r] >= threshold) {
				dependency_instance->SetHead(arc->modifier(), arc->head());
				dependency_instance->SetDependencyRelation(
						arc->modifier(), GetDependencyDictionary()->GetLabelName(
								arc->label()));
			}
		}
	} else {
		int offset, num_basic_parts;
		dependency_parts->GetOffsetArc(&offset, &num_basic_parts);
		for (int r = 0; r < num_basic_parts; ++r) {
			auto arc = static_cast<DependencyPartArc *>((*dependency_parts)[
							offset + r]);
			if (output[offset + r] >= threshold) {
				dependency_instance->SetHead(arc->modifier(), arc->head());
			}
		}
	}
	for (int m = 1; m < instance_length; ++m) {
		if (dependency_instance->GetHead(m) < 0) {
			VLOG(2) << "Word without head.";
			dependency_instance->SetHead(m, 0);
			if (labeled) {
				dependency_instance->SetDependencyRelation(
						m, GetDependencyDictionary()->GetLabelName(0));
			}
		}
	}
}

void SemanticPipe::SemanticLabelInstance(Parts *parts, const vector<double> &output,
                                         Instance *instance) {
	auto semantic_parts = static_cast<SemanticParts *>(parts);
	auto semantic_instance = static_cast<SemanticInstance *>(instance);
	auto semantic_dictionary = GetSemanticDictionary();

	int slen = semantic_instance->size() - 1;
	double threshold = 0.5;
	semantic_instance->ClearPredicates();
	for (int p = 0; p < slen; ++p) {
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

void SemanticPipe::Train() {
	CreateInstances("dependency");
	CreateInstances("semantic");
	SemanticOptions *semantic_options = GetSemanticOptions();
	if (semantic_options->use_pretrained_embedding()) {
		LoadPretrainedEmbedding();
	}
	BuildFormCount();
	if (semantic_options->prune_basic()) {
		LoadPruner("dependency");
		LoadPruner("semantic");
	}
	vector<int> dependency_idxs, semantic_idxs;
	for (int i = 0; i < semantic_instances_.size(); ++i) semantic_idxs.push_back(i);
	for (int i = 0; i < dependency_instances_.size(); ++i)
		dependency_idxs.push_back(i);
	double unlabeled_F1 = 0, labeled_F1 = 0, best_labeled_F1 = -1;

	for (int i = 0; i < options_->GetNumEpochs(); ++i) {
		semantic_options->train_on();
		random_shuffle(dependency_idxs.begin(), dependency_idxs.end());
		random_shuffle(semantic_idxs.begin(), semantic_idxs.end());
		TrainEpoch(dependency_idxs, semantic_idxs,
		           i, best_labeled_F1);
		semantic_options->train_off();
		Run(unlabeled_F1, labeled_F1);
		if (labeled_F1 > best_labeled_F1 && labeled_F1 > 0.6) {
			SaveModelFile();
			SaveNeuralModel();
			LOG(INFO) << semantic_options->dependency_num_updates_
			          <<" " << semantic_options->semantic_num_updates_;
			best_labeled_F1 = labeled_F1;
		}
	}
}

void SemanticPipe::TrainPruner() {
	CreateInstances("semantic");
	CreateInstances("dependency");
	SemanticOptions *semantic_options = GetSemanticOptions();
	BuildFormCount();

	vector<int> idxs;
	for (int i = 0; i < semantic_instances_.size(); ++i) {
		idxs.push_back(i);
	}
	for (int i = 0; i < 3; ++i) {
		semantic_options->train_on();
		random_shuffle(idxs.begin(), idxs.end());
		TrainPrunerEpoch("semantic", idxs, i);
		semantic_options->train_off();
		LOG(INFO) << semantic_options->semantic_pruner_num_updates_;
		SavePruner("semantic");
	}
	idxs.clear();
	for (int i = 0; i < dependency_instances_.size(); ++i) {
		idxs.push_back(i);
	}
	for (int i = 0; i < 1; ++i) {
		semantic_options->train_on();
		random_shuffle(idxs.begin(), idxs.end());
		TrainPrunerEpoch("dependency", idxs, i);
		LOG(INFO) << semantic_options->dependency_pruner_num_updates_;
		SaveModelFile();
		SavePruner("dependency");
	}
	return;
}

double
SemanticPipe::TrainEpoch(vector<int> &dependency_idxs,
                         vector<int> &semantic_idxs,
                         int epoch, double &best_F1) {

	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();

	vector<Instance *> dependency_instance(batch_size, nullptr);
	vector<vector<double>> dependency_scores(batch_size, vector<double> ());
	vector<vector<double>> dependency_gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> dependency_predicted_outputs(batch_size, vector<double> ());
	vector<Parts *> dependency_parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) dependency_parts[i] = CreateParts("dependency");

	vector<Instance *> semantic_instance(batch_size, nullptr);
	vector<Instance *> semantic_dep_instance(batch_size, nullptr);
	vector<vector<double>> semantic_scores(batch_size, vector<double> ());
	vector<vector<double>> semantic_gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> semantic_predicted_outputs(batch_size, vector<double> ());
	vector<Parts *> semantic_parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) semantic_parts[i] = CreateParts("semantic");

	bool struct_att = semantic_options->struct_att();

	float forward_loss = 0.0;
	int dependency_num_instances = dependency_idxs.size();
	int semantic_num_instances = semantic_idxs.size();

	int num_instances = semantic_num_instances + dependency_num_instances;

	LOG(INFO) << " Iteration # " << epoch + 1
	          << "; Number of Dependency instances: "
	          << dependency_num_instances
	          << "; Number of Semantic instances: " << semantic_num_instances;

	int dependency_ite = 0, semantic_ite = 0, n_instances = 0;

	int n_batch = 0;
	int checkpoint_ite = 0;
	for (int i = 0; i < num_instances; i += n_batch) {
		float rand_float =
				static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
		string config;
		if (rand_float < dependency_num_instances * 1.0 / num_instances
		    && dependency_ite < dependency_num_instances
		    || semantic_ite >= semantic_num_instances) {
			config = "dependency";
		} else {
			config = "semantic";
		}
		if (config == "dependency") {
			n_batch = min(batch_size, dependency_num_instances - dependency_ite);
			for (int j = 0; j < n_batch; ++j) {
				dependency_instance[j] = dependency_instances_[dependency_idxs[dependency_ite++]];
				MakeParts("dependency", dependency_instance[j],
				          dependency_parts[j], &dependency_gold_outputs[j]);
			}
			ComputationGraph cg;
			parser_->StartGraph(cg, true);
			vector <Expression> ex_losses;
			for (int j = 0; j < n_batch; ++j) {
				Expression y_pred, ex_score, i_loss;
				if (struct_att) {
					i_loss = static_cast<StructuredAttention *> (parser_)->BuildGraph(
							dependency_instance[j], dependency_parts[j],
							&dependency_scores[j],
							&dependency_gold_outputs[j],
							&dependency_predicted_outputs[j],
							ex_score, y_pred, dependency_form_count_,
							true, false, cg);
				} else {
					i_loss = static_cast<Dependency *> (parser_)->BuildGraph(
							dependency_instance[j],
							dependency_parts[j],
							&dependency_scores[j],
							&dependency_gold_outputs[j],
							&dependency_predicted_outputs[j],
							ex_score, y_pred, dependency_form_count_,
							true, cg);
				}
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
			int corr = 0, num_parts = 0;
			for (int j = 0; j < n_batch; ++j) {
				for (int r = 0; r < dependency_parts[j]->size(); ++r) {
					if (NEARLY_EQ_TOL(dependency_gold_outputs[j][r],
					                  dependency_predicted_outputs[j][r], 1e-6))
						corr += 1;
					else break;
				}
				num_parts += dependency_parts[j]->size();
			}
			if (corr < num_parts || struct_att) {
				cg.backward(ex_loss);
				dependency_trainer_->update();
				++semantic_options->dependency_num_updates_;
			}
		} else if (config == "semantic") {
			n_batch = min(batch_size, semantic_num_instances - semantic_ite);
			for (int j = 0; j < n_batch; ++j) {
				semantic_instance[j] = semantic_instances_[semantic_idxs[semantic_ite]];
				semantic_dep_instance[j] = semantic_dep_instances_[semantic_idxs[semantic_ite++]];
				MakeParts("semantic", semantic_instance[j],
				          semantic_parts[j], &semantic_gold_outputs[j]);
				MakeParts("dependency", semantic_dep_instance[j],
				          dependency_parts[j], nullptr);
			}
			ComputationGraph cg;
			semantic_parser_->StartGraph(cg, true);
			parser_->StartGraph(cg, false);
			vector <Expression> ex_losses, ex_scores(n_batch);
			for (int j = 0; j < n_batch; ++j) {
				Expression y_pred, i_loss;
				if (struct_att) {
					static_cast<StructuredAttention *> (parser_)->BuildGraph(
							semantic_dep_instance[j], dependency_parts[j],
							&dependency_scores[j], nullptr,
							&dependency_predicted_outputs[j],
							ex_scores[j], y_pred, dependency_form_count_,
							false, false, cg);
				} else {
					static_cast<Dependency *> (parser_)->BuildGraph(
							semantic_dep_instance[j], dependency_parts[j],
							&dependency_scores[j], nullptr,
							&dependency_predicted_outputs[j],
							ex_scores[j], y_pred, dependency_form_count_,
							false, cg);
				}
				i_loss = semantic_parser_->BuildGraph(
						semantic_instance[j], semantic_parts[j],
						dependency_parts[j], &semantic_scores[j],
						&semantic_gold_outputs[j],
						&semantic_predicted_outputs[j],
						y_pred, semantic_form_count_, true, cg);
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);

			double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
			forward_loss += loss;
			int corr = 0, num_parts = 0;
			for (int j = 0; j < n_batch; ++j) {
				for (int r = 0; r < semantic_parts[j]->size(); ++r) {
					if (NEARLY_EQ_TOL(semantic_gold_outputs[j][r],
					                  semantic_predicted_outputs[j][r], 1e-6))
						corr += 1;
					else break;
				}
				num_parts += semantic_parts[j]->size();
			}
			if (corr < num_parts) {
				cg.backward(ex_loss);
				semantic_trainer_->update();
				++semantic_options->semantic_num_updates_;
			}
			dependency_trainer_->update();
			++semantic_options->dependency_num_updates_;
		}
		checkpoint_ite += n_batch;
		if (checkpoint_ite > 25000 && epoch > 5) {
			double unlabeled_F1, labeled_F1;
			semantic_options->train_off();
			Run(unlabeled_F1, labeled_F1);
			semantic_options->train_on();
			if (labeled_F1 > best_F1) {
				SaveModelFile();
				SaveNeuralModel();
				LOG(INFO) << semantic_options->dependency_num_updates_
				          <<" " << semantic_options->semantic_num_updates_;
				best_F1 = labeled_F1;
			}
			checkpoint_ite = 0;
		}
	}
	for (int i = 0;i < batch_size; ++ i) {
		if (dependency_parts[i]) delete dependency_parts[i];
		if (semantic_parts[i]) delete semantic_parts[i];
	}
	dependency_parts.clear(); semantic_parts.clear();
	semantic_trainer_->status(); LOG(INFO) << endl;
	dependency_trainer_->status();  LOG(INFO) << endl;
	if ((epoch + 1) % 10 == 0) {
		semantic_options->semantic_eta0_ /= 2;
		semantic_trainer_->learning_rate /= 2;
	}

	if ((epoch + 1) % 10 == 0) {
		semantic_options->dependency_eta0_ /= 2;
		dependency_trainer_->learning_rate /= 2;
	}

	forward_loss /= semantic_num_instances;
	LOG(INFO) << "training loss: " << forward_loss << endl;
	return forward_loss;
}

double
SemanticPipe::TrainPrunerEpoch(const string &formalism, const vector<int> &idxs,
                               int epoch) {
	if (formalism != "dependency" && formalism != "semantic") {
		CHECK(false)
		<< "Unsupported formalism: " << formalism << ". Giving up...";
	}
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();
	vector<Instance *> instance(batch_size, nullptr);
	vector<Parts *> parts(batch_size, nullptr);
	for (int i = 0; i < batch_size; ++i) parts[i] = CreateParts(formalism);
	vector<vector<double>> scores(batch_size, vector<double>());
	vector<vector<double>> gold_outputs(batch_size, vector<double>());
	vector<vector<double>> predicted_outputs(batch_size, vector<double>());

	double forward_loss = 0.0;
	int num_instances = idxs.size();
	if (epoch == 0) {
		LOG(INFO) << "Number of instances: " << num_instances << endl;
	}
	LOG(INFO) << " Iteration #" << epoch + 1;
	for (int i = 0; i < idxs.size(); i += batch_size) {
		int n_batch = min(batch_size, num_instances - i);
		if (formalism == "dependency") {
			for (int j = 0; j < n_batch; ++j) {
				instance[j] = dependency_instances_[idxs[i + j]];
				MakeParts(formalism, instance[j], parts[j], &gold_outputs[j]);
			}

			ComputationGraph cg;
			dependency_pruner_->StartGraph(cg, true);
			vector<Expression> ex_losses;
			for (int j = 0; j < n_batch; ++j) {
				Expression i_loss = dependency_pruner_
						->BuildGraph(instance[j], parts[j],
						             &scores[j], &gold_outputs[j],
						             &predicted_outputs[j],
						             dependency_form_count_, true, cg);
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = as_scalar(cg.forward(ex_loss));
			cg.backward(ex_loss);
			dependency_pruner_trainer_->update();
			++semantic_options->dependency_pruner_num_updates_;
			forward_loss += loss;
		} else if (formalism == "semantic") {
			for (int j = 0; j < n_batch; ++j) {
				instance[j] = semantic_instances_[idxs[i + j]];
				MakeParts(formalism, instance[j], parts[j], &gold_outputs[j]);
			}
			ComputationGraph cg;
			semantic_pruner_->StartGraph(cg, true);
			vector<Expression> ex_losses;
			for (int j = 0; j < n_batch; ++j) {
				Expression i_loss = semantic_pruner_
								->BuildGraph(instance[j], parts[j],
								             &scores[j],  &gold_outputs[j],
								             &predicted_outputs[j],
								             semantic_form_count_, true, cg);
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = as_scalar(cg.forward(ex_loss));
			cg.backward(ex_loss);
			semantic_pruner_trainer_->update();
			++semantic_options->semantic_pruner_num_updates_;
			forward_loss += loss;
		}
	}
	for (int i = 0; i < batch_size; ++i) {
		if (parts[i]) delete parts[i];
	}
	parts.clear();

	LOG(INFO) << "training loss: " << forward_loss / num_instances << endl;
	return forward_loss;
}

void SemanticPipe::Test() {
	CreateInstances("dependency");
	CreateInstances("semantic");
	SemanticOptions *semantic_options = GetSemanticOptions();
	LoadNeuralModel();
	LoadPruner("semantic");
	LoadPruner("dependency");
	semantic_options->train_off();
	double unlabeled_F1 = 0, labeled_F1 = 0;
	Run(unlabeled_F1, labeled_F1);
}

void SemanticPipe::Run(double &unlabeled_F1, double &labeled_F1) {
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();

	vector<Instance *> dependency_instance(batch_size, nullptr);
	vector<vector<double>> dependency_scores(batch_size, vector<double> ());
	vector<vector<double>> dependency_gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> dependency_predicted_outputs(batch_size, vector<double> ());
	vector<Parts *> dependency_parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) dependency_parts[i] = CreateParts("dependency");

	vector<Instance *> semantic_instance(batch_size, nullptr);
	vector<Instance *> semantic_dep_instance(batch_size, nullptr);
	vector<vector<double>> semantic_scores(batch_size, vector<double> ());
	vector<vector<double>> semantic_gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> semantic_predicted_outputs(batch_size, vector<double> ());
	vector<Parts *> semantic_parts(batch_size, nullptr);
	for (int i = 0;i < batch_size; ++ i) semantic_parts[i] = CreateParts("semantic");

	timeval start, end;
	gettimeofday(&start, nullptr);
	bool struct_att = semantic_options->struct_att();

	if (options_->evaluate()) BeginEvaluation();
	double forward_loss = 0.0;
	int n_instances = 0;

	{
		dependency_writer_->Open(semantic_options->GetOutputFilePath("dependency"));
		int num_instances = dependency_dev_instances_.size();
		for (int i = 0; i < num_instances; i += batch_size) {
			int n_batch = min(batch_size, num_instances - i);
			for (int j = 0;j < n_batch; ++ j) {
				dependency_instance[j] = GetFormattedInstance(
						"dependency", dependency_dev_instances_[i + j]);
				MakeParts("dependency", dependency_instance[j],
				          dependency_parts[j], &dependency_gold_outputs[j]);
			}
			ComputationGraph cg;
			parser_->StartGraph(cg, false);
			vector<Expression> ex_losses, ex_scores(n_batch);
			for (int j = 0;j < n_batch; ++ j) {
				Expression y_pred, i_loss;
				if (struct_att) {
					i_loss = static_cast<StructuredAttention *> (parser_)->BuildGraph(
							dependency_instance[j],
							dependency_parts[j],
							&dependency_scores[j],
							&dependency_gold_outputs[j],
							&dependency_predicted_outputs[j],
							ex_scores[j], y_pred,
							dependency_form_count_,
							false, true, cg);
				} else {
					i_loss = static_cast<Dependency *> (parser_)->BuildGraph(
							dependency_instance[j],
							dependency_parts[j],
							&dependency_scores[j],
							&dependency_gold_outputs[j],
							&dependency_predicted_outputs[j],
							ex_scores[j], y_pred,
							dependency_form_count_,
							false, cg);
				}
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
			for (int j = 0;j < n_batch; ++ j) {
				Instance *dependency_output_instance
						= dependency_dev_instances_[i + j]->Copy();
				DependencyLabelInstance(dependency_parts[j],
				                        dependency_predicted_outputs[j],
				                        dependency_output_instance);
				if (options_->evaluate()) {
					EvaluateInstance("dependency",
					                 dependency_dev_instances_[i + j],
					                 dependency_output_instance,
					                 dependency_parts[j],
					                 dependency_gold_outputs[j],
					                 dependency_predicted_outputs[j]);
				}
				dependency_writer_->Write(dependency_output_instance);
				if (dependency_instance[j] != dependency_dev_instances_[i + j]) delete dependency_instance[j];
				delete dependency_output_instance;
			}
		}
		dependency_writer_->Close();
	}

	{
		semantic_writer_->Open(semantic_options->GetOutputFilePath("semantic"));

		int num_instances = semantic_dev_instances_.size();
		n_instances += num_instances;
		for (int i = 0; i < num_instances; i += batch_size) {
			int n_batch = min(batch_size, num_instances - i);
			for (int j = 0;j < n_batch; ++ j) {
				semantic_instance[j] = GetFormattedInstance(
						"semantic", semantic_dev_instances_[i + j]);
				semantic_dep_instance[j] = GetFormattedInstance(
						"dependency", semantic_dev_instances_[i + j]);
				MakeParts("semantic", semantic_instance[j],
				          semantic_parts[j], &semantic_gold_outputs[j]);
				MakeParts("dependency", semantic_dep_instance[j],
				          dependency_parts[j], nullptr);
			}
			ComputationGraph cg;
			parser_->StartGraph(cg, false);
			semantic_parser_->StartGraph(cg, false);
			vector<Expression> ex_losses, ex_scores(n_batch);
			for (int j = 0;j < n_batch; ++ j) {
				Expression y_pred, i_loss;

				if (struct_att) {
					static_cast<StructuredAttention *> (parser_)->BuildGraph(
							semantic_dep_instance[j], dependency_parts[j],
							&dependency_scores[j], nullptr,
							&dependency_predicted_outputs[j],
							ex_scores[j], y_pred, dependency_form_count_,
							false, false, cg);
				} else {
					static_cast<Dependency *> (parser_)->BuildGraph(
							semantic_dep_instance[j], dependency_parts[j],
							&dependency_scores[j], nullptr,
							&dependency_predicted_outputs[j],
							ex_scores[j], y_pred, dependency_form_count_,
							false, cg);
				}
				i_loss = semantic_parser_->BuildGraph(
						semantic_instance[j], semantic_parts[j],
						dependency_parts[j], &semantic_scores[j],
						&semantic_gold_outputs[j],
						&semantic_predicted_outputs[j],
						y_pred, semantic_form_count_, false, cg);
				ex_losses.push_back(i_loss);
			}
			Expression ex_loss = sum(ex_losses);
			double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
			forward_loss += loss;
			for (int j = 0;j < n_batch; ++ j) {

				Instance *semantic_predicted_instance
						= semantic_dev_instances_[i + j]->Copy();
				Instance *semantic_gold_instance
						= semantic_dev_instances_[i + j]->Copy();
				static_cast<SemanticInstance *> (
						semantic_predicted_instance)->ClearPredicates();
				static_cast<SemanticInstance *> (
						semantic_gold_instance)->ClearPredicates();
				SemanticLabelInstance(semantic_parts[j], semantic_predicted_outputs[j],
				                      semantic_predicted_instance);
				if (options_->evaluate()) {
					SemanticEvaluateInstance(semantic_dev_instances_[i + j],
					                         semantic_predicted_instance,
					                         semantic_parts[j],
					                         semantic_gold_outputs[j],
					                         semantic_predicted_outputs[j]);
				}
				semantic_writer_->Write(semantic_predicted_instance);
				if (semantic_instance[j] != semantic_dev_instances_[i + j])
					delete semantic_instance[j];
				if (semantic_dep_instance[j] != semantic_dev_instances_[i + j])
					delete semantic_dep_instance[j];
				delete semantic_predicted_instance;
				delete semantic_gold_instance;
			}
		}
		semantic_writer_->Close();
	}

	forward_loss /= n_instances;
	LOG(INFO) << "dev loss: " << forward_loss << endl;

	for (int i = 0;i < batch_size; ++ i) {
		if (dependency_parts[i]) delete dependency_parts[i];
		if (semantic_parts[i]) delete semantic_parts[i];
	}
	dependency_parts.clear(); semantic_parts.clear();
	gettimeofday(&end, nullptr);
	if (options_->evaluate()) EndEvaluation(unlabeled_F1, labeled_F1);
}

void SemanticPipe::LoadPretrainedEmbedding() {
	SemanticOptions *semantic_option = GetSemanticOptions();
	dependency_embedding_ = new unordered_map<int, vector<float>>();
	semantic_embedding_ = new unordered_map<int, vector<float>>();
	unsigned dim = semantic_option->word_dim();
	ifstream in(semantic_option->GetPretrainedEmbeddingFilePath());

	if (!in.is_open()) {
		cerr << "Pretrained embeddings FILE NOT FOUND!" << endl;
	}
	string line;
	getline(in, line);
	vector<float> v(dim, 0);
	string word;
	int dep_found = 0, sem_found = 0;
	while (getline(in, line)) {
		istringstream lin(line);
		lin >> word;
		for (unsigned i = 0; i < dim; ++i)
			lin >> v[i];
		int form_id = dependency_token_dictionary_->GetFormId(word);
		if (form_id >= 0) {
			dep_found += 1;
			(*dependency_embedding_)[form_id] = v;
		}
		form_id = semantic_token_dictionary_->GetFormId(word);
		if (form_id >= 0) {
			sem_found += 1;
			(*semantic_embedding_)[form_id] = v;
		}
	}
	in.close();
	LOG(INFO) << "Dependency: " << dep_found << "/" << dependency_token_dictionary_->GetNumForms()
	          << " words found in the pretrained embedding" << endl;
	LOG(INFO) << "Semantic: " << sem_found << "/" << semantic_token_dictionary_->GetNumForms()
	          << " words found in the pretrained embedding" << endl;
	semantic_parser_->LoadEmbedding(semantic_embedding_);
	parser_->LoadEmbedding(dependency_embedding_);
	delete semantic_embedding_;
	delete dependency_embedding_;
}

void SemanticPipe::BuildFormCount() {
	dependency_form_count_ = new unordered_map<int, int>();
	semantic_form_count_ = new unordered_map<int, int>();
	for (int i = 0; i < dependency_instances_.size(); i++) {
		Instance *instance = dependency_instances_[i];
		auto sentence = static_cast<DependencyInstanceNumeric *>(instance);
		const vector<int> form_ids = sentence->GetFormIds();
		for (int r = 0; r < form_ids.size(); ++r) {
			int form_id = form_ids[r];
			CHECK_NE(form_id, UNK_ID);
			if (dependency_form_count_->find(form_id) == dependency_form_count_->end()) {
				(*dependency_form_count_)[form_id] = 1;
			} else {
				(*dependency_form_count_)[form_id] += 1;
			}
		}
	}

	for (int i = 0; i < semantic_instances_.size(); i++) {
		Instance *instance = semantic_instances_[i];
		auto sentence = static_cast<SemanticInstanceNumeric *>(instance);
		const vector<int> form_ids = sentence->GetFormIds();
		for (int r = 0; r < form_ids.size(); ++r) {
			int form_id = form_ids[r];
			CHECK_NE(form_id, UNK_ID);
			if (semantic_form_count_->find(form_id) == semantic_form_count_->end()) {
				(*semantic_form_count_)[form_id] = 1;
			} else {
				(*semantic_form_count_)[form_id] += 1;
			}
		}
	}
}
