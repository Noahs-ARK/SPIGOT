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

#ifndef SEMANTIC_OPTIONS_H_
#define SEMANTIC_OPTIONS_H_

#include "Options.h"
#include "Utils.h"

class SemanticOptions : public Options {
public:
	SemanticOptions() {};

	virtual ~SemanticOptions() {};

	// Serialization functions.
	void Load(FILE *fs);

	void Save(FILE *fs);

	// Initialization: set options based on the flags.
	void Initialize();

	// Get option values.
	const string &file_format() { return file_format_; }

	bool labeled() { return labeled_; }

	bool deterministic_labels() { return deterministic_labels_; }

	bool allow_self_loops() { return allow_self_loops_; }

	bool allow_root_predicate() { return allow_root_predicate_; }

	bool allow_unseen_predicates() { return allow_unseen_predicates_; }

	bool use_predicate_senses() { return use_predicate_senses_; }

	bool prune_labels() { return prune_labels_; }

	bool prune_labels_with_senses() { return prune_labels_with_senses_; }

	bool prune_labels_with_relation_paths() {
		return prune_labels_with_relation_paths_;
	}

	bool prune_distances() { return prune_distances_; }

	bool prune_basic() { return prune_basic_; }

	double GetPrunerPosteriorThreshold() { return pruner_posterior_threshold_; }

	double GetPrunerMaxArguments() { return pruner_max_arguments_; }

	void train_off() { train_ = false; }

	void train_on() { train_ = true; }

	float word_dropout(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_word_dropout_;
		} else if (formalism == "semantic") {
			return semantic_word_dropout_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	int num_lstm_layers(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_num_lstm_layers_;
		} else if (formalism == "semantic") {
			return semantic_num_lstm_layers_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	int lstm_dim(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_lstm_dim_;
		} else if (formalism == "semantic") {
			return semantic_lstm_dim_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	int mlp_dim(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_mlp_dim_;
		} else if (formalism == "semantic") {
			return semantic_mlp_dim_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	float eta0(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_eta0_;
		} else if (formalism == "semantic") {
			return semantic_eta0_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	bool use_pretrained_embedding() { return use_pretrained_embedding_; }

	const string &
	GetPretrainedEmbeddingFilePath() { return file_pretrained_embedding_; }

	int pruner_num_lstm_layers() { return pruner_num_lstm_layers_; }

	int word_dim() { return word_dim_; }

	int lemma_dim() { return lemma_dim_; }

	int pos_dim() { return pos_dim_; }

	int pruner_lstm_dim() { return pruner_lstm_dim_; }

	int pruner_mlp_dim() { return pruner_mlp_dim_; }

	string trainer(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_trainer_;
		} else if (formalism == "semantic") {
			return semantic_trainer_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	const bool train_pruner() {
		return train_pruner_;
	}

	const string GetTrainingFilePath(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_file_train_;
		} else if (formalism == "semantic") {
			return semantic_file_train_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	const string GetTestFilePath(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_file_test_;
		} else if (formalism == "semantic") {
			return semantic_file_test_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	const string GetOutputFilePath(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_file_prediction_;
		} else if (formalism == "semantic") {
			return semantic_file_prediction_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	const string &GetPrunerModelFilePath(const string &formalism) {
		if (formalism == "dependency") {
			return dependency_file_pruner_model_;
		} else if (formalism == "semantic") {
			return semantic_file_pruner_model_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	void train_pruner_on() { train_pruner_ = true; }

	void train_pruner_off() { train_pruner_ = false; }

	// TODO: temporary options for syntax
	bool projective() { return true; }

	int batch_size() { return batch_size_; }

	bool proj() { return proj_; }

	bool struct_att() { return struct_att_; }

	uint64_t dependency_num_updates_, semantic_num_updates_; // used for dealing with weight_decay in save/load.
	uint64_t dependency_pruner_num_updates_, semantic_pruner_num_updates_;
	float dependency_eta0_, semantic_eta0_;

protected:
	string dependency_file_train_;
	string dependency_file_test_;
	string dependency_file_pruner_model_;
	string dependency_file_prediction_;

	string semantic_file_train_;
	string semantic_file_test_;
	string semantic_file_pruner_model_;
	string semantic_file_prediction_;


	string file_format_;
	string model_type_;
	bool deterministic_labels_;
	bool allow_self_loops_;
	bool allow_root_predicate_;
	bool allow_unseen_predicates_;
	bool use_predicate_senses_;
	bool prune_labels_;
	bool prune_labels_with_senses_;
	bool prune_labels_with_relation_paths_;
	bool prune_distances_;
	bool prune_basic_;
	double pruner_posterior_threshold_;
	int pruner_max_arguments_;
	bool use_pretrained_embedding_;
	string file_pretrained_embedding_;
	int pruner_num_lstm_layers_;
	int word_dim_;
	int lemma_dim_;
	int pos_dim_;
	int pruner_lstm_dim_;
	int pruner_mlp_dim_;


	int dependency_lstm_dim_;
	int dependency_mlp_dim_;
	int dependency_num_lstm_layers_;
	string dependency_trainer_;
	float dependency_word_dropout_;


	int semantic_lstm_dim_;
	int semantic_mlp_dim_;
	int semantic_num_lstm_layers_;
	string semantic_trainer_;
	float semantic_word_dropout_;

	bool train_pruner_;
	bool labeled_;
	int batch_size_;
	bool proj_;
	bool struct_att_;
};

#endif // SEMANTIC_OPTIONS_H_
