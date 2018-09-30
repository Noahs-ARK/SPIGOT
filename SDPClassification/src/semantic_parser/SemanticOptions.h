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

    const string &GetPrunerModelFilePath() { return file_pruner_model_; }

    double GetPrunerPosteriorThreshold() { return pruner_posterior_threshold_; }

    double GetPrunerMaxArguments() { return pruner_max_arguments_; }

    void train_off() { train_ = false; }

    void train_on() { train_ = true; }

    bool use_pretrained_embedding() { return use_pretrained_embedding_; }

    const string &GetEmbeddingFilePath(const string &task) {
	    if (task == "parser") {
		    return parser_file_embedding_;
	    } else if (task == "classification") {
		    return classification_file_embedding_;
	    }
    }

    int pruner_num_lstm_layers() { return pruner_num_lstm_layers_; }

    int lemma_dim() { return lemma_dim_; }

    int pos_dim() { return pos_dim_; }

    int pruner_lstm_dim() { return pruner_lstm_dim_; }

	int word_dim(const string &formalism) {
		if (formalism == "parser") {
			return parser_word_dim_;
		} else if (formalism == "classification") {
			return classification_word_dim_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	float dropout(const string &formalism) {
		if (formalism == "parser") {
			return parser_dropout_;
		} else if (formalism == "classification") {
			return classification_dropout_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	float word_dropout(const string &formalism) {
		if (formalism == "parser") {
			return parser_word_dropout_;
		} else if (formalism == "classification") {
			return classification_word_dropout_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	int num_lstm_layers(const string &formalism) {
		if (formalism == "parser") {
			return parser_num_lstm_layers_;
		} else if (formalism == "classification") {
			return classification_num_lstm_layers_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	int lstm_dim(const string &formalism) {
		if (formalism == "parser") {
			return parser_lstm_dim_;
		} else if (formalism == "classification") {
			return classification_lstm_dim_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

    int mlp_dim(const string &formalism) {
	    if (formalism == "parser") {
		    return parser_mlp_dim_;
	    } else if (formalism == "classification") {
		    return classification_mlp_dim_;
	    } else {
		    CHECK(false)
		    << "Unsupported formalism: " << formalism << ". Giving up..."
		    << endl;
	    }
    }

	float eta0(const string &formalism) {
		if (formalism == "parser") {
			return eta0_;
		} else if (formalism == "classification") {
			return classification_eta0_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	int halve(const string &formalism) {
		if (formalism == "parser") {
			return parser_halve_;
		} else if (formalism == "classification") {
			return classification_halve_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	string trainer(const string &formalism) {
		if (formalism == "parser") {
			return parser_trainer_;
		} else if (formalism == "classification") {
			return classification_trainer_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}


	bool train_pruner() { return train_pruner_; }

    int pruner_mlp_dim() { return pruner_mlp_dim_; }

    int parser_epochs() { return parser_epochs_; }

    bool update_parser() { return update_parser_; }

    float parser_fraction() { return parser_fraction_; }

    void train_pruner_off() { train_pruner_ = false; }

	const string GetTrainingFilePath(const string &formalism) {
		if (formalism == "parser") {
			return file_train_;
		} else if (formalism == "classification") {
			return classification_file_train_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	const string GetTestFilePath(const string &formalism) {
		if (formalism == "parser") {
			return file_test_;
		} else if (formalism == "classification") {
			return classification_file_test_;
		} else {
			CHECK(false)
			<< "Unsupported formalism: " << formalism << ". Giving up..."
			<< endl;
		}
	}

	const string &feature() { return feature_; };

	int batch_size() { return batch_size_; }

	bool proj() { return proj_; }

	bool pretrain_parser() { return pretrain_parser_; }

	const string &pretrained_parser_model() { return pretrained_parser_model_; }

    // temporary solution to weight_decay issue in dynet
    // TODO: save the weight_decay along with the model.
    uint64_t parser_num_updates_, classification_num_updates_;
    uint64_t pruner_num_updates_;
	float eta0_;
	float classification_eta0_;

protected:
    string classification_file_train_;
    string classification_file_test_;
    bool labeled_;
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
    string file_pruner_model_;
    double pruner_posterior_threshold_;
    int pruner_max_arguments_;
    bool use_pretrained_embedding_;
    string parser_file_embedding_;
	string classification_file_embedding_;
    int lemma_dim_;
    int pos_dim_;

	int parser_word_dim_;
    int parser_lstm_dim_, pruner_lstm_dim_;
    int parser_mlp_dim_, pruner_mlp_dim_;
    int parser_num_lstm_layers_, pruner_num_lstm_layers_;
	int parser_halve_;
    string parser_trainer_;
	float parser_dropout_;
	float parser_word_dropout_;

	int classification_word_dim_;
	int classification_lstm_dim_;
	int classification_mlp_dim_;
	int classification_num_lstm_layers_;
	int classification_halve_;
	string classification_trainer_;
	float classification_dropout_;
	float classification_word_dropout_;

    bool train_pruner_;
    bool update_parser_;
    float parser_fraction_;
    int parser_epochs_;
	string feature_;
	int batch_size_;
	bool proj_;
	bool pretrain_parser_;
	string pretrained_parser_model_;
};

#endif // SEMANTIC_OPTIONS_H_
