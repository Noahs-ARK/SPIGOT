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

#include "SemanticOptions.h"
#include "SerializationUtils.h"

using namespace std;

DEFINE_string(dependency_file_train, "",
              "Path to the file containing DM training data.");
DEFINE_string(dependency_file_test, "",
              "Path to the file containing DM test data.");
DEFINE_string(dependency_file_prediction, "",
              "Path to the file where the DM predictions are output.");
DEFINE_string(dependency_file_pruner_model, "",
              "Path to the file containing the pre-trained pruner model. Must "
		              "activate the flag --use_pretrained_pruner");

DEFINE_string(semantic_file_train, "",
              "Path to the file containing PAS training data.");
DEFINE_string(semantic_file_test, "",
              "Path to the file containing PAS test data.");
DEFINE_string(semantic_file_prediction, "",
              "Path to the file where the PAS predictions are output.");
DEFINE_string(semantic_file_pruner_model, "",
              "Path to the file containing the pre-trained pruner model. Must "
		              "activate the flag --use_pretrained_pruner");
DEFINE_string(file_format, "conll",
              "Format of the input file containing the data. Use ""conll"" for "
		              "the format used in CONLL 2008, ""sdp"" for the format in "
		              "SemEval 2014, and ""text"" for tokenized sentences"
		              "(one per line, with tokens separated by white-spaces.");
DEFINE_string(model_type, "basic", "Model type.");
DEFINE_bool(labeled, true,
            "True for training a parser with labeled arcs (if false, the "
		            "parser outputs just the backbone dependencies.)");
DEFINE_bool(deterministic_labels, true,
            "True for forcing a set of labels (found in the training set) to be "
		            "deterministic (i.e. to not occur in more than one argument for the "
		            "same predicate).");
DEFINE_bool(allow_self_loops, false,
            "True for allowing self-loops (a predicate being its own argument.)");
DEFINE_bool(allow_root_predicate, true,
            "True for allowing the root to be a predicate (useful for handling "
		            "top nodes).)");
DEFINE_bool(allow_unseen_predicates, false,
            "True for allowing an unseen predicate to be have a predicate sense "
		            "(assumes --use_predicate_senses=true.)");
DEFINE_bool(use_predicate_senses, false,
            "True for using predicate senses (e.g. temperature.01). If false, "
		            "any word can be a predicate and (eventual) sense information will "
		            "be ignored.");
DEFINE_bool(prune_labels, true,
            "True for pruning the set of possible labels taking into account "
		            "the labels that have occured for each pair of POS tags in the "
		            "training data.");
DEFINE_bool(prune_labels_with_senses, false,
            "True for pruning the set of possible labels taking into account "
		            "the predicate sense occurring in the training data.");
DEFINE_bool(prune_labels_with_relation_paths, false, //true,
            "True for pruning the set of possible labels taking into account "
		            "the labels that have occured for syntactic dependency relation "
		            "paths in the training data.");
DEFINE_bool(prune_distances, true,
            "True for pruning the set of possible left/right distances taking "
		            "into account the distances that have occured for each pair of POS "
		            "tags in the training data.");
DEFINE_bool(prune_basic, true,
            "True for using a basic pruner from a probabilistic first-order "
		            "model.");
DEFINE_string(file_pruner_model, "",
              "Path to the file containing the pre-trained pruner model. Must "
		              "activate the flag --use_pretrained_pruner");
DEFINE_double(pruner_posterior_threshold, 0.0001,
              "Posterior probability threshold for an arc to be pruned, in basic "
		              "pruning. For each word p, if "
		              "P(p,a) < pruner_posterior_threshold * P(p,a'), "
		              "where a' is the best scored argument, then (p,a) will be pruned out.");
DEFINE_int32(pruner_max_arguments, 20,
             "Maximum number of possible arguments for a given word, in basic "
		             "pruning.");
DEFINE_bool(use_pretrained_embedding, false,
            "Optional use of pretrained embedding.");
DEFINE_string(file_pretrained_embedding, "path_to_embedding",
              "If using pretrained embedding, provide the path");
DEFINE_int32(pruner_num_lstm_layers, 1, "Number of layers of biLSTM encoder.");
DEFINE_int32(pruner_lstm_dim, 32, "Dimension of biLSTM.");
DEFINE_int32(pruner_mlp_dim, 32, "Dimension of MLP.");
DEFINE_int32(word_dim, 100, "Dimension of pretrained word embedding.");
DEFINE_int32(lemma_dim, 50, "Dimension of word embedding.");
DEFINE_int32(pos_dim, 50, "Dimension of POS tag embedding.");

DEFINE_int32(dependency_num_lstm_layers, 2, "Number of layers of biLSTM encoder.");
DEFINE_int32(dependency_lstm_dim, 200, "Dimension of biLSTM.");
DEFINE_int32(dependency_mlp_dim, 100, "Dimension of MLP.");
DEFINE_string(dependency_trainer, "adadelta",
              "Trainer to use: sgd_momentum, adam, adadelta");
DEFINE_double(dependency_eta0, 0.001, "dependency eta0");
DEFINE_double(dependency_word_dropout, 0.0, "Word dropout rate.");

DEFINE_int32(semantic_num_lstm_layers, 2, "Number of layers of biLSTM encoder.");
DEFINE_int32(semantic_lstm_dim, 200, "Dimension of biLSTM.");
DEFINE_int32(semantic_mlp_dim, 100, "Dimension of MLP.");
DEFINE_string(semantic_trainer, "adam",
              "Trainer to use: sgd, adam, adadelta");
DEFINE_double(semantic_eta0, 0.001, "srl eta0");
DEFINE_double(semantic_word_dropout, 0.0, "Word dropout rate.");
DEFINE_bool(train_pruner, false,
            "True if using a pre-trained basic pruner. Must specify the file "
		            "path through --file_pruner_model. If this flag is set to false "
		            "and train=true and prune_basic=true, a pruner will be trained "
		            "along with the parser.");
DEFINE_uint64(dependency_num_updates, 0,
              "used for dealint with weight_decay in save/load.");
DEFINE_uint64(semantic_num_updates, 0,
              "used for dealint with weight_decay in save/load.");
DEFINE_uint64(dependency_pruner_num_updates, 1259,
              "used for dealint with weight_decay in save/load.");
DEFINE_uint64(semantic_pruner_num_updates, 3186,
              "used for dealint with weight_decay in save/load.");
DEFINE_int32(batch_size, 1, "");
DEFINE_bool(proj, false, "");
DEFINE_bool(struct_att, false, "");

// Save current option flags to the model file.
void SemanticOptions::Save(FILE *fs) {
	Options::Save(fs);

	bool success;
	success = WriteString(fs, model_type_);
	CHECK(success);
	success = WriteBool(fs, labeled_);
	CHECK(success);
	success = WriteBool(fs, deterministic_labels_);
	CHECK(success);
	success = WriteBool(fs, allow_self_loops_);
	CHECK(success);
	success = WriteBool(fs, allow_root_predicate_);
	CHECK(success);
	success = WriteBool(fs, allow_unseen_predicates_);
	CHECK(success);
	success = WriteBool(fs, use_predicate_senses_);
	CHECK(success);
	success = WriteBool(fs, prune_labels_);
	CHECK(success);
	success = WriteBool(fs, prune_labels_with_senses_);
	CHECK(success);
	success = WriteBool(fs, prune_labels_with_relation_paths_);
	CHECK(success);
	success = WriteBool(fs, prune_distances_);
	CHECK(success);
	success = WriteBool(fs, prune_basic_);
	CHECK(success);
	success = WriteDouble(fs, pruner_posterior_threshold_);
	CHECK(success);
	success = WriteInteger(fs, pruner_max_arguments_);
	CHECK(success);
	success = WriteInteger(fs, word_dim_);
	CHECK(success);
	success = WriteInteger(fs, lemma_dim_);
	CHECK(success);
	success = WriteInteger(fs, pos_dim_);
	CHECK(success);
	success = WriteInteger(fs, dependency_num_lstm_layers_);
	CHECK(success);
	success = WriteInteger(fs, dependency_lstm_dim_);
	CHECK(success);
	success = WriteInteger(fs, dependency_mlp_dim_);
	CHECK(success);
	success = WriteDouble(fs, dependency_word_dropout_);
	CHECK(success);
	success = WriteInteger(fs, semantic_num_lstm_layers_);
	CHECK(success);
	success = WriteInteger(fs, semantic_lstm_dim_);
	CHECK(success);
	success = WriteInteger(fs, semantic_mlp_dim_);
	CHECK(success);
	success = WriteDouble(fs, semantic_word_dropout_);
	CHECK(success);
	success = WriteUINT64(fs, dependency_num_updates_);
	CHECK(success);
	success = WriteUINT64(fs, semantic_num_updates_);
	CHECK(success);
	success = WriteUINT64(fs, dependency_pruner_num_updates_);
	CHECK(success);
	success = WriteUINT64(fs, semantic_pruner_num_updates_);
	CHECK(success);
	success = WriteBool(fs, proj_);
	CHECK(success);
	success = WriteBool(fs, struct_att_);
	CHECK(success);
}

// Load current option flags to the model file.
// Note: this will override the user-specified flags.
void SemanticOptions::Load(FILE *fs) {
	Options::Load(fs);

	bool success;
	success = ReadString(fs, &FLAGS_model_type);
	CHECK(success);
	LOG(INFO) << "Setting --model_type=" << FLAGS_model_type;
	success = ReadBool(fs, &FLAGS_labeled);
	CHECK(success);
	LOG(INFO) << "Setting --labeled=" << FLAGS_labeled;
	success = ReadBool(fs, &FLAGS_deterministic_labels);
	CHECK(success);
	LOG(INFO) << "Setting --deterministic_labels="
	          << FLAGS_deterministic_labels;
	success = ReadBool(fs, &FLAGS_allow_self_loops);
	CHECK(success);
	LOG(INFO) << "Setting --allow_self_loops=" << FLAGS_allow_self_loops;
	success = ReadBool(fs, &FLAGS_allow_root_predicate);
	CHECK(success);
	LOG(INFO) << "Setting --allow_root_predicate="
	          << FLAGS_allow_root_predicate;
	success = ReadBool(fs, &FLAGS_allow_unseen_predicates);
	CHECK(success);
	LOG(INFO) << "Setting --allow_unseen_predicates="
	          << FLAGS_allow_unseen_predicates;
	success = ReadBool(fs, &FLAGS_use_predicate_senses);
	CHECK(success);
	LOG(INFO) << "Setting --use_predicate_senses="
	          << FLAGS_use_predicate_senses;
	success = ReadBool(fs, &FLAGS_prune_labels);
	CHECK(success);
	LOG(INFO) << "Setting --prune_labels=" << FLAGS_prune_labels;
	success = ReadBool(fs, &FLAGS_prune_labels_with_senses);
	CHECK(success);
	LOG(INFO) << "Setting --prune_labels_with_senses="
	          << FLAGS_prune_labels_with_senses;
	success = ReadBool(fs, &FLAGS_prune_labels_with_relation_paths);
	CHECK(success);
	LOG(INFO) << "Setting --prune_labels_with_relation_paths="
	          << FLAGS_prune_labels_with_relation_paths;
	success = ReadBool(fs, &FLAGS_prune_distances);
	CHECK(success);
	LOG(INFO) << "Setting --prune_distances=" << FLAGS_prune_distances;
	success = ReadBool(fs, &FLAGS_prune_basic);
	CHECK(success);
	LOG(INFO) << "Setting --prune_basic=" << FLAGS_prune_basic;
	success = ReadDouble(fs, &FLAGS_pruner_posterior_threshold);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_posterior_threshold="
	          << FLAGS_pruner_posterior_threshold;
	success = ReadInteger(fs, &FLAGS_pruner_max_arguments);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_max_arguments="
	          << FLAGS_pruner_max_arguments;

	success = ReadInteger(fs, &FLAGS_word_dim);
	CHECK(success);
	LOG(INFO) << "Setting --word_dim="
	          << FLAGS_word_dim;

	success = ReadInteger(fs, &FLAGS_lemma_dim);
	CHECK(success);
	LOG(INFO) << "Setting --lemma_dim="
	          << FLAGS_lemma_dim;

	success = ReadInteger(fs, &FLAGS_pos_dim);
	CHECK(success);
	LOG(INFO) << "Setting --pos_dim="
	          << FLAGS_pos_dim;

	success = ReadInteger(fs, &FLAGS_dependency_num_lstm_layers);
	CHECK(success);
	LOG(INFO) << "Setting --dependency_num_lstm_layers="
	          << FLAGS_dependency_num_lstm_layers;

	success = ReadInteger(fs, &FLAGS_dependency_lstm_dim);
	CHECK(success);
	LOG(INFO) << "Setting --dependency_lstm_dim="
	          << FLAGS_dependency_lstm_dim;

	success = ReadInteger(fs, &FLAGS_dependency_mlp_dim);
	CHECK(success);
	LOG(INFO) << "Setting --dependency_mlp_dim="
	          << FLAGS_dependency_mlp_dim;

	success = ReadDouble(fs, &FLAGS_dependency_word_dropout);
	CHECK(success);
	LOG(INFO) << "Setting --dependency_word_dropout="
	          << FLAGS_dependency_word_dropout;

	success = ReadInteger(fs, &FLAGS_semantic_num_lstm_layers);
	CHECK(success);
	LOG(INFO) << "Setting --semantic_num_lstm_layers="
	          << FLAGS_semantic_num_lstm_layers;

	success = ReadInteger(fs, &FLAGS_semantic_lstm_dim);
	CHECK(success);
	LOG(INFO) << "Setting --semantic_lstm_dim="
	          << FLAGS_semantic_lstm_dim;

	success = ReadInteger(fs, &FLAGS_semantic_mlp_dim);
	CHECK(success);
	LOG(INFO) << "Setting --semantic_mlp_dim="
	          << FLAGS_semantic_mlp_dim;

	success = ReadDouble(fs, &FLAGS_semantic_word_dropout);
	CHECK(success);
	LOG(INFO) << "Setting --semantic_word_dropout="
	          << FLAGS_semantic_word_dropout;

	success = ReadUINT64(fs, &FLAGS_dependency_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --dependency_num_updates="
	          << FLAGS_dependency_num_updates;

	success = ReadUINT64(fs, &FLAGS_semantic_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --semantic_num_updates="
	          << FLAGS_semantic_num_updates;

	success = ReadUINT64(fs, &FLAGS_dependency_pruner_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --dependencyc_pruner_num_updates="
	          << FLAGS_dependency_pruner_num_updates;

	success = ReadUINT64(fs, &FLAGS_semantic_pruner_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --semantic_pruner_num_updates="
	          << FLAGS_semantic_pruner_num_updates;

	success = ReadBool(fs, &FLAGS_proj);
	CHECK(success);
	LOG(INFO) << "Setting --proj="
	          << FLAGS_proj;

	success = ReadBool(fs, &FLAGS_struct_att);
	CHECK(success);
	LOG(INFO) << "Setting --struct_att="
	          << FLAGS_struct_att;

	Initialize();
}

void SemanticOptions::Initialize() {
	Options::Initialize();
	dependency_file_train_ = FLAGS_dependency_file_train;
	dependency_file_test_ = FLAGS_dependency_file_test;
	dependency_file_prediction_ = FLAGS_dependency_file_prediction;
	dependency_file_pruner_model_ = FLAGS_dependency_file_pruner_model;

	semantic_file_train_ = FLAGS_semantic_file_train;
	semantic_file_test_ = FLAGS_semantic_file_test;
	semantic_file_prediction_ = FLAGS_semantic_file_prediction;
	semantic_file_pruner_model_ = FLAGS_semantic_file_pruner_model;

	file_format_ = FLAGS_file_format;

	model_type_ = FLAGS_model_type;
	labeled_ = FLAGS_labeled;
	deterministic_labels_ = FLAGS_deterministic_labels;
	allow_self_loops_ = FLAGS_allow_self_loops;
	allow_root_predicate_ = FLAGS_allow_root_predicate;
	allow_unseen_predicates_ = FLAGS_allow_unseen_predicates;
	use_predicate_senses_ = FLAGS_use_predicate_senses;
	prune_labels_ = FLAGS_prune_labels;
	prune_labels_with_senses_ = FLAGS_prune_labels_with_senses;
	prune_labels_with_relation_paths_ =
			FLAGS_prune_labels_with_relation_paths;
	prune_distances_ = FLAGS_prune_distances;
	prune_basic_ = FLAGS_prune_basic;
	pruner_posterior_threshold_ = FLAGS_pruner_posterior_threshold;
	pruner_max_arguments_ = FLAGS_pruner_max_arguments;
	use_pretrained_embedding_ = FLAGS_use_pretrained_embedding;
	file_pretrained_embedding_ = FLAGS_file_pretrained_embedding;
	train_pruner_ = FLAGS_train_pruner;
	dependency_pruner_num_updates_ = FLAGS_dependency_pruner_num_updates;
	semantic_pruner_num_updates_ = FLAGS_semantic_pruner_num_updates;

	pruner_num_lstm_layers_ = FLAGS_pruner_num_lstm_layers;
	word_dim_ = FLAGS_word_dim;
	lemma_dim_ = FLAGS_lemma_dim;
	pos_dim_ = FLAGS_pos_dim;
	pruner_lstm_dim_ = FLAGS_pruner_lstm_dim;
	pruner_mlp_dim_ = FLAGS_pruner_mlp_dim;
	semantic_trainer_ = FLAGS_semantic_trainer;

	dependency_trainer_ = FLAGS_dependency_trainer;
	dependency_eta0_ = FLAGS_dependency_eta0;
	dependency_num_lstm_layers_ = FLAGS_dependency_num_lstm_layers;
	dependency_lstm_dim_ = FLAGS_dependency_lstm_dim;
	dependency_mlp_dim_ = FLAGS_dependency_mlp_dim;
	dependency_word_dropout_ = FLAGS_dependency_word_dropout;

	semantic_trainer_ = FLAGS_semantic_trainer;
	semantic_eta0_ = FLAGS_semantic_eta0;
	semantic_num_lstm_layers_ = FLAGS_semantic_num_lstm_layers;
	semantic_lstm_dim_ = FLAGS_semantic_lstm_dim;
	semantic_mlp_dim_ = FLAGS_semantic_mlp_dim;
	semantic_word_dropout_ = FLAGS_semantic_word_dropout;

	batch_size_ = FLAGS_batch_size;
	proj_ = FLAGS_proj;
	struct_att_ = FLAGS_struct_att;
	dependency_num_updates_ = FLAGS_dependency_num_updates;
	semantic_num_updates_ = FLAGS_semantic_num_updates;

	// Enable the parts corresponding to the model type.
	string model_type = FLAGS_model_type;
	if (model_type == "basic") {
		model_type = "af";
	} else {
		CHECK(false);
	}
	vector<string> enabled_parts;
	bool use_arc_factored = false;
	StringSplit(model_type, "+", &enabled_parts, true);
	for (int i = 0; i < enabled_parts.size(); ++i) {
		if (enabled_parts[i] == "af") {
			use_arc_factored = true;
			LOG(INFO) << "Arc factored parts enabled.";
		} else {
			CHECK(false) << "Unknown part in model type: " << enabled_parts[i];
		}
	}

	CHECK(use_arc_factored) << "Arc-factored parts are mandatory in model type";
}
