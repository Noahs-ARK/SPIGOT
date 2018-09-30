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
DEFINE_string(classification_file_train, "",
              "Path to the file containing the classification training data.");
DEFINE_string(classification_file_test, "",
              "Path to the file containing the classification test data.");
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
DEFINE_bool(use_pretrained_pruner, false,
            "True if using a pre-trained basic pruner. Must specify the file "
		            "path through --file_pruner_model. If this flag is set to false "
		            "and train=true and prune_basic=true, a pruner will be trained "
		            "along with the parser.");
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
DEFINE_bool(pruner_labeled, false,
            "True if pruner is a labeled parser. Currently, must be set to false.");

DEFINE_bool(use_pretrained_embedding, false, "Optional use of pretrained embedding.");
DEFINE_string(parser_file_embedding, "path_to_embedding", "If using pretrained embedding, provide the path");
DEFINE_string(classification_file_embedding, "path_to_embedding", "If using pretrained embedding, provide the path");

DEFINE_int32(pruner_num_lstm_layers, 1, "Pruner: number of layers of biLSTM encoder.");
DEFINE_int32(pruner_lstm_dim, 32, "Pruner: dimension of biLSTM.");
DEFINE_int32(pruner_mlp_dim, 32, "Pruner: dimension of MLP.");
DEFINE_int32(lemma_dim, 50, "Dimension of lemma embedding.");
DEFINE_int32(pos_dim, 50, "Dimension of POS tag embedding.");

DEFINE_int32(parser_num_lstm_layers, 1, "Number of layers of biLSTM encoder.");
DEFINE_int32(parser_lstm_dim, 200, "Dimension of biLSTM.");
DEFINE_int32(parser_word_dim, 100, "Dimension of word embedding.");
DEFINE_int32(parser_mlp_dim, 100, "Dimension of MLP.");
DEFINE_double(parser_eta0, 0.001, "eta0");
DEFINE_int32(parser_halve, 0, "scheduled halving. set to 0 to disable");
DEFINE_double(parser_dropout, 0.0, "Dropout rate.");
DEFINE_double(parser_word_dropout, 0.0, "Word dropout rate.");
DEFINE_string(parser_trainer, "adam", "Trainer to use: sgd, adam, adadelta");

DEFINE_int32(classification_num_lstm_layers, 1, "Number of layers of biLSTM encoder.");
DEFINE_int32(classification_lstm_dim, 200, "Dimension of biLSTM.");
DEFINE_int32(classification_word_dim, 100, "Dimension of word embedding.");
DEFINE_int32(classification_mlp_dim, 100, "Dimension of MLP.");
DEFINE_double(classification_eta0, 0.001, "eta0");
DEFINE_int32(classification_halve, 0, "scheduled halving. set to 0 to disable");
DEFINE_double(classification_dropout, 0.0, "Dropout rate.");
DEFINE_double(classification_word_dropout, 0.0, "Word dropout rate.");
DEFINE_string(classification_trainer, "adam", "Trainer to use: sgd, adam, adadelta");

DEFINE_bool(train_pruner, false, "");
DEFINE_uint64(parser_num_updates, 0, "used for dealint with weight_decay in save/load.");
DEFINE_uint64(classification_num_updates, 0, "used for dealint with weight_decay in save/load.");
DEFINE_uint64(pruner_num_updates, 4248, "used for dealint with weight_decay in save/load.");
DEFINE_int32(parser_epochs, 10, "Number of epoches to pre-train the parser.");
DEFINE_bool(update_parser, true, "Whether to update the parser parameters when training the classifier.");
DEFINE_double(parser_fraction, 0.2,
              "Fraction of parser data use when training the classifier (if parser if not fixed).");
DEFINE_string(feature, "average", "average/headword");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_bool(proj, false, "");
DEFINE_bool(pretrained_parser, false, "");
DEFINE_string(pretrained_parser_model, "", "");
// Save current option flags to the model file.
void SemanticOptions::Save(FILE *fs) {
	Options::Save(fs);

	bool success;
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
	success = WriteInteger(fs, lemma_dim_);
	CHECK(success);
	success = WriteInteger(fs, pos_dim_);
	CHECK(success);

	success = WriteInteger(fs, parser_word_dim_);
	CHECK(success);
	success = WriteInteger(fs, parser_num_lstm_layers_);
	CHECK(success);
	success = WriteInteger(fs, parser_lstm_dim_);
	CHECK(success);
	success = WriteInteger(fs, parser_mlp_dim_);
	CHECK(success);

	success = WriteInteger(fs, classification_word_dim_);
	CHECK(success);
	success = WriteInteger(fs, classification_num_lstm_layers_);
	CHECK(success);
	success = WriteInteger(fs, classification_lstm_dim_);
	CHECK(success);
	success = WriteInteger(fs, classification_mlp_dim_);
	CHECK(success);
	success = WriteUINT64(fs, parser_num_updates_);
	CHECK(success);
	success = WriteUINT64(fs, classification_num_updates_);
	CHECK(success);
	success = WriteUINT64(fs, pruner_num_updates_);
	CHECK(success);
	success = WriteString(fs, feature_);
	CHECK(success);
	success = WriteBool(fs, proj_);
	CHECK(success);
}

// Load current option flags to the model file.
// Note: this will override the user-specified flags.
void SemanticOptions::Load(FILE *fs) {
	Options::Load(fs);

	bool success;
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

	success = ReadInteger(fs, &FLAGS_lemma_dim);
	CHECK(success);
	LOG(INFO) << "Setting --lemma_dim="
	          << FLAGS_lemma_dim;

	success = ReadInteger(fs, &FLAGS_pos_dim);
	CHECK(success);
	LOG(INFO) << "Setting --pos_dim="
	          << FLAGS_pos_dim;

	success = ReadInteger(fs, &FLAGS_parser_word_dim);
	CHECK(success);
	LOG(INFO) << "Setting --parser_word_dim="
	          << FLAGS_parser_word_dim;

	success = ReadInteger(fs, &FLAGS_parser_num_lstm_layers);
	CHECK(success);
	LOG(INFO) << "Setting --parser_num_lstm_layers="
	          << FLAGS_parser_num_lstm_layers;

	success = ReadInteger(fs, &FLAGS_parser_lstm_dim);
	CHECK(success);
	LOG(INFO) << "Setting --parser_lstm_dim="
	          << FLAGS_parser_lstm_dim;

	success = ReadInteger(fs, &FLAGS_parser_mlp_dim);
	CHECK(success);
	LOG(INFO) << "Setting --parser_mlp_dim="
	          << FLAGS_parser_mlp_dim;

	success = ReadInteger(fs, &FLAGS_classification_word_dim);
	CHECK(success);
	LOG(INFO) << "Setting --classification_word_dim="
	          << FLAGS_classification_word_dim;

	success = ReadInteger(fs, &FLAGS_classification_num_lstm_layers);
	CHECK(success);
	LOG(INFO) << "Setting --classification_num_lstm_layers="
	          << FLAGS_classification_num_lstm_layers;

	success = ReadInteger(fs, &FLAGS_classification_lstm_dim);
	CHECK(success);
	LOG(INFO) << "Setting --classification_lstm_dim="
	          << FLAGS_classification_lstm_dim;

	success = ReadInteger(fs, &FLAGS_classification_mlp_dim);
	CHECK(success);
	LOG(INFO) << "Setting --classification_mlp_dim="
	          << FLAGS_classification_mlp_dim;

	success = ReadUINT64(fs, &FLAGS_parser_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --parser_num_updates="
	          << FLAGS_parser_num_updates;

	success = ReadUINT64(fs, &FLAGS_classification_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --classification_num_updates="
	          << FLAGS_classification_num_updates;

	success = ReadUINT64(fs, &FLAGS_pruner_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_num_updates="
	          << FLAGS_pruner_num_updates;

	success = ReadString(fs, &FLAGS_feature);
	CHECK(success);
	LOG(INFO) << "Setting --feature="
	          << FLAGS_feature;

	success = ReadBool(fs, &FLAGS_proj);
	CHECK(success);
	LOG(INFO) << "Setting --proj="
	          << FLAGS_proj;
}

void SemanticOptions::Initialize() {
	Options::Initialize();

	classification_file_train_ = FLAGS_classification_file_train;
	classification_file_test_ = FLAGS_classification_file_test;
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
	file_pruner_model_ = FLAGS_file_pruner_model;
	pruner_posterior_threshold_ = FLAGS_pruner_posterior_threshold;
	pruner_max_arguments_ = FLAGS_pruner_max_arguments;
	use_pretrained_embedding_ = FLAGS_use_pretrained_embedding;
	parser_file_embedding_ = FLAGS_parser_file_embedding;
	classification_file_embedding_ = FLAGS_classification_file_embedding;

	lemma_dim_ = FLAGS_lemma_dim;
	pos_dim_ = FLAGS_pos_dim;

	parser_word_dim_ = FLAGS_parser_word_dim;
	parser_num_lstm_layers_ = FLAGS_parser_num_lstm_layers;
	parser_lstm_dim_ = FLAGS_parser_lstm_dim;
	parser_mlp_dim_ = FLAGS_parser_mlp_dim;
	parser_trainer_ = FLAGS_parser_trainer;
	eta0_ = FLAGS_parser_eta0;
	parser_halve_ = FLAGS_parser_halve;
	parser_dropout_ = FLAGS_parser_dropout;
	parser_word_dropout_ = FLAGS_parser_word_dropout;

	classification_word_dim_ = FLAGS_classification_word_dim;
	classification_num_lstm_layers_ = FLAGS_classification_num_lstm_layers;
	classification_lstm_dim_ = FLAGS_classification_lstm_dim;
	classification_mlp_dim_ = FLAGS_classification_mlp_dim;
	classification_trainer_ = FLAGS_classification_trainer;
	classification_eta0_ = FLAGS_classification_eta0;
	classification_halve_ = FLAGS_classification_halve;
	classification_dropout_ = FLAGS_classification_dropout;
	classification_word_dropout_ = FLAGS_classification_word_dropout;

	pruner_num_lstm_layers_ = FLAGS_pruner_num_lstm_layers;
	pruner_lstm_dim_ = FLAGS_pruner_lstm_dim;
	pruner_mlp_dim_ = FLAGS_pruner_mlp_dim;
	train_pruner_ = FLAGS_train_pruner;
	pruner_num_updates_ = FLAGS_pruner_num_updates;

	feature_ = FLAGS_feature;
	parser_epochs_ = FLAGS_parser_epochs;
	update_parser_ = FLAGS_update_parser;
	parser_fraction_ = FLAGS_parser_fraction;
	batch_size_ = FLAGS_batch_size;
	proj_ = FLAGS_proj;
	parser_num_updates_ =FLAGS_parser_num_updates;
	classification_num_updates_ = FLAGS_classification_num_updates;
	pretrain_parser_ = FLAGS_pretrained_parser;
	pretrained_parser_model_ = FLAGS_pretrained_parser_model;
}
