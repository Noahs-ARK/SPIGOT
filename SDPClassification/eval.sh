#!/usr/bin/env bash
curr_dir="$PWD"
mkdir -p model
mkdir -p log
mkdir -p prediction

train_epochs=100
pruner_epochs=1
lemma_dim=50
pos_dim=50
use_pretrained_embedding=true
language="english"

formalism="dm"
batch_size=32
parser_epochs=0
feature="headword"
proj=true
pretrained_parser=true
pretrained_parser_model=${curr_dir}/model/semantic_parser.maxmargin
update_parser=true

parser_word_dim=100
parser_num_lstm_layers=2
parser_mlp_dim=100
parser_lstm_dim=200
parser_trainer="adam"
parser_eta0=0.001
parser_halve=5 # halving learning rate after each x epochs, set to 0 to disable
parser_dropout=0.0
parser_word_dropout=0.25

classification_word_dim=300
classification_num_lstm_layers=1
classification_mlp_dim=100
classification_lstm_dim=150
classification_trainer="adam"
classification_eta0=0.001
classification_halve=5 # halving learning rate after each x epochs, set to 0 to disable
classification_dropout=0.2
classification_word_dropout=0.5

parser_file=${curr_dir}/build/sdp_classification
parser_file_embedding=${curr_dir}/../embedding/glove.${parser_word_dim}.sst.pruned
classification_file_embedding=${curr_dir}/../embedding/glove.${classification_word_dim}.sst.pruned

file_pruner_model=${curr_dir}/model/${language}_semantic.pruner.model
file_train=${curr_dir}/../data/${formalism}/train
file_dev=${curr_dir}/../data/${formalism}/dev
file_test=${curr_dir}/../data/${formalism}/test
classification_file_train=${curr_dir}/../data/sst/train
classification_file_dev=${curr_dir}/../data/sst/dev
classification_file_test=${curr_dir}/../data/sst/test

parser_fraction=0.2
classification_mlp_dims="100"
classification_droptouts="0.1 0.2 0.3"

file_model=${curr_dir}/model/${classification_trainer}.init_lr${classification_eta0}.halve${classification_halve}.mlp${classification_mlp_dim}.lstm${classification_lstm_dim}.wdrop${classification_word_dropout}.dropout${classification_dropout}.proj${proj}
file_prediction=${curr_dir}/prediction/${classification_trainer}.init_lr${classification_eta0}.halve${classification_halve}.mlp${classification_mlp_dim}.lstm${classification_lstm_dim}.wdrop${classification_word_dropout}.dropout${classification_dropout}.proj${proj}

${parser_file} --test --evaluate \
--dynet_mem 512 \
--dynet_weight_decay 1e-6 \
--dynet-autobatch 0 \
--train_epochs=${train_epochs}  \
--parser_epochs=${parser_epochs} \
--parser_fraction=${parser_fraction} \
--file_train=${file_train} \
--file_test=${file_dev} \
--classification_file_train=${classification_file_train} \
--classification_file_test=${classification_file_test} \
--file_pruner_model=${file_pruner_model} \
--use_pretrained_embedding=${use_pretrained_embedding} \
--parser_file_embedding=${parser_file_embedding} \
--classification_file_embedding=${classification_file_embedding} \
--lemma_dim=${lemma_dim} \
--pos_dim=${pos_dim} \
--parser_word_dim=${parser_word_dim} \
--parser_num_lstm_layers=${parser_num_lstm_layers} \
--parser_lstm_dim=${parser_lstm_dim} \
--parser_mlp_dim=${parser_mlp_dim} \
--parser_dropout=${parser_dropout} \
--parser_word_dropout=${parser_word_dropout} \
--parser_trainer=${parser_trainer} \
--parser_eta0=${parser_eta0} \
--parser_halve=${parser_halve} \
--classification_word_dim=${classification_word_dim} \
--classification_num_lstm_layers=${classification_num_lstm_layers} \
--classification_lstm_dim=${classification_lstm_dim} \
--classification_mlp_dim=${classification_mlp_dim} \
--classification_dropout=${classification_dropout} \
--classification_word_dropout=${classification_word_dropout} \
--classification_trainer=${classification_trainer} \
--classification_eta0=${classification_eta0} \
--classification_halve=${classification_halve} \
--file_model=${file_model} \
--file_prediction=${file_prediction} \
--feature=${feature} \
--proj=${proj} \
--pretrained_parser=${pretrained_parser} \
--pretrained_parser_model=${pretrained_parser_model} \
--update_parser=${update_parser} \
--batch_size=${batch_size} \
--logtostderr \


