curr_dir="$PWD"
mkdir -p model
mkdir -p log
mkdir -p prediction

train_epochs=100
lemma_dim=50
word_dim=100
pos_dim=50
mlp_dim=100
lstm_dim=200
num_lstm_layers=2
trainer="adam"
eta0=0.001
word_dropout=0.25
batch_size=1
proj=true
struct_att=false

use_pretrained_embedding=true
language="english"
form="dm"

parser_file=${curr_dir}/build/syntactic_semantic
file_pretrained_embedding=${curr_dir}/../embedding/glove.100.pruned

dependency_file_train=${curr_dir}/../data/wsj/train
dependency_file_dev=${curr_dir}/../data//wsj/dev
dependency_file_test=${curr_dir}/../data/wsj/test
dependency_file_pruner=${curr_dir}/model/${language}_dependency.pruner

semantic_file_train=${curr_dir}/../data/${form}/train
semantic_file_dev=${curr_dir}/../data/${form}/dev
semantic_file_test=${curr_dir}/../data/${form}/test
semantic_file_pruner=${curr_dir}/model/${language}_${form}.pruner

file_model=${curr_dir}/model/${form}.init_lr${eta0}.lstm${lstm_dim}.mlp${mlp_dim}.pos${pos_dim}.wdrop${word_dropout}.proj${proj}.sa${struct_att}
semantic_file_prediction=${curr_dir}/prediction/${form}.init_lr${eta0}.lstm${lstm_dim}.mlp${mlp_dim}.pos${pos_dim}.wdrop${word_dropout}.proj${proj}.sa${struct_att}
dependency_file_prediction=${curr_dir}/prediction/${form}.init_lr${eta0}.lstm${lstm_dim}.mlp${mlp_dim}.pos${pos_dim}.wdrop${word_dropout}.proj${proj}.sa${struct_att}

${parser_file} --test --evaluate \
--dynet_mem 512 \
--dynet_weight_decay 1e-6 \
--train_epochs=${train_epochs} \
--dependency_file_pruner_model=${dependency_file_pruner} \
--dependency_file_train=${dependency_file_train} \
--dependency_file_test=${dependency_file_test} \
--dependency_file_prediction=${dependency_file_prediction} \
--semantic_file_train=${semantic_file_train} \
--semantic_file_test=${semantic_file_dev} \
--semantic_file_pruner_model=${semantic_file_pruner} \
--semantic_file_prediction=${semantic_file_prediction} \
--train_pruner=false \
--lemma_dim=${lemma_dim} \
--word_dim=${word_dim} \
--pos_dim=${pos_dim} \
--file_model=${file_model} \
--use_pretrained_embedding=${use_pretrained_embedding} \
--file_pretrained_embedding=${file_pretrained_embedding} \
--dependency_num_lstm_layers=${num_lstm_layers} \
--dependency_lstm_dim=${lstm_dim} \
--dependency_mlp_dim=${mlp_dim} \
--dependency_trainer=${trainer} \
--dependency_eta0=${eta0} \
--dependency_word_dropout=${word_dropout} \
--semantic_num_lstm_layers=${num_lstm_layers} \
--semantic_lstm_dim=${lstm_dim} \
--semantic_mlp_dim=${mlp_dim} \
--semantic_trainer=${trainer} \
--semantic_eta0=${eta0} \
--semantic_word_dropout=${word_dropout} \
--batch_size=${batch_size} \
--proj=${proj} \
--struct_att=${struct_att} 