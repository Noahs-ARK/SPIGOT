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

#ifndef SemanticPipe_H_
#define SemanticPipe_H_

#include "Pipe.h"
#include "SemanticOptions.h"
#include "SemanticDictionary.h"
#include "DependencyReader.h"
#include "DependencyDictionary.h"
#include "TokenDictionary.h"
#include "DependencyInstanceNumeric.h"
#include "DependencyWriter.h"
#include "DependencyPart.h"
#include "DependencyPruner.h"
#include "SemanticReader.h"
#include "SemanticDictionary.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticWriter.h"
#include "SemanticPart.h"
#include "SemanticPruner.h"
#include "dynet/io.h"
#include "Dependency.h"
#include "SemanticParser.h"
#include "StructuredAttention.h"

class SemanticDecoder;
class SemanticPipe : public Pipe {
public:
    SemanticPipe(Options *semantic_options) : Pipe(semantic_options) {
        dependency_token_dictionary_ = nullptr;
	    semantic_token_dictionary_ = nullptr;
        depdendency_reader_ = nullptr;
        semantic_reader_ = nullptr;
        dependency_writer_ = nullptr;
        semantic_writer_ = nullptr;
        dependency_dictionary_ = nullptr;
        semantic_dictionary_ = nullptr;

	    semantic_trainer_ = semantic_pruner_trainer_ = nullptr;
	    semantic_model_ = semantic_pruner_model_ = nullptr;
	    semantic_pruner_ = nullptr;

	    dependency_trainer_ = dependency_pruner_trainer_ = nullptr;
	    dependency_model_ = dependency_pruner_model_ = nullptr;
	    dependency_pruner_ = nullptr;

	    parser_ = nullptr;
	    semantic_parser_ = nullptr;
        semantic_pruner_ = nullptr;
    }

    virtual ~SemanticPipe() {
        delete dependency_token_dictionary_;
	    delete semantic_token_dictionary_;
        delete depdendency_reader_;
        delete semantic_reader_;
        delete dependency_writer_;
        delete semantic_writer_;
        delete dependency_dictionary_;
        delete semantic_dictionary_;
	    
	    delete semantic_pruner_trainer_; delete semantic_pruner_model_; delete semantic_pruner_;
	    delete semantic_trainer_; delete semantic_model_;
	    delete dependency_pruner_trainer_; delete dependency_pruner_model_; delete dependency_pruner_;
	    delete dependency_trainer_; delete dependency_model_;
        delete parser_; delete semantic_parser_;
    }

    DependencyReader *GetDependencyReader() {
        return static_cast<DependencyReader *> (depdendency_reader_);
    }

    SemanticReader *GetSemanticReader() {
        return static_cast<SemanticReader *> (semantic_reader_);
    }

    DependencyDictionary *GetDependencyDictionary() {
        return static_cast<DependencyDictionary *> (dependency_dictionary_);
    }

	SemanticDictionary *GetSemanticDictionary() {
		return static_cast<SemanticDictionary *> (semantic_dictionary_);
	}

	DependencyDecoder *GetDepdendencyDecoder() {
        return static_cast<DependencyDecoder *>(dependency_decoder_);
    }

	SemanticDecoder *GetSemanticDecoder() {
		return static_cast<SemanticDecoder *>(semantic_decoder_);
	}

    SemanticOptions *GetSemanticOptions() {
        return static_cast<SemanticOptions *>(options_);
    }

    void Initialize();

    void LoadPretrainedEmbedding();

    void Train();

    void TrainPruner();

    double TrainEpoch(vector<int> &dependency_idxs, vector<int> &semantic_idxs,
                      int epoch, double &best_F1);

    double TrainPrunerEpoch(const string &formalism, const vector<int> &idxs, int epoch);

    void Test();

    void Run(double &unlabeled_F1, double &labeled_F1);

    void LoadNeuralModel();

    void SaveNeuralModel();

    void LoadPruner(const std::string &file_name);

    void SavePruner(const std::string &file_name);

    void BuildFormCount();

protected:

    void CreateDictionary() {
        dependency_dictionary_ = new DependencyDictionary(this);
        static_cast<DependencyDictionary *> (dependency_dictionary_)
		        ->SetTokenDictionary(dependency_token_dictionary_);

        semantic_dictionary_ = new SemanticDictionary(this);
        static_cast<SemanticDictionary *> (semantic_dictionary_)
		        ->SetTokenDictionary(semantic_token_dictionary_);
    }

    void CreateReader() {
        depdendency_reader_ = new DependencyReader();
        semantic_reader_ = new SemanticReader(options_);
    }

    void CreateWriter() {
        dependency_writer_ = new DependencyWriter();
        semantic_writer_ = new SemanticWriter(options_);
    }

    void CreateDecoder() {
	    dependency_decoder_ = new DependencyDecoder(this);
	    semantic_decoder_ = new SemanticDecoder(this);
    }

    Parts *CreateParts(const string &formalism) {
        if (formalism == "dependency") {
            return new DependencyParts;
        } else if (formalism == "semantic") {
            return new SemanticParts;
        } else {
            CHECK(false) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        }
    }

    void CreateTokenDictionary() {
        semantic_token_dictionary_ = new TokenDictionary(this);
	    dependency_token_dictionary_ = new TokenDictionary(this);
    }

	void CreateDependencyDictionary() {
		dependency_dictionary_ = new DependencyDictionary(this);
	}

    void PreprocessData();

    void CreateInstances(const string &formalism) {
        if (formalism != "dependency" && formalism != "semantic") {
            CHECK(false) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        }
        timeval start, end;
        gettimeofday(&start, nullptr);
        SemanticOptions *semantic_options = GetSemanticOptions();
        DeleteInstances(formalism);
        if (formalism == "dependency") {
            LOG(INFO) << "Creating parser instances...";
            depdendency_reader_->Open(semantic_options->GetTrainingFilePath(formalism));
            Instance *instance = static_cast<DependencyReader *> (depdendency_reader_)->GetNext();
            while (instance) {
	            Instance *formatted_instance = GetFormattedInstance(formalism, instance);
	            dependency_instances_.push_back(formatted_instance);
	            if (instance != formatted_instance) delete instance;
                instance = static_cast<DependencyReader *> (depdendency_reader_)->GetNext();
            }
            depdendency_reader_->Close();


	        depdendency_reader_->Open(semantic_options->GetTestFilePath(formalism));
	        instance = static_cast<DependencyReader *> (depdendency_reader_)->GetNext();
	        while (instance) {
		        dependency_dev_instances_.push_back(instance);
		        instance = static_cast<DependencyReader *> (depdendency_reader_)->GetNext();
	        }
	        depdendency_reader_->Close();

            LOG(INFO) << "Number of instances: " << dependency_instances_.size();
        } else if (formalism == "semantic") {
            LOG(INFO) << "Creating Semantic instances...";
            semantic_reader_->Open(semantic_options->GetTrainingFilePath(formalism));
            Instance *instance = static_cast<SemanticReader *> (semantic_reader_)->GetNext();
            while (instance) {
	            Instance *formatted_instance = GetFormattedInstance(formalism, instance);
	            semantic_instances_.push_back(formatted_instance);

	            formatted_instance = GetFormattedInstance("dependency", instance);
	            semantic_dep_instances_.push_back(formatted_instance);
	            if (instance != formatted_instance) delete instance;

                instance = static_cast<SemanticReader *> (semantic_reader_)->GetNext();
            }
            semantic_reader_->Close();

	        semantic_reader_->Open(semantic_options->GetTestFilePath(formalism));
	        instance = static_cast<SemanticReader *> (semantic_reader_)->GetNext();
	        while (instance) {
		        semantic_dev_instances_.push_back(instance);
		        instance = static_cast<SemanticReader *> (semantic_reader_)->GetNext();
	        }
	        semantic_reader_->Close();

	        CHECK_EQ(semantic_instances_.size(), semantic_dep_instances_.size());
            LOG(INFO) << "Number of instances: " << semantic_instances_.size();
        }
        gettimeofday(&end, nullptr);
        LOG(INFO) << "Time: " << diff_ms(end, start);
    }

    void DeleteInstances(const string &formalism) {
        if (formalism == "dependency") {
            for (int i = 0; i < dependency_instances_.size(); ++i) {
                delete dependency_instances_[i];
            }
            dependency_instances_.clear();

	        for (int i = 0; i < dependency_dev_instances_.size(); ++i) {
		        delete dependency_dev_instances_[i];
	        }
	        dependency_dev_instances_.clear();
        } else if (formalism == "semantic") {
            for (int i = 0; i < semantic_instances_.size(); ++i) {
                delete semantic_instances_[i];
            }
            semantic_instances_.clear();
	        for (int i = 0; i < semantic_dep_instances_.size(); ++i) {
		        delete semantic_dep_instances_[i];
	        }
	        semantic_dep_instances_.clear();
	        for (int i = 0; i < semantic_dev_instances_.size(); ++i) {
		        delete semantic_dev_instances_[i];
	        }
	        semantic_dev_instances_.clear();
        } else {
            CHECK(false) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        }
    }

    Instance *GetFormattedInstance(const string &formalism, Instance *instance) {
        if (formalism == "dependency") {
	        auto instance_numeric = new DependencyInstanceNumeric;
            instance_numeric->Initialize(*GetDependencyDictionary(),
                                         static_cast<DependencyInstance *>(instance));
            return instance_numeric;
        } else if (formalism == "semantic") {
            auto instance_numeric = new SemanticInstanceNumeric;
            instance_numeric->Initialize(*GetSemanticDictionary(),
                                         static_cast<SemanticInstance *>(instance));
            return instance_numeric;
        } else {
            CHECK(false) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        }

    }

    void SaveModel(FILE *fs);

    void LoadModel(FILE *fs);

	void EnforceWellFormedGraph(Instance *instance,
	                            const vector<Part *> &arcs,
	                            vector<int> *inserted_heads,
	                            vector<int> *inserted_modifiers);

	void EnforceConnectedGraph(Instance *instance,
	                           const vector<Part *> &arcs,
	                           vector<int> *inserted_heads,
	                           vector<int> *inserted_modifiers);

	void EnforceProjectiveGraph(Instance *instance,
	                            const vector<Part *> &arcs,
	                            vector<int> *inserted_heads,
	                            vector<int> *inserted_modifiers);

    void MakeParts(const string &formalism, Instance *instance, Parts *parts,
                   vector<double> *gold_outputs);

    void DependencyMakePartsBasic(Instance *instance, Parts *parts,
                                  vector<double> *gold_outputs);

    void DependencyMakePartsBasic(Instance *instance, bool add_labeled_parts,
                                  Parts *parts,
                                  vector<double> *gold_outputs);

	void SemanticMakePartsBasic(Instance *instance, Parts *parts,
	                    vector<double> *gold_outputs);

	void SemanticMakePartsBasic(Instance *instance, bool add_labeled_parts, Parts *parts,
	                    vector<double> *gold_outputs);

    void DependencyLabelInstance(Parts *parts, const vector<double> &output,
                                 Instance *instance);

    void SemanticLabelInstance(Parts *parts, const vector<double> &output,
                               Instance *instance);

    void DependencyPrune(Instance *instance, Parts *parts,
                         vector<double> *gold_outputs,
                         bool preserve_gold);

	void SemanticPrune(Instance *instance, Parts *parts,
	                     vector<double> *gold_outputs,
	                     bool preserve_gold);


    virtual void BeginEvaluation() {

	    num_predicted_unlabeled_arcs_ = 0;
	    num_gold_unlabeled_arcs_ = 0;
	    num_matched_unlabeled_arcs_ = 0;
	    num_unlabeled_arcs_after_pruning_ = 0;
	    num_pruned_gold_unlabeled_arcs_ = 0;
	    num_possible_unlabeled_arcs_ = 0;
	    num_predicted_labeled_arcs_ = 0;
	    num_gold_labeled_arcs_ = 0;
	    num_matched_labeled_arcs_ = 0;
	    num_labeled_arcs_after_pruning_ = 0;
	    num_pruned_gold_labeled_arcs_ = 0;
	    num_possible_labeled_arcs_ = 0;

	    num_head_mistakes_ = 0;
	    num_label_mistakes_ = 0;
	    num_head_pruned_mistakes_ = 0;
	    num_heads_after_pruning_ = 0;
	    num_tokens_ = 0;

        gettimeofday(&start_clock_, nullptr);
    }

    void EvaluateInstance(const string &formalism, Instance *instance, Instance *output_instance,
                          Parts *parts, const vector<double> &gold_outputs,
                          const vector<double> &predicted_outputs) {
        if (formalism == "dependency") {
	        DependencyEvaluateInstance(instance, output_instance, parts, gold_outputs, predicted_outputs);
        } else {
            CHECK(false) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        }
    }


	void DependencyEvaluateInstance(Instance *instance,
	                              Instance *output_instance,
	                              Parts *parts,
	                              const vector<double> &gold_outputs,
	                              const vector<double> &predicted_outputs) {
		auto dependency_instance = static_cast<DependencyInstance *>(instance);
		auto dependency_parts = static_cast<DependencyParts *>(parts);
		int offset_labeled_arcs, num_labeled_arcs;
		dependency_parts->GetOffsetLabeledArc(&offset_labeled_arcs, &num_labeled_arcs);
		bool labeled = num_labeled_arcs > 0;
		const int slen =  dependency_instance->size() - 1;
		for (int m = 1; m < slen; ++m) {
			int head = -1;
			int num_possible_heads = 0;
			for (int h = 0; h < slen; ++h) {
				int r = dependency_parts->FindArc(h, m);
				if (r < 0) continue;
				++num_possible_heads;
				if (gold_outputs[r] >= 0.5) {
					CHECK_EQ(gold_outputs[r], 1.0);
					if (!NEARLY_EQ_TOL(gold_outputs[r], predicted_outputs[r], 1e-6)) {
						++num_head_mistakes_;
					}
					head = h;
					if (labeled) {
						const vector<int> &labeled_arcs =
								dependency_parts->FindLabeledArcs(h, m);
						for (int k = 0; k < labeled_arcs.size(); ++k) {
							int lab_r = labeled_arcs[k];
							if (lab_r < 0) continue;
							if (gold_outputs[lab_r] >= 0.5) {
								CHECK_EQ(gold_outputs[lab_r], 1.0);
								if (!NEARLY_EQ_TOL(gold_outputs[lab_r], predicted_outputs[lab_r], 1e-6)) {
									++num_label_mistakes_;
								}
							}
						}
					}
					//break;
				}
			}
			if (head < 0) {
				VLOG(2) << "Pruned gold part...";
				++num_head_mistakes_;
				++num_label_mistakes_;
				++num_head_pruned_mistakes_;
			}
			++num_tokens_;
			num_heads_after_pruning_ += num_possible_heads;
		}
	}

	void SemanticEvaluateInstance(Instance *instance, Instance *output_instance,
	                              Parts *parts, const vector<double> &gold_outputs,
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

    virtual void EndEvaluation(double &unlabeled_F1, double &labeled_F1) {
	    double unlabeled_precision =
			    static_cast<double>(num_matched_unlabeled_arcs_) /
			    static_cast<double>(num_predicted_unlabeled_arcs_);
	    double unlabeled_recall =
			    static_cast<double>(num_matched_unlabeled_arcs_) /
			    static_cast<double>(num_gold_unlabeled_arcs_);
	    unlabeled_F1 = 2.0 * unlabeled_precision * unlabeled_recall /
	                   (unlabeled_precision + unlabeled_recall);
	    double pruning_unlabeled_recall =
			    static_cast<double>(num_gold_unlabeled_arcs_ -
			                        num_pruned_gold_unlabeled_arcs_) /
			    static_cast<double>(num_gold_unlabeled_arcs_);
	    double pruning_unlabeled_efficiency =
			    static_cast<double>(num_possible_unlabeled_arcs_) /
			    static_cast<double>(num_tokens_);

	    double labeled_precision =
			    static_cast<double>(num_matched_labeled_arcs_) /
			    static_cast<double>(num_predicted_labeled_arcs_);
	    double labeled_recall =
			    static_cast<double>(num_matched_labeled_arcs_) /
			    static_cast<double>(num_gold_labeled_arcs_);
	    labeled_F1 = 2.0 * labeled_precision * labeled_recall /
	                 (labeled_precision + labeled_recall);
	    double pruning_labeled_recall =
			    static_cast<double>(num_gold_labeled_arcs_ -
			                        num_pruned_gold_labeled_arcs_) /
			    static_cast<double>(num_gold_labeled_arcs_);
	    double pruning_labeled_efficiency =
			    static_cast<double>(num_possible_labeled_arcs_) /
			    static_cast<double>(num_tokens_);

	    LOG(INFO) << "Unlabeled precision: " << unlabeled_precision
	              << " (" << num_matched_unlabeled_arcs_ << "/"
	              << num_predicted_unlabeled_arcs_ << ")" << " recall: " << unlabeled_recall
	              << " (" << num_matched_unlabeled_arcs_ << "/"
	              << num_gold_unlabeled_arcs_ << ")" << " F1: " << unlabeled_F1;
	    LOG(INFO) << "Pruning unlabeled recall: " << pruning_unlabeled_recall
	              << " ("
	              << num_gold_unlabeled_arcs_ - num_pruned_gold_unlabeled_arcs_
	              << "/"
	              << num_gold_unlabeled_arcs_ << ")";

	    LOG(INFO) << "Labeled precision: " << labeled_precision
	              << " (" << num_matched_labeled_arcs_ << "/"
	              << num_predicted_labeled_arcs_ << ")" << " recall: " << labeled_recall
	              << " (" << num_matched_labeled_arcs_ << "/"
	              << num_gold_labeled_arcs_ << ")" << " F1: " << labeled_F1;
	    LOG(INFO) << "Pruning labeled recall: " << pruning_labeled_recall
	              << " ("
	              << num_gold_labeled_arcs_ - num_pruned_gold_labeled_arcs_
	              << "/"
	              << num_gold_labeled_arcs_ << ")";

	    LOG(INFO) << "Unlabeled parsing accuracy: " <<
	              static_cast<double>(num_tokens_ - num_head_mistakes_) /
	              static_cast<double>(num_tokens_);
	    LOG(INFO) << "Pruning recall: " <<
	              static_cast<double>(num_tokens_ - num_head_pruned_mistakes_) /
	              static_cast<double>(num_tokens_);
	    if (num_matched_unlabeled_arcs_ == 0) {
		    labeled_F1 = static_cast<double>(num_tokens_ - num_head_mistakes_) /
				    static_cast<double>(num_tokens_);
	    }
    }

    /* Virtual function from Pipe.h but not implemented. */
    void ComputeScores(Instance *instance, Parts *parts, Features *features,
                       vector<double> *scores) {
        CHECK(false) << "Not implemented." << endl;
    }

    void RemoveUnsupportedFeatures(Instance *instance, Parts *parts,
                                   const vector<bool> &selected_parts,
                                   Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeFeatureDifference(Parts *parts,
                               Features *features,
                               const vector<double> &gold_output,
                               const vector<double> &predicted_output,
                               FeatureVector *difference) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeGradientStep(Parts *parts,
                          Features *features,
                          double eta,
                          int iteration,
                          const vector<double> &gold_output,
                          const vector<double> &predicted_output) {
        CHECK(false) << "Not implemented." << endl;
    }

    void TouchParameters(Parts *parts, Features *features,
                         const vector<bool> &selected_parts) {
        CHECK(false) << "Not implemented." << endl;
    }

    Features *CreateFeatures() { CHECK(false) << "Not implemented." << endl; }

    Parts *CreateParts() { CHECK(false) << "Not implemented." << endl; }

    void MakeSelectedFeatures(Instance *instance, Parts *parts,
                              const vector<bool> &selected_parts, Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeParts(Instance *instance, Parts *parts,
                   vector<double> *gold_outputs) { CHECK(false) << "Not implemented." << endl; }

    void LabelInstance(Parts *parts, const vector<double> &output,
                       Instance *instance) { CHECK(false) << "Not implemented." << endl; }

public:
	ParameterCollection *semantic_model_;
    ParameterCollection *dependency_model_;
    ParameterCollection *semantic_pruner_model_;
    ParameterCollection *dependency_pruner_model_;
	BiLSTM *parser_;
	SemanticParser *semantic_parser_;
    SemanticPruner *semantic_pruner_;
    DependencyPruner *dependency_pruner_;
	Trainer *semantic_trainer_;
    Trainer *dependency_trainer_;
    Trainer *semantic_pruner_trainer_;
    Trainer *dependency_pruner_trainer_;
protected:
    TokenDictionary *dependency_token_dictionary_;
	TokenDictionary *semantic_token_dictionary_;

    Reader *depdendency_reader_;
    Reader *semantic_reader_;

    Writer *dependency_writer_;
    Writer *semantic_writer_;

    Dictionary *dependency_dictionary_;
    Dictionary *semantic_dictionary_;

	Decoder *dependency_decoder_;
	Decoder *semantic_decoder_;

	int num_head_mistakes_;
	int num_label_mistakes_;
	int num_head_pruned_mistakes_;
	int num_heads_after_pruning_;
	int num_tokens_;
	
	int num_predicted_unlabeled_arcs_;
	int num_gold_unlabeled_arcs_;
	int num_matched_unlabeled_arcs_;
	int num_unlabeled_arcs_after_pruning_;
	int num_pruned_gold_unlabeled_arcs_;
	int num_possible_unlabeled_arcs_;
	int num_predicted_labeled_arcs_;
	int num_gold_labeled_arcs_;
	int num_matched_labeled_arcs_;
	int num_labeled_arcs_after_pruning_;
	int num_pruned_gold_labeled_arcs_;
	int num_possible_labeled_arcs_;

    vector<Instance *> dependency_instances_;
	vector<Instance *> dependency_dev_instances_;
    vector<Instance *> semantic_instances_;
	vector<Instance *> semantic_dep_instances_;
	vector<Instance *> semantic_dev_instances_;

    timeval start_clock_;
    unordered_map<int, vector<float>> *dependency_embedding_;
	unordered_map<int, vector<float>> *semantic_embedding_;
    unordered_map<int, int> *dependency_form_count_;
	unordered_map<int, int> *semantic_form_count_;

};

#endif /* SemanticPipe_H_ */
