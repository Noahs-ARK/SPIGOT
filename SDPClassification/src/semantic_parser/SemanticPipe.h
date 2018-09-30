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

#include <SstInstanceNumeric.h>
#include "Pipe.h"
#include "SemanticOptions.h"
#include "SemanticReader.h"
#include "SemanticDictionary.h"
#include "TokenDictionary.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticWriter.h"
#include "SemanticPart.h"
#include "SstReader.h"
#include "Classifier.h"
#include "SemanticPruner.h"
#include "SemanticParser.h"
#include "SstDictionary.h"

class SemanticPipe : public Pipe {
public:
    SemanticPipe(Options *semantic_options) : Pipe(semantic_options) {
	    token_dictionary_ = nullptr;
        dependency_dictionary_ = nullptr;
	    classification_token_dictionary_ = nullptr;
        classifier_trainer_ = nullptr;
        classifier_model_ = nullptr;
        classifier_ = nullptr;
	    parser_trainer_ = nullptr;
	    parser_model_ = nullptr;
	    parser_ = nullptr;
        pruner_trainer_ = nullptr;
        pruner_model_ = nullptr;
        pruner_ = nullptr;
    }

    virtual ~SemanticPipe() {
	    delete token_dictionary_;
        delete dependency_dictionary_;
	    delete classification_token_dictionary_;
        delete classifier_trainer_;
        delete classifier_model_;
	    delete classifier_;
	    delete parser_trainer_;
	    delete parser_model_;
	    delete parser_;
        delete pruner_trainer_;
        delete pruner_model_;
	    delete pruner_;
	    delete parser_form_count_;
	    delete classification_form_count_;
    }

    SemanticReader *GetSemanticReader() {
        return static_cast<SemanticReader *>(reader_);
    }

	DependencyReader *GetDependencyReader() {
		return static_cast<DependencyReader *>(reader_);
	}

    SstReader *GetSstReader(){
        return static_cast<SstReader *>(classification_reader_);
    }

    SemanticDictionary *GetSemanticDictionary() {
        return static_cast<SemanticDictionary *>(dictionary_);
    }

	SstDictionary *GetSstDictionary() {
		return classification_token_dictionary_;
	}

    SemanticDecoder *GetSemanticDecoder() {
        return static_cast<SemanticDecoder *>(decoder_);
    }

    SemanticOptions *GetSemanticOptions() {
        return static_cast<SemanticOptions *>(options_);
    }

    void Initialize();

    void DeleteInstances() {
        for (int i = 0; i < parser_instances_.size(); ++i) {
            delete parser_instances_[i];
        }
        parser_instances_.clear();

	    for (int i = 0; i < parser_dev_instances_.size(); ++i) {
		    delete parser_dev_instances_[i];
	    }
	    parser_dev_instances_.clear();

        for (int i = 0; i < classification_instances_.size(); ++i) {
            delete classification_instances_[i];
        }
        classification_instances_.clear();

	    for (int i = 0; i < classification_parser_instances_.size(); ++i) {
		    delete classification_parser_instances_[i];
	    }
	    classification_parser_instances_.clear();

	    for (int i = 0; i < classification_dev_instances_.size(); ++i) {
		    delete classification_dev_instances_[i];
	    }
	    classification_dev_instances_.clear();
    }

    void AddInstance(Instance *instance) {
        Instance *formatted_instance = GetFormattedInstance(instance);
        parser_instances_.push_back(formatted_instance);
        if (instance != formatted_instance) delete instance;
    }

    void CreateInstances() {
        timeval start, end;
        gettimeofday(&start, nullptr);

        LOG(INFO) << "Creating instances...";

        reader_->Open(options_->GetTrainingFilePath());
        DeleteInstances();
        Instance *instance = reader_->GetNext();
        while (instance) {
            AddInstance(instance);
            instance = reader_->GetNext();
        }
        reader_->Close();

	    reader_->Open(options_->GetTestFilePath());
	    instance = reader_->GetNext();
	    while (instance) {
		    parser_dev_instances_.push_back(instance);
		    instance = reader_->GetNext();
	    }
	    reader_->Close();

        classification_reader_->Open(GetSemanticOptions()->GetTrainingFilePath("classification"));
        instance = classification_reader_->GetNext();
        while (instance) {
	        Instance *formatted_instance = GetSstFormattedInstance(instance);
	        classification_instances_.push_back(formatted_instance);
	        formatted_instance = GetSstParserInstance(instance);
	        classification_parser_instances_.push_back(formatted_instance);
	        if (instance != formatted_instance) delete instance;
            instance = classification_reader_->GetNext();
        }
	    classification_reader_->Close();

	    classification_reader_->Open(GetSemanticOptions()->GetTestFilePath("classification"));
	    instance = classification_reader_->GetNext();
	    while (instance) {
		    classification_dev_instances_.push_back(instance);
		    instance = classification_reader_->GetNext();
	    }
	    classification_reader_->Close();

	    CHECK_EQ(classification_instances_.size(), classification_parser_instances_.size());
        LOG(INFO) << "Number of instances: " << parser_instances_.size();
        gettimeofday(&end, nullptr);
        LOG(INFO) << "Time: " << diff_ms(end, start);
    }

    void LoadPretrainedEmbedding(bool load_classifier_embedding,
                                 bool load_parser_embedding);

    void BuildFormCount();

    void Train();

    void TrainPruner();

    double TrainEpoch(const vector<int> &parser_idxs, const vector<int> &classifier_idxs, int epoch);


    double TrainPrunerEpoch(const vector<int> &idxs, int epoch);

    void Test();

    void Run(double &unlabeled_F1, double &labeled_F1, double &accuracy);

    void LoadNeuralModel();

    void SaveNeuralModel();

    void LoadPruner();

    void SavePruner();

protected:
    void CreateDictionary() {
        dictionary_ = new SemanticDictionary(this);
        GetSemanticDictionary()->SetTokenDictionary(token_dictionary_);
        GetSemanticDictionary()->SetDependencyDictionary(dependency_dictionary_);
    }

    void CreateReader() {
	    reader_ = new SemanticReader(options_);
        classification_reader_ = new SstReader(options_);
    }

    void CreateWriter() {
	    writer_ = new SemanticWriter(options_);
    }

    void CreateDecoder() {
	    decoder_ = new SemanticDecoder(this);
    }

    Parts *CreateParts() {
	    return new SemanticParts;
    }

    void CreateTokenDictionary() {
        token_dictionary_ = new TokenDictionary(this);
	    classification_token_dictionary_ = new SstDictionary(this);
    }

    void CreateDependencyDictionary() {
        dependency_dictionary_ = new DependencyDictionary(this);
    }

    void PreprocessData();

    Instance *GetFormattedInstance(Instance *instance) {
	    auto instance_numeric = new SemanticInstanceNumeric;
	    instance_numeric->Initialize(*GetSemanticDictionary(),
	                                 static_cast<SemanticInstance *>(instance));
	    return instance_numeric;
    }

    Instance *GetSstFormattedInstance(Instance *instance) {
        auto instance_numeric = new SstInstanceNumeric;
        instance_numeric->Initialize(*GetSstDictionary(),
                                     static_cast<SstInstance *>(instance));

	    auto ret = instance_numeric->Copy();
	    if (ret != instance_numeric)
		    delete instance_numeric;
	    return ret;
    }

	Instance *GetSstParserInstance(Instance *instance) {
		auto instance_numeric = new SstInstanceNumeric;
		instance_numeric->Initialize(*static_cast<SstDictionary *>(
				GetSemanticDictionary()->GetTokenDictionary()),
		                             static_cast<SstInstance *>(instance));
		auto ret = instance_numeric->Copy();
		if (ret != instance_numeric)
			delete instance_numeric;
		return ret;
	}

    void SaveModel(FILE *fs);

    void LoadModel(FILE *fs);

	void MakeParts(Instance *instance, Parts *parts,
	               vector<double> *gold_outputs);

	void SemanticMakePartsBasic(Instance *instance, Parts *parts,
	                            vector<double> *gold_outputs);

	void SemanticMakePartsBasic(Instance *instance, bool add_labeled_parts, Parts *parts,
	                            vector<double> *gold_outputs);

	void Prune(Instance *instance, Parts *parts,
	           vector<double> *gold_outputs,
	           bool preserve_gold);


	void LabelInstance(Parts *parts, const vector<double> &output,
	                   Instance *instance);
    
    virtual void BeginEvaluation() {
        num_predicted_unlabeled_arcs_ = 0;
        num_gold_unlabeled_arcs_ = 0;
        num_matched_unlabeled_arcs_ = 0;
        num_tokens_ = 0;
        num_unlabeled_arcs_after_pruning_ = 0;
        num_pruned_gold_unlabeled_arcs_ = 0;
        num_possible_unlabeled_arcs_ = 0;
        num_predicted_labeled_arcs_ = 0;
        num_gold_labeled_arcs_ = 0;
        num_matched_labeled_arcs_ = 0;
        num_labeled_arcs_after_pruning_ = 0;
        num_pruned_gold_labeled_arcs_ = 0;
        num_possible_labeled_arcs_ = 0;
	    num_tokens_ = 0;

	    gettimeofday(&start_clock_, nullptr);
    }

	void EvaluateInstance(Instance *instance,
	                      Instance *output_instance,
	                      Parts *parts,
	                      const vector<double> &gold_outputs,
	                      const vector<double> &predicted_outputs);

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

	    if (GetSemanticOptions()->labeled()) {
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

    void MakeSelectedFeatures(Instance *instance, Parts *parts,
                              const vector<bool> &selected_parts, Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }


public:
    ParameterCollection *classifier_model_;
    ParameterCollection *parser_model_;
    ParameterCollection *pruner_model_;
    Classifier *classifier_;
    BiLSTM *parser_;
	BiLSTM *pruner_;
    Trainer *classifier_trainer_;
	Trainer *parser_trainer_;
    Trainer *pruner_trainer_;

protected:
	vector<Instance *> parser_dev_instances_;
    vector<Instance *> classification_instances_;
	vector<Instance *> classification_parser_instances_;
    vector<Instance *> classification_dev_instances_;
    Reader *classification_reader_;
	
    TokenDictionary *token_dictionary_;
	DependencyDictionary *dependency_dictionary_;
	
	SstDictionary *classification_token_dictionary_;

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
    timeval start_clock_;
    unordered_map<int, vector<float>> *embedding_;
    unordered_map<int, int> *parser_form_count_;
	unordered_map<int, int> *classification_form_count_;
};

#endif /* SemanticPipe_H_ */
