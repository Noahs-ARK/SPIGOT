CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(neurboparser)

ADD_LIBRARY(semantic_parser
        SemanticDecoder.cpp SemanticPipe.cpp
        SemanticOptions.cpp SemanticDictionary.cpp
        SemanticInstance.cpp SemanticReader.cpp
        SemanticPart.cpp SemanticInstanceNumeric.cpp SemanticWriter.cpp
        FactorSemanticGraph.h SemanticPredicate.h
        nodes-argmax-proj-sdp.cpp
        nodes-argmax-proj-singlehead.cpp nodes-argmax-ste.cpp nodes-argmax-proj.cpp nodes-argmax-proj01.cpp expr.cpp
        ProjectSimplex.cpp SemanticPruner.cpp Classifier.cpp BiLSTM.cpp SemanticParser.cpp)

target_link_libraries(semantic_parser parser dynet pthread gflags ad3 glog
        classifier util sequence)