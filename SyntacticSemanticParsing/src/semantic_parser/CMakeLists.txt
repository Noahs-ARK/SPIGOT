CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(semantic_parser)

ADD_LIBRARY(semantic_parser
        SemanticPipe.cpp
        SemanticOptions.cpp SemanticDictionary.cpp
        SemanticDecoder.cpp SemanticInstance.cpp SemanticInstanceNumeric.cpp
        SemanticPart.cpp SemanticPredicate.h SemanticReader.cpp SemanticWriter.cpp
        FactorSemanticGraph.h ProjectSimplex.cpp
        BiLSTM.cpp Dependency.cpp DependencyPruner.cpp StructuredAttention.cpp
        SemanticPruner.cpp SemanticParser.cpp
        expr.cpp nodes-argmax-ste.cpp nodes-argmax-proj.cpp nodes-argmax-proj01.cpp
        nodes-argmax-proj-singlehead.cpp
        )

target_link_libraries(semantic_parser dynet pthread gflags glog
        classifier util sequence parser)