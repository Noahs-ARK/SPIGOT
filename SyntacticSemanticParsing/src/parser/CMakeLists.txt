CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(parser)


ADD_LIBRARY(parser DependencyWriter.cpp DependencyDictionary.cpp
        DependencyDictionary.cpp DependencyInstance.cpp
        DependencyReader.cpp DependencyInstanceNumeric.cpp
        DependencyWriter.cpp DependencyPart.cpp
        DependencyReader.cpp FactorTree.h DependencyDecoder.cpp)

target_link_libraries(parser dynet pthread gflags ad3 glog)