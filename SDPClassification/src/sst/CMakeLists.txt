CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(sst)

ADD_LIBRARY(sst
        SstInstance.cpp SstReader.cpp
        SstInstanceNumeric.cpp
        SstDictionary.cpp SstDictionary.h)

target_link_libraries(sst dynet pthread gflags ad3 glog
        classifier util sequence)
