cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
PROJECT(ad3)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/include/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ad3)

ADD_LIBRARY(ad3 ad3/FactorGraph.cpp ad3/GenericFactor.cpp ad3/Factor.cpp ad3/Utils.cpp)