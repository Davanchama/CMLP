cmake_minimum_required(VERSION 3.23)
project(cmlp)

set(CMAKE_CXX_STANDARD 14)

add_library(cmlp SHARED Layer.cpp Layer.h MLPWrapper.cpp MLPWrapper.h)
