cmake_minimum_required(VERSION 3.12)

project(transform)

add_library(transform STATIC src/err.cu err/transform.cu err/vector.cu)
target_compile_features(transform PUBLIC cxx_std_14)
