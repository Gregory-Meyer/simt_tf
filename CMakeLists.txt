cmake_minimum_required(VERSION 3.12)

project(transform LANGUAGES CXX CUDA)

include_directories(include src)

add_executable(transform src/err.cu src/main.cu src/matrix.cu
                         src/transform.cu src/vector.cu)
target_compile_features(transform PUBLIC cxx_std_14)
set_target_properties(transform PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
