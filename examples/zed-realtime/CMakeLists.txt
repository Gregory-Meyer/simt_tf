cmake_minimum_required(VERSION 3.12)

project(SimtTfZedRealtimeExample)

find_package(SimtTf 0.1 REQUIRED)

add_executable(transform transform.cpp)
set_target_properties(transform PROPERTIES CXX_STANDARD 14
                                           CXX_STANDARD_REQUIRED ON
                                           CXX_EXTENSIONS OFF)
target_link_libraries(transform SimtTf::SimtTf)
