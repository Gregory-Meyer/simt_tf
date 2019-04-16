cmake_minimum_required(VERSION 3.12)

project(transform LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(ZED 2.7 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

link_directories(${ZED_LIBRARY_DIR})

include_directories(include src ${ZED_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(transform src/err.cu src/main.cu src/matrix.cu
                         src/transform.cu src/vector.cu)
set_target_properties(transform PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(transform ${CUDA_NPP_LIBRARIES_ZED} ${CUDA_nppc_LIBRARY}
                                ${ZED_LIBRARIES} ${OpenCV_LIBS})
