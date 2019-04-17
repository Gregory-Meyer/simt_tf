cmake_minimum_required(VERSION 3.12)

project(SimtTf VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(ZED 2.7 REQUIRED)

add_library(simt_tf SHARED src/err.cu src/matrix.cu src/simt_tf.cu
                           src/transform.cu src/vector.cu)
add_library(SimtTf::simt_tf ALIAS simt_tf)
target_include_directories(simt_tf
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${ZED_INCLUDE_DIRS}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(simt_tf PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                         CUDA_STANDARD 14
                                         CUDA_STANDARD_REQUIRED ON
                                         CUDA_EXTENSIONS OFF)
target_link_libraries(simt_tf
    PUBLIC
        ${ZED_LIBRARIES} ${OpenCV_LIBS}
    PRIVATE
        ${CUDA_NPP_LIBRARIES_ZED} ${CUDA_nppc_LIBRARY}
)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/SimtTf)

install(TARGETS simt_tf
    EXPORT simt_tf-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(simt_tf PROPERTIES EXPORT_NAME SimtTf)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT simt_tf-targets
  FILE
    SimtTfTargets.cmake
  NAMESPACE
    SimtTf::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SimtTfConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/SimtTfConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SimtTfConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

#Install the config, configversion and custom find modules
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SimtTfConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SimtTfConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

export(EXPORT simt_tf-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/SimtTfTargets.cmake NAMESPACE SimtTf::)

#Register package in user's package registry
export(PACKAGE SimtTf)
