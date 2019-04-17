#ifndef SIMT_TF_SIMT_TF_H
#define SIMT_TF_SIMT_TF_H

#include <simt_tf/transform.h>

#include <cstddef>
#include <cstdint>

#include <opencv2/core/cuda.hpp>

#include <sl/Core.hpp>
#include <sl/Camera.hpp>

namespace simt_tf {

cv::cuda::GpuMat pointcloud_birdseye(
    const Transform &tf, sl::Camera &camera, std::size_t output_cols,
    std::size_t output_rows, float output_resolution
);

} // namespace simt_tf

#endif
