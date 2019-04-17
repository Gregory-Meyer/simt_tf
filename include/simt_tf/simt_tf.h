// Copyright (c) 2019 Gregory Meyer
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice (including
// the next paragraph) shall be included in all copies or substantial
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef SIMT_TF_SIMT_TF_H
#define SIMT_TF_SIMT_TF_H

#include <simt_tf/transform.h>

#include <cstddef>
#include <cstdint>

#include <opencv2/core/cuda.hpp>

#include <sl/Core.hpp>
#include <sl/Camera.hpp>

namespace simt_tf {

/**
 *  Fetches the left-side pointcloud from a ZED camera, then performs a
 *  GPU-accelerated transform (translate then rotate) on each point and
 *  projects it into a bird's eye view.
 *
 *  @param tf Must be a valid coordinate transform, meaning that the
 *            determinant of its basis rotation matrix must be 1.
 *  @param camera Must be opened, have the latest data grabbed, and
 *                have support for depth images and pointclouds.
 *                It is assumed that x is forward, y, is left, and
 *                z is up, in accordance with ROS REP 103.
 *  @param output_cols The number of columns in the output matrix.
 *  @param output_rows The number of rows in the output matrix.
 *  @param output_resolution The resolution, in whatever unit the ZED
 *                           is configured to output in, of each cell
 *                           of the output matrix. In other words, each
 *                           cell represents a (resolution x
 *                           resolution) square in free space.
 *  @returns A transformed and bird's eye'd view of the camera's left
 *           side point cloud.
 *
 *  @throws std::system_error if any ZED SDK or CUDA function call
 *          fails.
 */
cv::cuda::GpuMat pointcloud_birdseye(
    const Transform &tf, sl::Camera &camera, std::size_t output_cols,
    std::size_t output_rows, float output_resolution
);

} // namespace simt_tf

#endif
