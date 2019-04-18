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

#include <simt_tf/simt_tf.h>

#include "err.cuh"

#include <cassert>
#include <cmath>
#include <atomic>
#include <mutex>

namespace {

__host__ __device__ bool isnan(const sl::float4 &v) noexcept {
    return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]);
}

__host__ __device__ std::uint32_t as_u32(float x) noexcept {
    union Converter {
        float f;
        std::uint32_t i;
    };

    return Converter{x}.i;
}

constexpr std::size_t div_to_inf(std::size_t x, std::size_t y) noexcept {
    const std::size_t res = x / y;

    if (x % y != 0) {
        return res + 1;
    }

    return res;
}

/**
 *  @param input Points to an array of length n. Each element must be
 *               a 4-tuple (X, Y, Z, RGBA) corresponding to a depth
 *               map. The RGBA component is packed into a 32-bit float,
 *               with each element being an 8-bit unsigned integer.
 *  @param output Points to a matrix with m rows and p columns. Each
 *                element must be a 4-tuple (B, G, R, A). The elements
 *                should be stored contiguously in row-major order.
 *  @param resolution The resolution (in meters) of each pixel in
 *                    output, such that each pixel represents a
 *                    (resolution x resolution) square.
 *  @param x_offset The offset (in meters) between the center of the
 *                  matrix and its leftmost edge, such that the pixel
 *                  at (0, 0) is located in free space at (-x_offset,
 *                  -y_offset).
 *  @param y_offset The offset (in meters) between the center of the
 *                  matrix and its topmost edge, such that the pixel
 *                  at (0, 0) is located in free space at (-x_offset,
 *                  -y_offset).
 */
__global__ void transform(
    simt_tf::Transform tf, const sl::float4 *input,
    std::uint32_t n, std::uint32_t *output, std::uint32_t output_rows,
    std::uint32_t output_cols, std::uint32_t output_stride, float resolution,
    float x_offset,
    float y_offset
) {
    const std::uint32_t pixel_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pixel_idx >= n) {
        return;
    }

    const sl::float4 elem = input[pixel_idx];

    if (isnan(elem)) {
        return;
    }

    const simt_tf::Vector transformed = tf({elem[0], elem[1], elem[2]});

    const float pixel_x = (transformed.x() + x_offset) / resolution;
    const float pixel_y = (transformed.y() + y_offset) / resolution;

    if (pixel_x < 0 || pixel_y < 0) {
        return;
    }

    const auto col = static_cast<std::uint32_t>(pixel_x);
    const auto row = static_cast<std::uint32_t>(pixel_y);

    if (col >= output_cols || row >= output_rows) {
        return;
    }

    const std::uint32_t output_idx = row * output_stride + col;
    output[output_idx] = as_u32(elem[3]);
}

} // namespace

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
 *  @param pointcloud The pointcloud to place the pointcloud retrieved
 *                    from the ZED.
 *  @param output The matrix to place the bird's eye transformed image
 *                in.
 *  @param resolution The resolution, in whatever unit the ZED is
 *                    configured to output in, of each cell of the
 *                    output matrix. In other words, each cell
 *                    represents a (resolution x resolution) square in
 *                    free space.
 *
 *  @throws std::system_error if any ZED SDK or CUDA function call
 *          fails.
 */
void pointcloud_birdseye(
    const Transform &tf, sl::Camera &camera, sl::Mat &pointcloud,
    cv::cuda::GpuMat &output, float resolution
) {
    assert(camera.isOpened());

    cuCtxSetCurrent(camera.getCUDAContext());

    const auto output_numel = static_cast<std::uint32_t>(output.size().area());
    constexpr std::uint32_t BLOCKSIZE = 256;

    cudaMemset(output.ptr<std::uint32_t>(), 0, output_numel * sizeof(std::uint32_t));

    const std::error_code pc_ret =
        camera.retrieveMeasure(pointcloud, sl::MEASURE_XYZBGRA, sl::MEM_GPU);

    if (pc_ret) {
        throw std::system_error(pc_ret);
    }

    const auto output_cols = static_cast<std::uint32_t>(output.size().width);
    const auto output_rows = static_cast<std::uint32_t>(output.size().height);
    const auto output_stride = static_cast<std::uint32_t>(output.step / output.elemSize());
    const float x_range = static_cast<float>(output_cols) * resolution;
    const float y_range = static_cast<float>(output_rows) * resolution;

    const auto numel = static_cast<std::uint32_t>(pointcloud.getResolution().area());
    const std::uint32_t num_blocks = div_to_inf(numel, BLOCKSIZE);

    transform<<<num_blocks, BLOCKSIZE>>>(
        tf,
        pointcloud.getPtr<sl::float4>(sl::MEM_GPU),
        numel,
        output.ptr<std::uint32_t>(),
        output_rows,
        output_cols,
        output_stride,
        resolution,
        x_range / 2,
        y_range / 2
    );
}

} // namespace simt_tf
