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

#include <cassert>
#include <cmath>

namespace simt_tf {
namespace {

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
__global__ void transform(Transform tf, MatrixSpan<const Vector4> pointcloud,
                          MatrixSpan<const std::uint8_t> pixels,
                          MatrixSpan<std::uint8_t> transformed,
                          float resolution, float x_offset, float y_offset) {
  const std::uint32_t pointcloud_row = (blockIdx.y * blockDim.y) + threadIdx.y;
  const std::uint32_t pointcloud_col = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (pointcloud_row >= pointcloud.num_rows() ||
      pointcloud_col >= pointcloud.num_cols()) {
    return;
  }

  const Vector3 this_point =
      pointcloud[pointcloud_row][pointcloud_col].as_vec3();

  if (isnan(this_point)) {
    return;
  }

  const Vector3 point_transformed =
      tf({this_point.x(), this_point.y(), this_point.z()});

  const float pixel_x = (point_transformed.x() + x_offset) / resolution;
  const float pixel_y = (point_transformed.y() + y_offset) / resolution;

  if (pixel_x < 0 || pixel_y < 0) {
    return;
  }

  const auto output_col = static_cast<std::uint32_t>(pixel_x);
  const auto output_row = static_cast<std::uint32_t>(pixel_y);

  if (output_row >= transformed.num_rows() ||
      output_col >= transformed.num_cols()) {
    return;
  }

  transformed[output_row][output_col] = pixels[pointcloud_row][pointcloud_col];
}

} // namespace

/**
 *  Transforms each point in a pointcloud to another coordinate frame
 *  and projects them onto the x-y plane.
 *
 *  @param tf The coordinate transform to apply.
 *  @param pointcloud The pointcloud to read data from. Its memory
 *                    should be accessible from the GPU in the current
 *                    CUDA context.
 *  @param pixels The pixels to transform and project. Its memory
 *                should be accessible from the GPU in the current CUDA
 *                context.
 *  @param transformed The matrix to transform and project the pixels
 *                     onto. Its memory should be accessible from the
 *                     GPU in the current CUDA context.
 *  @param resolution The resolution of each cell in the output matrix.
 *                    In other words, each cell represents a
 *                    (resolution x resolution) square in whatever
 *                    units that input pointcloud is in.
 */
void transform_project(const Transform &tf,
                       MatrixSpan<const Vector4> pointcloud,
                       MatrixSpan<const std::uint8_t> pixels,
                       MatrixSpan<std::uint8_t> transformed, float resolution) {
  assert(pointcloud.num_rows() == pixels.num_rows());
  assert(pointcloud.num_cols() == pixels.num_cols());

  const float x_range = static_cast<float>(transformed.num_cols()) * resolution;
  const float y_range = static_cast<float>(transformed.num_rows()) * resolution;

  constexpr std::size_t BLOCKSIZE = 16;
  const std::size_t num_blocks_x = div_to_inf(pointcloud.num_rows(), BLOCKSIZE);
  const std::size_t num_blocks_y = div_to_inf(pointcloud.num_cols(), BLOCKSIZE);

  transform<<<dim3(num_blocks_x, num_blocks_y), dim3(BLOCKSIZE, BLOCKSIZE)>>>(
      tf, pointcloud, pixels, transformed, resolution, x_range / 2,
      y_range / 2);
}

} // namespace simt_tf
