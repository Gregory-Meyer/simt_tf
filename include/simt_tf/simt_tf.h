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

#include <simt_tf/matrix_span.h>
#include <simt_tf/transform.h>
#include <simt_tf/vector4.h>

#include <cstddef>
#include <cstdint>

namespace simt_tf {

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
                       MatrixSpan<std::uint8_t> transformed, float resolution);

} // namespace simt_tf

#endif
