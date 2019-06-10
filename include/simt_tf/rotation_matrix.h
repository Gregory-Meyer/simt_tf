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

#ifndef SIMT_TF_ROTATION_MATRIX_H
#define SIMT_TF_ROTATION_MATRIX_H

#include <simt_tf/macro.h>
#include <simt_tf/vector3.h>

#include <cassert>
#include <iosfwd>
#include <type_traits>

namespace simt_tf {

/** A 3x3 orthogonal matrix for use as a basis matrix in transforms. */
class RotationMatrix {
public:
  /** @returns a RotationMatrix with uninitialized elements. */
  RotationMatrix() noexcept = default;

  /**
   *  The determinant of the provided elements must be either one or negative
   *  one.
   *
   *  @returns a RotationMatrix with its elements initialized in row-
   *           major order.
   */
  SIMT_TF_HOST_DEVICE constexpr RotationMatrix(float xx, float xy, float xz,
                                               float yx, float yy, float yz,
                                               float zx, float zy,
                                               float zz) noexcept
      : data_{{xx, xy, xz}, {yx, yy, yz}, {zx, zy, zz}} {
    assert(is_determinant_one());
  }

  /**
   * @returns a mutable reference to the I-th row of this RotationMatrix.
   * @tparam I the index of the row to get.
   */
  template <std::size_t I,
            std::enable_if_t<I<3, int> = 0>
                SIMT_TF_HOST_DEVICE constexpr Vector3 &get() noexcept {
    return data_[I];
  }
  /**
   * @returns an immutable reference to the I-th row of this RotationMatrix.
   * @tparam I the index of the row to get.
   */
  template <std::size_t I, std::enable_if_t<I<3, int> = 0>
                               SIMT_TF_HOST_DEVICE constexpr const Vector3
                                   &get() const noexcept {
    return data_[I];
  }

  /**
   *  @param i must be in the range [0, 3).
   *  @returns a mutable reference to the i-th row of this
   *           RotationMatrix.
   */
  SIMT_TF_HOST_DEVICE constexpr Vector3 &operator[](std::size_t i) noexcept {
    assert(i < 3);

    return data_[i];
  }

  /**
   *  @param i must be in the range [0, 3).
   *  @returns an immutable reference to the i-th row of this RotationMatrix.
   */
  SIMT_TF_HOST_DEVICE constexpr const Vector3 &operator[](std::size_t i) const
      noexcept {
    assert(i < 3);

    return data_[i];
  }

  /**
   *  @returns The dot product of this RotationMatrix's 0th column and a
   *           vector.
   */
  SIMT_TF_HOST_DEVICE constexpr float tdotx(const Vector3 &v) const noexcept {
    return (data_[0].x() * v.x()) + (data_[1].x() * v.y()) +
           (data_[2].x() * v.z());
  }

  /**
   *  @returns The dot product of this RotationMatrix's 1st column and a
   *           vector.
   */
  SIMT_TF_HOST_DEVICE constexpr float tdoty(const Vector3 &v) const noexcept {
    return (data_[0].y() * v.x()) + (data_[1].y() * v.y()) +
           (data_[2].y() * v.z());
  }

  /**
   *  @returns The dot product of this RotationMatrix's 2nd column and a
   *           vector.
   */
  SIMT_TF_HOST_DEVICE constexpr float tdotz(const Vector3 &v) const noexcept {
    return (data_[0].z() * v.x()) + (data_[1].z() * v.y()) +
           (data_[2].z() * v.z());
  }

  /**
   *  @returns The determinant of this matrix. Either 1 or -1.
   */
  SIMT_TF_HOST_DEVICE constexpr float det() const noexcept {
    // |a b c|
    // |d e f| = a(ei - fh) - b(di - fg) + c(dh - eg)
    // |g h i|
    return (data_[0].x() *
            (data_[1].y() * data_[2].z() - data_[1].z() * data_[2].y())) -
           (data_[0].y() *
            (data_[1].x() * data_[2].z() - data_[1].x() * data_[2].y())) +
           (data_[0].z() *
            (data_[1].x() * data_[2].y() - data_[1].x() * data_[2].y()));
  }

private:
  SIMT_TF_HOST_DEVICE constexpr bool is_determinant_one() const noexcept {
    float d = det();

    if (d < 0) {
      d = -d;
    }

    float diff = d - 1;

    if (diff < 0) {
      diff = -d;
    }

    return diff < 1e-5;
  }

  Vector3 data_[3];
};

/**
 *  @returns The result of matrix-vector multiplication between lhs
 *           and rhs.
 */
SIMT_TF_HOST_DEVICE constexpr Vector3 operator*(const RotationMatrix &lhs,
                                                const Vector3 &rhs) noexcept {
  return {dot(lhs.get<0>(), rhs), dot(lhs.get<1>(), rhs),
          dot(lhs.get<2>(), rhs)};
}

/**
 *  @returns The result of matrix-matrix multiplication between lhs
 *           and rhs.
 */
SIMT_TF_HOST_DEVICE constexpr RotationMatrix
operator*(const RotationMatrix &lhs, const RotationMatrix &rhs) noexcept {
  return {rhs.tdotx(lhs.get<0>()), rhs.tdoty(lhs.get<0>()),
          rhs.tdotz(lhs.get<0>()), rhs.tdotx(lhs.get<1>()),
          rhs.tdoty(lhs.get<1>()), rhs.tdotz(lhs.get<1>()),
          rhs.tdotx(lhs.get<2>()), rhs.tdoty(lhs.get<2>()),
          rhs.tdotz(lhs.get<2>())};
}

} // namespace simt_tf

#endif
