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

#ifndef SIMT_TF_VECTOR3_H
#define SIMT_TF_VECTOR3_H

#include <simt_tf/macro.h>

#include <cassert>
#include <cstddef>
#include <iosfwd>

namespace simt_tf {

/** A 3-element vector for use in coordinate transforms.*/
class Vector3 {
public:
  /** @returns a Vector3 with its elements uninitialized. */
  Vector3() noexcept = default;

  /** @returns a Vector3 {x, y, z}. */
  SIMT_TF_HOST_DEVICE constexpr Vector3(float x, float y, float z) noexcept
      : data_{x, y, z} {}

  /**
   *  @param i must be in the range [0, 3).
   *  @returns a mutable reference to the i-th element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr float &operator[](std::size_t idx) noexcept {
    assert(idx < 3);

    return data_[idx];
  }

  /**
   *  @param i must be in the range [0, 3).
   *  @returns an immutable reference to the i-th element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr const float &operator[](std::size_t idx) const
      noexcept {
    assert(idx < 3);

    return data_[idx];
  }

  /**
   *  @returns a mutable reference to the first element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr float &x() noexcept { return data_[0]; }

  /**
   *  @returns an immutable reference to the first element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr const float &x() const noexcept {
    return data_[0];
  }

  /**
   *  @returns a mutable reference to the second element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr float &y() noexcept { return data_[1]; }

  /**
   *  @returns an immutable reference to the second element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr const float &y() const noexcept {
    return data_[1];
  }

  /**
   *  @returns a mutable reference to the third element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr float &z() noexcept { return data_[2]; }

  /**
   *  @returns an immutable reference to the third element of this
   *           Vector3.
   */
  SIMT_TF_HOST_DEVICE constexpr const float &z() const noexcept {
    return data_[2];
  }

private:
  alignas(16) float data_[3];
};

/** @returns The dot product of two vectors. */
SIMT_TF_HOST_DEVICE constexpr float dot(const Vector3 &lhs,
                                        const Vector3 &rhs) noexcept {
  return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

/** @returns The sum of two vectors. */
SIMT_TF_HOST_DEVICE constexpr Vector3 operator+(const Vector3 &lhs,
                                                const Vector3 &rhs) noexcept {
  return {lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()};
}

/** @returns true if the Vector3 contains any NaN values. */
SIMT_TF_HOST_DEVICE constexpr bool isnan(const Vector3 &v) noexcept {
  return v.x() != v.x() || v.y() != v.y() || v.z() != v.z();
}

} // namespace simt_tf

#endif
