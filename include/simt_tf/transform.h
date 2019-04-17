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

#ifndef SIMT_TF_TRANSFORM_H
#define SIMT_TF_TRANSFORM_H

#include <simt_tf/matrix.h>
#include <simt_tf/vector.h>

namespace simt_tf {

/**
 *  A coordinate transform.
 *
 *  Its basis matrix must have a determinant of 1.
 */
class Transform {
public:
    /** @returns a Transform with its elements uninitialized. */
    Transform() noexcept = default;

    /**
     *  @param basis must have a determinant of 1.
     *  @returns a Transform with the specified basis rotation matrix
     *           and origin.
     */
    SIMT_TF_HOST_DEVICE constexpr Transform(const Matrix &basis, const Vector &origin) noexcept
    : basis_(basis), origin_(origin) { }

    /**
     *  @returns v after translation and rotation into a new coordinate
     *           frame.
     */
    SIMT_TF_HOST_DEVICE constexpr Vector operator()(const Vector &v) const noexcept {
        return basis_ * (v + origin_);
    }

    /**
     *  If the basis matrix is modified, it must retain a determinant
     *  of 1.
     *
     *  @returns a mutable reference to the basis rotation matrix of
     *           this Transform.
     */
    SIMT_TF_HOST_DEVICE constexpr Matrix& basis() noexcept {
        return basis_;
    }

    /**
     *  @returns an immutable reference to the basis rotation matrix of
     *           this Transform.
     */
    SIMT_TF_HOST_DEVICE constexpr const Matrix& basis() const noexcept {
        return basis_;
    }

    /**
     *  @returns a mutable reference to the origin of this Transform.
     */
    SIMT_TF_HOST_DEVICE constexpr Vector& origin() noexcept {
        return origin_;
    }

    /**
     *  @returns an immutable reference to the origin of this
     *           Transform.
     */
    SIMT_TF_HOST_DEVICE constexpr const Vector& origin() const noexcept {
        return origin_;
    }

private:
    Matrix basis_;
    Vector origin_;
};

/**
 *  @returns the product of two Transforms, such that
 *           t(v) = rhs(lhs(v)).
 */
SIMT_TF_HOST_DEVICE constexpr Transform operator*(const Transform &lhs,
                                                  const Transform &rhs) noexcept {
    return {lhs.basis() * rhs.basis(), lhs(rhs.origin())};
}

} // namespace simt_tf

#endif
