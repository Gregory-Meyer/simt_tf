#ifndef SIMT_TF_MATRIX_H
#define SIMT_TF_MATRIX_H

#include <simt_tf/vector.h>

#include <cassert>
#include <cstddef>
#include <iosfwd>

namespace simt_tf {

/**
 *  A 3x3 matrix for use in coordinate transformations.
 *
 *  If this matrix will be used for transformations, its determinant
 *  must be 1.
 *
 *  Rows are stored contiguously, but each row is aligned to a 16-byte
 *  boundary.
 */
class Matrix {
public:
    /** @returns a Matrix with its elements uninitialized. */
    Matrix() noexcept = default;

    /**
     *  @returns a Matrix with its elements initialized in row-major
     *           order.
     */
    SIMT_TF_HOST_DEVICE constexpr Matrix(float xx, float xy, float xz,
                                         float yx, float yy, float yz,
                                         float zx, float zy, float zz) noexcept
    : data_{{xx, xy, xz}, {yx, yy, yz}, {zx, zy, zz}} { }

    /**
     *  @param i must be in the range [0, 3).
     *  @returns a mutable reference to the i-th row of this Matrix.
     */
    SIMT_TF_HOST_DEVICE constexpr Vector& operator[](std::size_t i) noexcept {
        assert(i < 3);

        return data_[i];
    }

    /**
     *  @param i must be in the range [0, 3).
     *  @returns an immutable reference to the i-th row of this Matrix.
     */
    SIMT_TF_HOST_DEVICE constexpr const Vector& operator[](std::size_t i) const noexcept {
        assert(i < 3);

        return data_[i];
    }

    /**
     *  @returns The dot product of this Matrix's 0th column and a
     *           vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float tdotx(const Vector &v) const noexcept {
        return (data_[0].x() * v.x()) + (data_[1].x() * v.y()) + (data_[2].x() * v.z());
    }

    /**
     *  @returns The dot product of this Matrix's 1st column and a
     *           vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float tdoty(const Vector &v) const noexcept {
        return (data_[0].y() * v.x()) + (data_[1].y() * v.y()) + (data_[2].y() * v.z());
    }

    /**
     *  @returns The dot product of this Matrix's 2nd column and a
     *           vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float tdotz(const Vector &v) const noexcept {
        return (data_[0].z() * v.x()) + (data_[1].z() * v.y()) + (data_[2].z() * v.z());
    }

private:
    Vector data_[3];
};

/**
 *  Serializes a matrix to a std::ostream.
 *
 *  Elements are output in row-major order on a single line, like:
 *  {{a00, a01, a02}, {a10, a11, a12}, {a20, a21, a22}}.
 */
std::ostream& operator<<(std::ostream &os, const Matrix &m);

/**
 *  @returns The result of matrix-vector multiplication between lhs
 *           and rhs.
 */
SIMT_TF_HOST_DEVICE constexpr Vector operator*(const Matrix &lhs, const Vector &rhs) noexcept {
    return {dot(lhs[0], rhs), dot(lhs[1], rhs), dot(lhs[2], rhs)};
}

/**
 *  @returns The result of matrix-matrix multiplication between lhs
 *           and rhs.
 */
SIMT_TF_HOST_DEVICE constexpr Matrix operator*(const Matrix &lhs, const Matrix &rhs) noexcept {
    return {
        rhs.tdotx(lhs[0]), rhs.tdoty(lhs[0]), rhs.tdotz(lhs[0]),
        rhs.tdotx(lhs[1]), rhs.tdoty(lhs[1]), rhs.tdotz(lhs[1]),
        rhs.tdotx(lhs[2]), rhs.tdoty(lhs[2]), rhs.tdotz(lhs[2])
    };
}

} // namespace simt_tf

#endif
