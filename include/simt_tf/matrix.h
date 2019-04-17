#ifndef SIMT_TF_MATRIX_H
#define SIMT_TF_MATRIX_H

#include <simt_tf/vector.h>

#include <cassert>
#include <cstddef>
#include <iosfwd>

namespace simt_tf {

class Matrix {
public:
    Matrix() noexcept = default;

    SIMT_TF_HOST_DEVICE constexpr Matrix(float xx, float xy, float xz,
                                         float yx, float yy, float yz,
                                         float zx, float zy, float zz) noexcept
    : data_{{xx, xy, xz}, {yx, yy, yz}, {zx, zy, zz}} { }

    SIMT_TF_HOST_DEVICE constexpr Vector& operator[](std::size_t i) noexcept {
        assert(i < 3);

        return data_[i];
    }

    SIMT_TF_HOST_DEVICE constexpr const Vector& operator[](std::size_t i) const noexcept {
        assert(i < 3);

        return data_[i];
    }

    SIMT_TF_HOST_DEVICE constexpr Vector* begin() noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr const Vector* begin() const noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr Vector* cbegin() noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr Vector* end() noexcept {
        return data_ + 3;
    }

    SIMT_TF_HOST_DEVICE constexpr const Vector* end() const noexcept {
        return data_ + 3;
    }

    SIMT_TF_HOST_DEVICE constexpr Vector* cend() noexcept {
        return data_ + 3;
    }

    SIMT_TF_HOST_DEVICE constexpr Vector* data() noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr const Vector* data() const noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr float tdotx(const Vector &v) const noexcept {
        return (data_[0].x() * v.x()) + (data_[1].x() * v.y()) + (data_[2].x() * v.z());
    }

    SIMT_TF_HOST_DEVICE constexpr float tdoty(const Vector &v) const noexcept {
        return (data_[0].y() * v.x()) + (data_[1].y() * v.y()) + (data_[2].y() * v.z());
    }

    SIMT_TF_HOST_DEVICE constexpr float tdotz(const Vector &v) const noexcept {
        return (data_[0].z() * v.x()) + (data_[1].z() * v.y()) + (data_[2].z() * v.z());
    }

private:
    Vector data_[3];
};

std::ostream& operator<<(std::ostream &os, const Matrix &m);

SIMT_TF_HOST_DEVICE constexpr Vector operator*(const Matrix &lhs, const Vector &rhs) noexcept {
    return {dot(lhs[0], rhs), dot(lhs[1], rhs), dot(lhs[2], rhs)};
}

SIMT_TF_HOST_DEVICE constexpr Matrix operator*(const Matrix &lhs, const Matrix &rhs) noexcept {
    return {
        rhs.tdotx(lhs[0]), rhs.tdoty(lhs[0]), rhs.tdotz(lhs[0]),
        rhs.tdotx(lhs[1]), rhs.tdoty(lhs[1]), rhs.tdotz(lhs[1]),
        rhs.tdotx(lhs[2]), rhs.tdoty(lhs[2]), rhs.tdotz(lhs[2])
    };
}

} // namespace simt_tf

#endif
