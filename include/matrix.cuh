#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "vector.cuh"

#include <cassert>
#include <cstddef>
#include <iosfwd>

class Matrix {
public:
    constexpr Matrix() noexcept = default;

    __host__ __device__ constexpr Matrix(float xx, float xy, float xz,
                                         float yx, float yy, float yz,
                                         float zx, float zy, float zz) noexcept
    : data_{{xx, xy, xz}, {yx, yy, yz}, {zx, zy, zz}} { }

    __host__ __device__ constexpr Vector& operator[](std::size_t i) noexcept {
        assert(i < 3);

        return data_[i];
    }

    __host__ __device__ constexpr const Vector& operator[](std::size_t i) const noexcept {
        assert(i < 3);

        return data_[i];
    }

    __host__ __device__ constexpr Vector* begin() noexcept {
        return data_;
    }

    __host__ __device__ constexpr const Vector* begin() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr Vector* cbegin() noexcept {
        return data_;
    }

    __host__ __device__ constexpr Vector* end() noexcept {
        return data_ + 3;
    }

    __host__ __device__ constexpr const Vector* end() const noexcept {
        return data_ + 3;
    }

    __host__ __device__ constexpr Vector* cend() noexcept {
        return data_ + 3;
    }

    __host__ __device__ constexpr Vector* data() noexcept {
        return data_;
    }

    __host__ __device__ constexpr const Vector* data() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr float tdotx(const Vector &v) const noexcept {
        return (data_[0].x() * v.x()) + (data_[1].x() * v.y()) + (data_[2].x() * v.z());
    }

    __host__ __device__ constexpr float tdoty(const Vector &v) const noexcept {
        return (data_[0].y() * v.x()) + (data_[1].y() * v.y()) + (data_[2].y() * v.z());
    }

    __host__ __device__ constexpr float tdotz(const Vector &v) const noexcept {
        return (data_[0].z() * v.x()) + (data_[1].z() * v.y()) + (data_[2].z() * v.z());
    }

private:
    Vector data_[3];
};

std::ostream& operator<<(std::ostream &os, const Matrix &m);

__host__ __device__ constexpr Vector operator*(const Matrix &lhs, const Vector &rhs) noexcept {
    return {dot(lhs[0], rhs), dot(lhs[1], rhs), dot(lhs[2], rhs)};
}

__host__ __device__ constexpr Matrix operator*(const Matrix &lhs, const Matrix &rhs) noexcept {
    return {
        rhs.tdotx(lhs[0]), rhs.tdoty(lhs[0]), rhs.tdotz(lhs[0]),
        rhs.tdotx(lhs[1]), rhs.tdoty(lhs[1]), rhs.tdotz(lhs[1]),
        rhs.tdotx(lhs[2]), rhs.tdoty(lhs[2]), rhs.tdotz(lhs[2])
    };
}

#endif
