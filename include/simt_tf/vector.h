#ifndef SIMT_TF_VECTOR_H
#define SIMT_TF_VECTOR_H

#include <simt_tf/macro.h>

#include <cassert>
#include <cstddef>
#include <iosfwd>

namespace simt_tf {

class Vector {
public:
    Vector() noexcept = default;

    SIMT_TF_HOST_DEVICE constexpr Vector(float x, float y, float z) noexcept : data_{x, y, z} { }

    SIMT_TF_HOST_DEVICE constexpr float& operator[](std::size_t idx) noexcept {
        assert(idx < 3);

        return data_[idx];
    }

    SIMT_TF_HOST_DEVICE constexpr const float& operator[](std::size_t idx) const noexcept {
        assert(idx < 3);

        return data_[idx];
    }

    SIMT_TF_HOST_DEVICE constexpr float& x() noexcept {
        return data_[0];
    }

    SIMT_TF_HOST_DEVICE constexpr const float& x() const noexcept {
        return data_[0];
    }

    SIMT_TF_HOST_DEVICE constexpr float& y() noexcept {
        return data_[1];
    }

    SIMT_TF_HOST_DEVICE constexpr const float& y() const noexcept {
        return data_[1];
    }

    SIMT_TF_HOST_DEVICE constexpr float& z() noexcept {
        return data_[2];
    }

    SIMT_TF_HOST_DEVICE constexpr const float& z() const noexcept {
        return data_[2];
    }

    SIMT_TF_HOST_DEVICE constexpr std::size_t size() const noexcept {
        return 3;
    }

    SIMT_TF_HOST_DEVICE constexpr float* begin() noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr const float* begin() const noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr const float* cbegin() const noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr float* end() noexcept {
        return data_ + 3;
    }

    SIMT_TF_HOST_DEVICE constexpr const float* end() const noexcept {
        return data_ + 3;
    }

    SIMT_TF_HOST_DEVICE constexpr const float* cend() const noexcept {
        return data_ + 3;
    }

    SIMT_TF_HOST_DEVICE constexpr float* data() noexcept {
        return data_;
    }

    SIMT_TF_HOST_DEVICE constexpr const float* data() const noexcept {
        return data_;
    }

private:
    alignas(16) float data_[3];
};

SIMT_TF_HOST_DEVICE constexpr float dot(const Vector &lhs, const Vector &rhs) noexcept {
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

SIMT_TF_HOST_DEVICE constexpr Vector operator+(const Vector &lhs, const Vector &rhs) noexcept {
    return {lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()};
}

std::ostream& operator<<(std::ostream &os, const Vector &v);

} // namespace simt_tf

#endif
