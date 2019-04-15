#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cassert>
#include <cstddef>
#include <iosfwd>

class Vector {
public:
    __host__ __device__ constexpr Vector() noexcept = default;

    __host__ __device__ constexpr Vector(float x, float y, float z) noexcept : data_{x, y, z} { }

    __host__ __device__ constexpr float& operator[](std::size_t idx) noexcept {
        assert(idx < 3);

        return data_[idx];
    }

    __host__ __device__ constexpr const float& operator[](std::size_t idx) const noexcept {
        assert(idx < 3);

        return data_[idx];
    }

    __host__ __device__ constexpr float& x() noexcept {
        return data_[0];
    }

    __host__ __device__ constexpr const float& x() const noexcept {
        return data_[0];
    }

    __host__ __device__ constexpr float& y() noexcept {
        return data_[1];
    }

    __host__ __device__ constexpr const float& y() const noexcept {
        return data_[1];
    }

    __host__ __device__ constexpr float& z() noexcept {
        return data_[2];
    }

    __host__ __device__ constexpr const float& z() const noexcept {
        return data_[2];
    }

    __host__ __device__ constexpr std::size_t size() const noexcept {
        return 3;
    }

    __host__ __device__ constexpr float* begin() noexcept {
        return data_;
    }

    __host__ __device__ constexpr const float* begin() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr const float* cbegin() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr float* begin() noexcept {
        return data_;
    }

    __host__ __device__ constexpr const float* begin() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr const float* cbegin() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr float* end() noexcept {
        return data_ + 3;
    }

    __host__ __device__ constexpr const float* end() const noexcept {
        return data_ + 3;
    }

    __host__ __device__ constexpr const float* cend() const noexcept {
        return data_ + 3;
    }

private:
    alignas(16) float data_[3];
};

std::ostream& operator<<(std::ostream &os, const Vector &v);

#endif
