#ifndef SIMT_TF_VECTOR_H
#define SIMT_TF_VECTOR_H

#include <simt_tf/macro.h>

#include <cassert>
#include <cstddef>
#include <iosfwd>

namespace simt_tf {

/**
 *  A 3-element vector for use in coordinate transforms.
 *
 *  Elements are stored contiguously, but Vectors are aligned on
 *  16-byte boundaries.
 */
class Vector {
public:
    /** @returns a Vector with its elements uninitialized. */
    Vector() noexcept = default;

    /** @returns a Vector {x, y, z}. */
    SIMT_TF_HOST_DEVICE constexpr Vector(float x, float y, float z) noexcept : data_{x, y, z} { }

    /**
     *  @param i must be in the range [0, 3).
     *  @returns a mutable reference to the i-th element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float& operator[](std::size_t idx) noexcept {
        assert(idx < 3);

        return data_[idx];
    }

    /**
     *  @param i must be in the range [0, 3).
     *  @returns an immutable reference to the i-th element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr const float& operator[](std::size_t idx) const noexcept {
        assert(idx < 3);

        return data_[idx];
    }

    /**
     *  @returns a mutable reference to the first element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float& x() noexcept {
        return data_[0];
    }

    /**
     *  @returns an immutable reference to the first element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr const float& x() const noexcept {
        return data_[0];
    }

    /**
     *  @returns a mutable reference to the second element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float& y() noexcept {
        return data_[1];
    }

    /**
     *  @returns an immutable reference to the second element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr const float& y() const noexcept {
        return data_[1];
    }

    /**
     *  @returns a mutable reference to the third element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr float& z() noexcept {
        return data_[2];
    }

    /**
     *  @returns an immutable reference to the third element of this
     *           Vector.
     */
    SIMT_TF_HOST_DEVICE constexpr const float& z() const noexcept {
        return data_[2];
    }

private:
    alignas(16) float data_[3];
};

/** @returns The dot product of two vectors. */
SIMT_TF_HOST_DEVICE constexpr float dot(const Vector &lhs, const Vector &rhs) noexcept {
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

/** @returns The sum of two vectors. */
SIMT_TF_HOST_DEVICE constexpr Vector operator+(const Vector &lhs, const Vector &rhs) noexcept {
    return {lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()};
}

/**
 *  Serializes a Vector to an ostream.
 *
 *  Elements are serialized on one line, in the format:
 *  {x, y, z}
 */
std::ostream& operator<<(std::ostream &os, const Vector &v);

} // namespace simt_tf

#endif
