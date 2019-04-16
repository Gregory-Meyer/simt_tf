#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

#include "matrix.cuh"
#include "vector.cuh"

class Transform {
public:
    Transform() noexcept = default;

    __host__ __device__ constexpr Transform(const Matrix &basis, const Vector &origin) noexcept
    : basis_(basis), origin_(origin) { }

    __host__ __device__ constexpr Vector operator()(const Vector &v) const noexcept {
        return basis_ * (v + origin_);
    }

    __host__ __device__ constexpr Matrix& basis() noexcept {
        return basis_;
    }

    __host__ __device__ constexpr const Matrix& basis() const noexcept {
        return basis_;
    }

    __host__ __device__ constexpr Vector& origin() noexcept {
        return origin_;
    }

    __host__ __device__ constexpr const Vector& origin() const noexcept {
        return origin_;
    }

private:
    Matrix basis_;
    Vector origin_;
};

__host__ __device__ constexpr Transform operator*(const Transform &lhs,
                                                  const Transform &rhs) noexcept {
    return {lhs.basis() * rhs.basis(), lhs(rhs.origin())};
}

#endif
