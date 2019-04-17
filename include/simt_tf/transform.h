#ifndef SIMT_TF_TRANSFORM_H
#define SIMT_TF_TRANSFORM_H

#include <simt_tf/matrix.h>
#include <simt_tf/vector.h>

namespace simt_tf {

class Transform {
public:
    Transform() noexcept = default;

    SIMT_TF_HOST_DEVICE constexpr Transform(const Matrix &basis, const Vector &origin) noexcept
    : basis_(basis), origin_(origin) { }

    SIMT_TF_HOST_DEVICE constexpr Vector operator()(const Vector &v) const noexcept {
        return basis_ * (v + origin_);
    }

    SIMT_TF_HOST_DEVICE constexpr Matrix& basis() noexcept {
        return basis_;
    }

    SIMT_TF_HOST_DEVICE constexpr const Matrix& basis() const noexcept {
        return basis_;
    }

    SIMT_TF_HOST_DEVICE constexpr Vector& origin() noexcept {
        return origin_;
    }

    SIMT_TF_HOST_DEVICE constexpr const Vector& origin() const noexcept {
        return origin_;
    }

private:
    Matrix basis_;
    Vector origin_;
};

SIMT_TF_HOST_DEVICE constexpr Transform operator*(const Transform &lhs,
                                                  const Transform &rhs) noexcept {
    return {lhs.basis() * rhs.basis(), lhs(rhs.origin())};
}

} // namespace simt_tf

#endif
