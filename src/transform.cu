#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <system_error>
#include <type_traits>
#include <vector>

#include "err.h"

struct CudaDeleter {
    template <typename T>
    void operator()(T *ptr) noexcept {
        static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible");

        cudaFree(ptr);
    }
};

template <typename T, std::enable_if_t<!std::is_array<T>::value, int> = 0>
std::unique_ptr<T, CudaDeleter> to_device(const T &host) {
    static_assert(std::is_trivially_copyable<T>::value, "");

    T *device_ptr;

    const std::error_code malloc_ec = cudaMalloc(&device_ptr, sizeof(T));

    if (malloc_ec) {
        throw std::bad_alloc();
    }

    const std::error_code memcpy_ec = cudaMemcpy(device_ptr, std::addressof(host), sizeof(T), cudaMemcpyHostToDevice);

    if (memcpy_ec) {
        throw std::system_error(memcpy_ec);
    }

    return {device_ptr, CudaDeleter()};
}

template <typename T, typename ...Ts>
constexpr T&& front(T &&t, Ts &&...ts) noexcept {
    return std::forward<T>(t);
}

template <typename T>
class CudaAllocator {
public:
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible");

    using value_type = T;

    T* allocate(std::size_t n) {
        T *device_ptr;

        const std::error_code ec = cudaMalloc(&device_ptr, n * sizeof(T));

        if (ec) {
            throw std::bad_alloc();
        }

        return device_ptr;
    }

    void deallocate(T *device_ptr, std::size_t) noexcept {
        cudaFree(device_ptr);
    }

    template <typename ...Ts>
    void construct(T *device_ptr, Ts &&...ts) {
        const T host(std::forward<Ts>(ts)...);

        const std::error_code ec = cudaMemcpy(device_ptr, std::addressof(host), sizeof(T), cudaMemcpyHostToDevice);

        if (ec) {
            throw std::system_error(ec);
        }
    }

    void construct(T *device_ptr, const T &host) {
        const std::error_code ec = cudaMemcpy(device_ptr, std::addressof(host), sizeof(T), cudaMemcpyHostToDevice);

        if (ec) {
            throw std::system_error(ec);
        }
    }

    void destroy(T*) noexcept {}
};

template <typename T, typename A = std::allocator<T>>
std::vector<T, A> to_host(const std::vector<T, CudaAllocator<T>> &device) {
    std::vector<T, A> host(device.size());

    const std::error_code ec = cudaMemcpy(host.data(), device.data(), device.size() * sizeof(T), cudaMemcpyDeviceToHost);

    if (ec) {
        throw std::system_error(ec);
    }

    return host;
}

class Vector {
public:
    __host__ __device__ Vector() noexcept : data_{0, 0, 0} { }

    __host__ __device__ Vector(float x, float y, float z) noexcept : data_{x, y, z} { }

    __host__ __device__ float& operator[](std::size_t idx) {
        return data_[idx];
    }

    __host__ __device__ const float& operator[](std::size_t idx) const {
        return data_[idx];
    }

    __host__ __device__ float& x()  {
        return data_[0];
    }

    __host__ __device__ const float& x() const {
        return data_[0];
    }

    __host__ __device__ float& y() {
        return data_[1];
    }

    __host__ __device__ const float& y() const {
        return data_[1];
    }

    __host__ __device__ float& z() {
        return data_[2];
    }

    __host__ __device__ const float& z() const {
        return data_[2];
    }

private:
    alignas(16) float data_[3];
};

std::ostream& operator<<(std::ostream &os, const Vector &v) {
    return os << '{' << v.x() << ", " << v.y() << ", " << v.z() << '}';
}

__host__ __device__ float dot(const Vector &lhs, const Vector &rhs) {
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

__host__ __device__ Vector operator+(const Vector &lhs, const Vector &rhs) {
    return {lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()};
}

struct Column {
    const float *data;

    __host__ __device__ const float& x() const {
        return data[0];
    }

    __host__ __device__ const float& y() const {
        return data[4];
    }

    __host__ __device__ const float& z() const {
        return data[8];
    }
};

__host__ __device__ float dot(const Column &lhs, const Column &rhs) {
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

__host__ __device__ float dot(const Vector &lhs, const Column &rhs) {
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

__host__ __device__ float dot(const Column &lhs, const Vector &rhs) {
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

struct Matrix {
    Vector rows[3];

    __host__ __device__ const Vector& operator[](std::size_t i) const {
        return rows[i];
    }

    __host__ __device__ Column col(std::size_t i) const {
        return Column{&rows[0][i]};
    }
};

std::ostream& operator<<(std::ostream &os, const Matrix &m) {
    return os << '{' << m[0] << ", " << m[1] << ", " << m[2] << '}';
}

__host__ __device__ Vector operator*(const Matrix &lhs, const Vector &rhs) {
    return {dot(lhs[0], rhs), dot(lhs[1], rhs), dot(lhs[2], rhs)};
}

__host__ __device__ Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
    return {{
        {dot(lhs[0], rhs.col(0)), dot(lhs[0], rhs.col(1)), dot(lhs[0], rhs.col(2))},
        {dot(lhs[1], rhs.col(0)), dot(lhs[1], rhs.col(1)), dot(lhs[1], rhs.col(2))},
        {dot(lhs[2], rhs.col(0)), dot(lhs[2], rhs.col(1)), dot(lhs[2], rhs.col(2))}
    }};
}

struct Transform {
    Matrix basis;
    Vector origin;

    __host__ __device__ Vector operator()(const Vector &v) const {
        return basis * (v + origin);
    }
};

__host__ __device__ Transform operator*(const Transform &lhs, const Transform &rhs) {
    return {lhs.basis * rhs.basis, lhs(rhs.origin)};
}

__global__ void transform(const Transform &tf, const Vector *input, Vector *output) {
    output[threadIdx.x] = tf(input[threadIdx.x]);
}

int main(int argc, char* argv[]) {
    const Transform host_tf = {
        {{
            {1, 0, 0},
            {0, 0, -1},
            {0, 1, 0}
        }},
        {1, 0, 0}
    };

    const auto device_tf_ptr = to_device(host_tf);

    const Vector host_inputs[8] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}
    };

    const std::vector<Vector, CudaAllocator<Vector>> device_inputs(std::cbegin(host_inputs), std::cend(host_inputs));
    std::vector<Vector, CudaAllocator<Vector>> device_outputs(8);

    // Launch the kernel.
    transform<<<1, 8>>>(*device_tf_ptr, device_inputs.data(), device_outputs.data());

    // Copy output data to host.
    cudaDeviceSynchronize();

    const auto host_outputs = to_host(device_outputs);

    for (const Vector &v : host_outputs) {
        std::cout << v << '\n';
    }
}
