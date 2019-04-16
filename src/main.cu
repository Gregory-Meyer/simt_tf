#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <system_error>
#include <type_traits>
#include <vector>

#include "err.cuh"
#include "matrix.cuh"
#include "transform.cuh"
#include "vector.cuh"

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

        const std::error_code ec = cudaMemcpy(
            device_ptr,
            std::addressof(host),
            sizeof(T),
            cudaMemcpyHostToDevice
        );

        if (ec) {
            throw std::system_error(ec);
        }
    }

    void construct(T *device_ptr, const T &host) {
        const std::error_code ec = cudaMemcpy(
            device_ptr,
            std::addressof(host),
            sizeof(T),
            cudaMemcpyHostToDevice
        );

        if (ec) {
            throw std::system_error(ec);
        }
    }

    void construct(T *device_ptr) noexcept(std::is_trivially_default_constructible<T>::value) {
        do_construct_default(std::is_trivially_default_constructible<T>(), device_ptr);
    }

    void destroy(T*) noexcept {}

private:
    void do_construct_default(std::true_type, T*) noexcept {}

    void do_construct_default(std::false_type, T *device_ptr) {
        static const T host_default;

        const std::error_code ec = cudaMemcpy(
            device_ptr,
            std::addressof(host_default),
            sizeof(T),
            cudaMemcpyHostToDevice
        );

        if (ec) {
            throw std::system_error(ec);
        }
    }
};

template <typename T>
using CudaVector = std::vector<T, CudaAllocator<T>>;

template <typename T, typename A = std::allocator<T>>
std::vector<T, A> to_host(const std::vector<T, CudaAllocator<T>> &device) {
    std::vector<T, A> host(device.size());

    const std::error_code ec = cudaMemcpy(
        host.data(),
        device.data(),
        device.size() * sizeof(T),
        cudaMemcpyDeviceToHost
    );

    if (ec) {
        throw std::system_error(ec);
    }

    return host;
}

template <typename T>
T to_host(const std::unique_ptr<T, CudaDeleter> &ptr) {
    T host;

    const std::error_code ec = cudaMemcpy(
        std::addressof(host),
        std::addressof(*ptr),
        sizeof(T),
        cudaMemcpyDeviceToHost
    );

    if (ec) {
        throw std::system_error(ec);
    }

    return host;
}

__global__ void transform(const Transform &tf, const Vector *input, Vector *output) {
    output[threadIdx.x] = tf(input[threadIdx.x]);
}

int main(int argc, char* argv[]) {
    const Transform host_tf = {
        {
            1, 0, 0,
            0, 0, -1,
            0, 1, 0
        },
        {1, 0, 0}
    };

    const auto device_tf_ptr = to_device(host_tf);

    const std::array<Vector, 8> host_inputs = {{
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}
    }};

    const CudaVector<Vector> device_inputs(host_inputs.cbegin(), host_inputs.cend());
    CudaVector<Vector> device_outputs(8);

    // Launch the kernel.
    transform<<<1, 8>>>(*device_tf_ptr, device_inputs.data(), device_outputs.data());

    const auto host_outputs = to_host(device_outputs);

    for (const Vector &v : host_outputs) {
        std::cout << v << '\n';
    }
}
