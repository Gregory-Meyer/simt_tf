#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <system_error>
#include <type_traits>
#include <vector>

#include "err.cuh"
#include "matrix.cuh"
#include "transform.cuh"
#include "vector.cuh"

#include <sl/Core.hpp>

struct CudaDeleter {
    template <typename T>
    void operator()(T *ptr) noexcept {
        static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible");

        cudaFree(ptr);
    }
};

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
CudaVector<T> to_device(const T host[], std::size_t n) {
    static_assert(
        std::is_trivially_default_constructible<T>::value && std::is_trivially_copyable<T>::value,
        "T must be trivially default constructible and copyable"
    );

    CudaVector<T> device(n);

    const std::error_code ec = cudaMemcpy(
        device.data(),
        host,
        n * sizeof(T),
        cudaMemcpyHostToDevice
    );

    if (ec) {
        throw std::system_error(ec);
    }

    return device;
}

template <typename T, typename A>
CudaVector<T> to_device(const std::vector<T, A> &host) {
    return to_device(host.data(), host.size());
}

template <typename T, std::size_t N>
CudaVector<T> to_device(const std::array<T, N> &host) {
    return to_device(host.data(), N);
}

template <typename T, std::size_t N>
CudaVector<T> to_device(const T (&host)[N]) {
    return to_device(host, N);
}

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

__global__ void transform(const Transform &tf, sl::float4 *to_transform, std::size_t n) {
    const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        sl::float4 &elem = to_transform[i];
        const Vector input = {elem[0], elem[1], elem[2]};
        const Vector transformed = tf(input);

        elem[0] = transformed[0];
        elem[1] = transformed[1];
        elem[2] = transformed[2];
    }
}

constexpr std::size_t div_to_inf(std::size_t x, std::size_t y) noexcept {
    const std::size_t res = x / y;

    if (x % y != 0) {
        return res + 1;
    }

    return res;
}

class Rgba {
public:
    Rgba() noexcept = default;

    __host__ __device__ Rgba(std::uint8_t r, std::uint8_t g,
                             std::uint8_t b, std::uint8_t a) noexcept
    : data_{r, g, b, a} { }

    static Rgba from_packed(float packed) noexcept {
        union Caster {
            float packed;
            Rgba unpacked;
        };

        return Caster{packed}.unpacked;
    }

    float to_packed() const noexcept {
        union Caster {
            Rgba unpacked;
            float packed;
        };

        return Caster{*this}.packed;
    }

    __host__ __device__ std::uint8_t& r() noexcept {
        return data_[0];
    }

    __host__ __device__ const std::uint8_t& r() const noexcept {
        return data_[0];
    }

    __host__ __device__ std::uint8_t& g() noexcept {
        return data_[1];
    }

    __host__ __device__ const std::uint8_t& g() const noexcept {
        return data_[1];
    }

    __host__ __device__ std::uint8_t& b() noexcept {
        return data_[2];
    }

    __host__ __device__ const std::uint8_t& b() const noexcept {
        return data_[2];
    }

    __host__ __device__ std::uint8_t& a() noexcept {
        return data_[3];
    }

    __host__ __device__ const std::uint8_t& a() const noexcept {
        return data_[3];
    }

    __host__ __device__ std::uint8_t& operator[](std::size_t idx) noexcept {
        assert(idx < 4);

        return data_[idx];
    }

    __host__ __device__ const std::uint8_t& operator[](std::size_t idx) const noexcept {
        assert(idx < 4);

        return data_[idx];
    }

private:
    alignas(float) std::uint8_t data_[4];
};

sl::Mat random_xyzrgba(std::size_t width, std::size_t height) {
    const std::size_t numel = width * height;
    sl::Mat m(width, height, sl::MAT_TYPE_32F_C4);

    const auto gen_ptr = std::make_unique<std::mt19937>();
    std::uniform_real_distribution<float> dist(-10, 10);

    const auto arr = m.getPtr<sl::float4>();
    for (std::size_t i = 0; i < numel; ++i) {
        arr[i][0] = dist(*gen_ptr);
        arr[i][1] = dist(*gen_ptr);
        arr[i][2] = dist(*gen_ptr);
        arr[i][3] = Rgba(127, 127, 127, 255).to_packed();
    }

    m.updateGPUfromCPU();

    return m;
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

    constexpr std::size_t WIDTH = 1280;
    constexpr std::size_t HEIGHT = 720;
    sl::Mat data = random_xyzrgba(WIDTH, HEIGHT);

    constexpr std::size_t NUMEL = WIDTH * HEIGHT;
    constexpr std::size_t BLOCKSIZE = 256;
    constexpr std::size_t NUM_BLOCKS = div_to_inf(NUMEL, BLOCKSIZE);

    cudaDeviceSynchronize();
    const auto start = std::chrono::steady_clock::now();
    // Launch the kernel.
    transform<<<NUM_BLOCKS, BLOCKSIZE>>>(
        *device_tf_ptr,
        data.getPtr<sl::float4>(sl::MEM_GPU),
        NUMEL
    );
    data.updateCPUfromGPU();

    cudaDeviceSynchronize();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";
}
