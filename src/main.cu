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

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

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

constexpr std::size_t div_to_inf(std::size_t x, std::size_t y) noexcept {
    const std::size_t res = x / y;

    if (x % y != 0) {
        return res + 1;
    }

    return res;
}

__host__ __device__ sl::uchar4 from_packed(float x) {
    union Converter {
        float scalar;
        sl::uchar4 vector;
    };

    return Converter{x}.vector;
}

__host__ __device__ float pack(sl::uchar4 x) noexcept {
    union Converter {
        sl::uchar4 vector;
        float scalar;
    };

    return Converter{x}.scalar;
}

sl::Mat random_xyzrgba(std::size_t width, std::size_t height) {
    const std::size_t numel = width * height;
    sl::Mat m(width, height, sl::MAT_TYPE_32F_C4);

    const auto gen_ptr = std::make_unique<std::mt19937>();
    std::uniform_real_distribution<float> pos_dist(-10, 10);
    std::uniform_int_distribution<std::uint8_t> color_dist;

    const auto gen_pos = [&gen_ptr, &pos_dist] {
        return pos_dist(*gen_ptr);
    };

    const auto gen_col = [&gen_ptr, &color_dist] {
        return color_dist(*gen_ptr);
    };

    const auto arr = m.getPtr<sl::float4>();
    for (std::size_t i = 0; i < numel; ++i) {
        arr[i][0] = gen_pos();
        arr[i][1] = gen_pos();
        arr[i][2] = gen_pos();
        arr[i][3] = pack({255, 255, 255, 127});
    }

    m.updateGPUfromCPU();

    return m;
}

__host__ __device__ std::uint32_t pack_bgra(
    std::uint8_t b, std::uint8_t g,
    std::uint8_t r, std::uint8_t a
)  noexcept {
    union Converter {
        std::uint8_t arr[4];
        std::uint32_t scalar;
    };

    return Converter{{b, g, r, a}}.scalar;
}

/**
 *  @param input Points to an array of length n. Each element must be
 *               a 4-tuple (X, Y, Z, RGBA) corresponding to a depth
 *               map. The RGBA component is packed into a 32-bit float,
 *               with each element being an 8-bit unsigned integer.
 *  @param output Points to a matrix with m rows and p columns. Each
 *                element must be a 4-tuple (B, G, R, A). The elements
 *                should be stored contiguously in row-major order.
 *  @param resolution The resolution (in meters) of each pixel in
 *                    output, such that each pixel represents a
 *                    (resolution x resolution) square.
 *  @param x_offset The offset (in meters) between the center of the
 *                  matrix and its leftmost edge, such that the pixel
 *                  at (0, 0) is located in free space at (-x_offset,
 *                  -y_offset).
 *  @param y_offset The offset (in meters) between the center of the
 *                  matrix and its topmost edge, such that the pixel
 *                  at (0, 0) is located in free space at (-x_offset,
 *                  -y_offset).
 */
__global__ void transform(const Transform &tf, const sl::float4 *input, std::uint32_t n,
                          std::uint32_t *output, std::uint32_t m, std::uint32_t p,
                          float resolution, float x_offset, float y_offset) {
    const std::uint32_t pixel_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pixel_idx >= n) {
        return;
    }

    const sl::float4 &elem = input[pixel_idx];
    const Vector transformed = tf({elem[0], elem[1], elem[2]});

    const float pixel_x = (transformed.x() + x_offset) / resolution;
    const float pixel_y = (transformed.y() + y_offset) / resolution;

    if (pixel_x < 0 || pixel_y < 0) {
        return;
    }

    const auto i = static_cast<std::uint32_t>(pixel_y); // row idx
    const auto j = static_cast<std::uint32_t>(pixel_x); // col idx

    if (i >= m || j >= p) {
        return;
    }

    const sl::uchar4 rgba = from_packed(elem[3]);
    const std::uint32_t output_idx = i * p + j;
    output[output_idx] = pack_bgra(rgba[2], rgba[1], rgba[0], rgba[3]);
}

int main(int argc, char* argv[]) {
    const Transform host_tf = {
        {
            0.5, -0.14644661, 0.85355339,
            0.5, 0.85355339, -0.14644661,
            0.70710678, 0.5, 0.5
        },
        {1, 0, 0}
    };

    const auto device_tf_ptr = to_device(host_tf);

    constexpr std::uint32_t WIDTH = 1280;
    constexpr std::uint32_t HEIGHT = 720;
    const sl::Mat input = random_xyzrgba(WIDTH, HEIGHT);

    constexpr float RESOLUTION = 0.05;
    constexpr float X_RANGE = 20;
    constexpr float Y_RANGE = 20;

    constexpr auto OUTPUT_ROWS = static_cast<std::uint32_t>(Y_RANGE / RESOLUTION);
    constexpr auto OUTPUT_COLS = static_cast<std::uint32_t>(X_RANGE / RESOLUTION);

    cv::cuda::GpuMat output(OUTPUT_COLS, OUTPUT_ROWS, CV_8UC4, cv::Scalar(0, 0, 0, 255));

    constexpr std::uint32_t NUMEL = WIDTH * HEIGHT;
    constexpr std::uint32_t BLOCKSIZE = 256;
    constexpr std::uint32_t NUM_BLOCKS = div_to_inf(NUMEL, BLOCKSIZE);

    cudaDeviceSynchronize();
    const auto start = std::chrono::steady_clock::now();

    transform<<<NUM_BLOCKS, BLOCKSIZE>>>(
        *device_tf_ptr,
        input.getPtr<sl::float4>(sl::MEM_GPU),
        NUMEL,
        output.ptr<std::uint32_t>(),
        OUTPUT_ROWS,
        OUTPUT_COLS,
        RESOLUTION,
        X_RANGE / 2,
        Y_RANGE / 2
    );

    const cv::Mat output_host(output);

    cudaDeviceSynchronize();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    cv::imwrite("output.png", output_host, {cv::IMWRITE_PNG_COMPRESSION, 9});
}
