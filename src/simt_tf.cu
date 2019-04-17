#include <simt_tf/simt_tf.h>

#include "err.cuh"

#include <cassert>
#include <cmath>

namespace {

__host__ __device__ bool isnan(const sl::float4 &v) noexcept {
    return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]);
}

__host__ __device__ std::uint32_t as_u32(float x) noexcept {
    union Converter {
        float f;
        std::uint32_t i;
    };

    return Converter{x}.i;
}

__host__ __device__ constexpr std::size_t div_to_inf(std::size_t x, std::size_t y) noexcept {
    const std::size_t res = x / y;

    if (x % y != 0) {
        return res + 1;
    }

    return res;
}

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

    const std::error_code memcpy_ec =
        cudaMemcpy(device_ptr, std::addressof(host), sizeof(T), cudaMemcpyHostToDevice);

    if (memcpy_ec) {
        throw std::system_error(memcpy_ec);
    }

    return {device_ptr, CudaDeleter()};
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
__global__ static void transform(
    const simt_tf::Transform &tf, const sl::float4 *input,
    std::uint32_t n, std::uint32_t *output, std::uint32_t m,
    std::uint32_t p, float resolution, float x_offset, float y_offset
) {
    const std::uint32_t pixel_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pixel_idx >= n) {
        return;
    }

    const sl::float4 &elem = input[pixel_idx];

    if (isnan(elem)) {
        return;
    }

    const simt_tf::Vector transformed = tf({elem[0], elem[1], elem[2]});

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

    const std::uint32_t output_idx = i * p + j;
    output[output_idx] = as_u32(elem[3]);
}

} // namespace

namespace simt_tf {

/**
 *  Fetches the left-side pointcloud from a ZED camera, then performs a
 *  GPU-accelerated transform (translate then rotate) on each point and
 *  projects it into a bird's eye view.
 *
 *  @param tf Must be a valid coordinate transform, meaning that the
 *            determinant of its basis rotation matrix must be 1.
 *  @param camera Must be opened, have the latest data grabbed, and
 *                have support for depth images and pointclouds.
 *                It is assumed that x is forward, y, is left, and
 *                z is up, in accordance with ROS REP 103.
 *  @param output_cols The number of columns in the output matrix.
 *  @param output_rows The number of rows in the output matrix.
 *  @param output_resolution The resolution, in whatever unit the ZED
 *                           is configured to output in, of each cell
 *                           of the output matrix. In other words, each
 *                           cell represents a (resolution x
 *                           resolution) square in free space.
 *  @returns A transformed and bird's eye'd view of the camera's left
 *           side point cloud.
 *
 *  @throws std::system_error if any ZED SDK or CUDA function call
 *          fails.
 */
cv::cuda::GpuMat pointcloud_birdseye(
    const Transform &tf, sl::Camera &camera, std::size_t output_cols,
    std::size_t output_rows, float output_resolution
) {
    assert(camera.isOpened());

    const auto device_tf_ptr = to_device(tf);

    cv::cuda::GpuMat output(
        static_cast<int>(output_rows), static_cast<int>(output_cols),
        CV_8UC4, cv::Scalar(0, 0, 0, 255)
    );

    sl::Mat pointcloud;
    const std::error_code pc_ret =
        camera.retrieveMeasure(pointcloud, sl::MEASURE_XYZBGRA, sl::MEM_GPU);

    if (pc_ret) {
        throw std::system_error(pc_ret);
    }

    const float x_range = static_cast<float>(output_cols) * output_resolution;
    const float y_range = static_cast<float>(output_rows) * output_resolution;

    const auto numel = static_cast<std::uint32_t>(pointcloud.getResolution().area());
    constexpr std::uint32_t BLOCKSIZE = 256;
    const std::uint32_t num_blocks = div_to_inf(numel, BLOCKSIZE);

    transform<<<num_blocks, BLOCKSIZE>>>(
        *device_tf_ptr,
        pointcloud.getPtr<sl::float4>(sl::MEM_GPU),
        numel,
        output.ptr<std::uint32_t>(),
        static_cast<float>(output_rows),
        static_cast<float>(output_cols),
        output_resolution,
        x_range / 2,
        y_range / 2
    );

    cudaDeviceSynchronize();

    return output;
}

} // namespace simt_tf
