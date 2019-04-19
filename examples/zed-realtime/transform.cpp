#include <simt_tf/simt_tf.h>

#include <csignal>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

#include <sl/Camera.hpp>
#include <sl/Core.hpp>

#define TRY(MSG, ...) \
    do {\
        const sl::ERROR_CODE TRY_RESERVED_ERRC = __VA_ARGS__;\
        \
        if (TRY_RESERVED_ERRC != sl::SUCCESS) {\
            const sl::String msg = sl::toString(TRY_RESERVED_ERRC);\
            std::cerr << MSG << ": " << msg.c_str() << '\n';\
            std::exit(EXIT_FAILURE);\
        }\
    } while (false)

static std::atomic<bool> keep_running(true);

static void handle_sigint(int) noexcept {
    keep_running.store(false);
}

int main(int argc, const char *const argv[]) {
    sl::InitParameters params;

    params.camera_fps = 15;
    params.coordinate_units = sl::UNIT_METER;
    params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD;
    params.depth_mode = sl::DEPTH_MODE_ULTRA;
    params.camera_resolution = sl::RESOLUTION_HD2K;

    bool from_svo = false;

    if (argc > 1) {
        params.svo_input_filename = argv[1];
        params.input.setFromSVOFile(argv[1]);
        from_svo = true;
    }

    sl::Camera camera;
    TRY("couldn't open ZED camera", camera.open(std::move(params)));
    std::signal(SIGINT, handle_sigint);

    const simt_tf::Transform tf = {
        {1, 0, 0,
         0, 1, 0,
         0, 0, 1},
        {0, 0, 0}
    };

    cv::cuda::GpuMat output(1024, 1024, CV_8UC4);
    cv::Mat host_output;
    sl::Mat pointcloud;

    const auto start = std::chrono::steady_clock::now();
    int num_frames = 0;

    while (keep_running.load()) {
        const sl::ERROR_CODE errc = camera.grab();

        if (errc != sl::SUCCESS) {
            if (errc == sl::ERROR_CODE_NOT_A_NEW_FRAME) {
                if (from_svo) {
                    break;
                } else {
                    continue;
                }
            }

            const sl::String msg = sl::toString(errc);
            std::cerr << "Camera::grab() failed: " << msg.c_str() << '\n';

            return 1;
        }

        simt_tf::pointcloud_birdseye(tf, camera, pointcloud, output, 0.01);
        output.download(host_output);

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - start;
        ++num_frames;

        const double fps = static_cast<double>(num_frames) / elapsed.count();

        std::cout << "frame " << num_frames << ": " << fps << " fps\n";

        cv::imshow("bird's eye transform", host_output);
        cv::waitKey(1);
    }
}
