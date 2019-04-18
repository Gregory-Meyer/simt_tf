#include <simt_tf/simt_tf.h>

#include <cstdlib>
#include <chrono>
#include <iostream>

#include <cuda.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

#include <sl/Camera.hpp>
#include <sl/Core.hpp>

#define TRY(MSG, ...) \
    do {\
        const sl::ERROR_CODE errc = __VA_ARGS__;\
        \
        if (errc != sl::SUCCESS) {\
            const sl::String msg = sl::toString(errc);\
            std::cerr << MSG << ": " << msg.c_str() << '\n';\
            std::exit(EXIT_FAILURE);\
        }\
    } while (false)

int main(int argc, const char *const argv[]) {
    sl::InitParameters params;

    params.camera_fps = 60;
    params.coordinate_units = sl::UNIT_METER;
    params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD;

    if (argc > 1) {
        params.svo_input_filename = argv[1];
        params.input.setFromSVOFile(argv[1]);
    }

    sl::Camera camera;
    TRY("couldn't open ZED camera", camera.open(std::move(params)));

    const simt_tf::Transform tf = {
        {1, 0, 0,
         0, 1, 0,
         0, 0, 1},
        {0, 0, 0}
    };

    cv::cuda::GpuMat output(1080, 1080, CV_8UC4);
    sl::Mat pointcloud;

    const auto start = std::chrono::steady_clock::now();
    int num_frames = 0;

    while (camera.grab() == sl::SUCCESS) {
        simt_tf::pointcloud_birdseye(tf, camera, pointcloud, output, 0.01);
        const cv::Mat host_birdseye(output);

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - start;
        ++num_frames;

        const double fps = static_cast<double>(num_frames) / elapsed.count();

        std::cout << fps << "\n";

        cv::imshow("transformed", host_birdseye);
        cv::waitKey(1);
    }
}
