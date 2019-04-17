#include <simt_tf/simt_tf.h>

#include <cstdlib>
#include <iostream>

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

int main() {
    sl::InitParameters params;

    params.camera_fps = 60;
    params.coordinate_units = sl::UNIT_METER;
    params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD;

    sl::Camera camera;
    TRY("couldn't open ZED camera", camera.open(std::move(params)));

    const simt_tf::Transform tf = {
        {1, 0, 0,
         0, 1, 0,
         0, 0, 1},
        {0, 0, 0}
    };

    while (camera.grab() == sl::SUCCESS) {
        const cv::cuda::GpuMat dev_birdseye =
            simt_tf::pointcloud_birdseye(tf, camera, 1080, 1920, 0.01);
        const cv::Mat host_birdseye(dev_birdseye);

        cv::imshow("transformed", host_birdseye);
        cv::waitKey();
    }
}
