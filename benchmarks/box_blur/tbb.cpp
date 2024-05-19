#include "box_blur.hpp"
#include <tbb/flow_graph.h>



std::chrono::microseconds measure_time_taskflow(
    unsigned char* image_data,
    size_t width, 
    size_t height, 
    size_t channels, 
    size_t blur_radius,
    unsigned num_threads
) {
    auto beg = std::chrono::high_resolution_clock::now();
    boxBlurEffectTBB(image_data, width, height, channels, blur_radius, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
