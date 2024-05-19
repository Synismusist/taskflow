#include "box_blur.hpp"
#include <tbb/flow_graph.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

void boxBlurEffectTBB(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius) {
    tbb::flow::graph g;

    auto compute_node = [&](const tbb::blocked_range<size_t>& r) {
        for (size_t y = r.begin(); y != r.end(); ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t sumRed = 0;
                size_t sumGreen = 0;
                size_t sumBlue = 0;
                size_t neighboringPixelCount = 0;

                for (int j = static_cast<int>(y) - blur_radius; j <= static_cast<int>(y) + blur_radius; ++j) {
                    for (int i = static_cast<int>(x) - blur_radius; i <= static_cast<int>(x) + blur_radius; ++i) {
                        if (i >= 0 && i < static_cast<int>(width) && j >= 0 && j < static_cast<int>(height)) {
                            size_t neighboringInd = (j * width + i) * channels;
                            sumRed += image_data[neighboringInd];
                            sumGreen += image_data[neighboringInd + 1];
                            sumBlue += image_data[neighboringInd + 2];
                            neighboringPixelCount++;
                        }
                    }
                }

                if (neighboringPixelCount > 0) {
                    sumRed /= neighboringPixelCount;
                    sumGreen /= neighboringPixelCount;
                    sumBlue /= neighboringPixelCount;
                }

                size_t currentInd = (y * width + x) * channels;
                image_data[currentInd] = sumRed;
                image_data[currentInd + 1] = sumGreen;
                image_data[currentInd + 2] = sumBlue;
            }
        }
    };

    tbb::flow::broadcast_node<tbb::blocked_range<size_t>> start_node(g);
    tbb::flow::function_node<tbb::blocked_range<size_t>> blur_node(g, tbb::flow::unlimited, compute_node);

    tbb::flow::make_edge(start_node, blur_node);

    start_node.try_put(tbb::blocked_range<size_t>(0, height));
    g.wait_for_all();
}

std::chrono::microseconds measure_time_tbb(
    unsigned char* image_data,
    size_t width,
    size_t height,
    size_t channels,
    size_t blur_radius
) {
    auto beg = std::chrono::high_resolution_clock::now();
    boxBlurEffectTBB(image_data, width, height, channels, blur_radius);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
