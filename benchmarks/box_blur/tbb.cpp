#include "box_blur.hpp"
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
#include <algorithm>
#include <memory>

void boxBlurEffectTBB(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius, unsigned num_threads) {
    using namespace tbb;
    using namespace tbb::flow;

    //limiting to the specific thread count
    global_control c(global_control::max_allowed_parallelism, num_threads);
    
    //flow graph initialization
    graph g;
    std::vector<continue_node<continue_msg>*> nodes;

    int chunk_size = (height + num_threads - 1) / num_threads; //distributing among rows 

    //creating a node for each row
    for (int start_row = 0; start_row < height; start_row += chunk_size) {
        int end_row = std::min(start_row + chunk_size, height);
        auto node = new continue_node<continue_msg>(g, [=, &image_data](const continue_msg&) {
            boxBlurTBB(image_data, width, channels, blur_radius, start_row, end_row);
        });
        nodes.push_back(node);
    }

    //connecting the nodes
    for (size_t i = 1; i < nodes.size(); ++i) {
        make_edge(*nodes[i-1], *nodes[i]);
    }

    nodes.front()->try_put(continue_msg()); //execute
    g.wait_for_all();

    //clean up
    for (auto node : nodes) {
        delete node;
    }
}

void boxBlurTBB(unsigned char* image_data, size_t width, size_t channels, size_t blur_radius, int start_row, int end_row) {
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t sumRed = 0;
            size_t sumGreen = 0;
            size_t sumBlue = 0;
            int neighboringPixelCount = 0;

            for (int j = y - blur_radius; j <= y + blur_radius; ++j) {
                for (int i = x - blur_radius; i <= x + blur_radius; ++i) {
                    if (i >= 0 && i < width && j >= 0 && j < end_row) {
                        int neighboringInd = (j * width + i) * channels;
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

            int currentInd = (y * width + x) * channels;
            image_data[currentInd] = sumRed;
            image_data[currentInd + 1] = sumGreen;
            image_data[currentInd + 2] = sumBlue;
        }
    }
}

std::chrono::microseconds measure_time_tbb(
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
