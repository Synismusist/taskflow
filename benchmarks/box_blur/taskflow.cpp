#include "box_blur.hpp"
#include <taskflow/taskflow.hpp>

void boxBlurTf(unsigned char* image_data, size_t width, size_t channels, size_t blur_radius, size_t start_row, size_t end_row) {
    for (size_t y = start_row; y < end_row; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t sumRed = 0;
            size_t sumGreen = 0;
            size_t sumBlue = 0;
            int neighboringPixelCount = 0;

            for (int j = y - blur_radius; j <= (int) (y + blur_radius); ++j) {
                for (int i = x - blur_radius; i <= (int) (x + blur_radius); ++i) {
                    if (i >= 0 && i < (int) width && j >= 0 && j < (int) end_row) {
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

void boxBlurEffectTf(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius, unsigned num_threads) {
    
    tf::Executor executor(num_threads);
    tf::Taskflow taskflow;

    int chunk_size = (height + num_threads - 1) / num_threads; //distributing among rows of image
    for (size_t start_row = 0; start_row < height; start_row += chunk_size) {
        size_t end_row = std::min((int) (start_row + chunk_size), (int)height);
        taskflow.emplace([&image_data, width, channels, blur_radius, start_row, end_row](){
            boxBlurTf(image_data, width, channels, blur_radius, start_row, end_row);
        });
    }
    
    executor.run(taskflow).wait();
}



std::chrono::microseconds measure_time_taskflow(
    unsigned char* image_data,
    size_t width, 
    size_t height, 
    size_t channels, 
    size_t blur_radius,
    unsigned num_threads
) {
    auto beg = std::chrono::high_resolution_clock::now();
    boxBlurEffectTf(image_data, width, height, channels, blur_radius, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}