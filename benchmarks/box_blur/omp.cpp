#include "box_blur.hpp"
#include <omp.h>

void boxBlurEffectOMP(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius, unsigned num_threads) {
    
    #pragma omp parallel for num_threads(num_threads)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t sumRed = 0;
            size_t sumGreen = 0;
            size_t sumBlue = 0;
            int neighboringPixelCount = 0;

            for (int j = y - blur_radius; j <= y + blur_radius; ++j) {
                for (int i = x - blur_radius; i <= x + blur_radius; ++i) {
                    if (i >= 0 && i < width && j >= 0 && j < height) {
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

std::chrono::microseconds measure_time_omp(
    unsigned char* image_data,
    size_t width, 
    size_t height, 
    size_t channels, 
    size_t blur_radius,
    unsigned num_threads
) {
    auto beg = std::chrono::high_resolution_clock::now();
    boxBlurEffectOMP(image_data, width, height, channels, blur_radius, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
