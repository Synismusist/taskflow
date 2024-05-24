#ifndef box_blur_hpp 
#define box_blur_hpp 

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cstdlib> 
#include <ctime>
#include <memory>
#include <random>
#include <thread>
#include <atomic>


std::chrono::microseconds measure_time_taskflow(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius, unsigned num_threads);
std::chrono::microseconds measure_time_tbb(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius, unsigned num_threads);
std::chrono::microseconds measure_time_omp(unsigned char* image_data, size_t width, size_t height, size_t channels, size_t blur_radius, unsigned num_threads);

#endif box_blur_hpp
