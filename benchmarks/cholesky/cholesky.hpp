#ifndef cholesky_hpp 
#define cholesky_hpp 

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
#include <vector>

std::chrono::microseconds measure_time_taskflow(std::vector<std::vector<int>>& matrix, unsigned num_threads);
std::chrono::microseconds measure_time_tbb(std::vector<std::vector<int>>& matrix, unsigned num_threads);
std::chrono::microseconds measure_time_omp(std::vector<std::vector<int>>& matrix, unsigned num_threads);
std::chrono::microseconds measure_time_taskflow_dynamic(std:: vector<std::vector<int>>& matrix, unsigned num_threads);

#endif // cholesky_hpp
