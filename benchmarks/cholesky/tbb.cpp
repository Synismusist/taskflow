#include "cholesky.hpp"
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <cmath>
#include <iostream>

void choleskyTBB(std::vector<std::vector<double>>& matrix, unsigned num_threads) {
    int n = matrix.size();

    tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, num_threads
    );

    // simple for because i was not able to create nodes and a flow graph for cholesky
    tbb::parallel_for(0, n, [&](int j) {
        matrix[j][j] = sqrt(matrix[j][j]);

        for (int i = j + 1; i < n; ++i) {
            matrix[i][j] /= matrix[j][j];
        }

        for (int i = j + 1; i < n; ++i) {
            for (int k = j + 1; k <= i; ++k) {
                matrix[i][k] -= matrix[i][j] * matrix[k][j];
            }
        }
    });

    // Set upper triangular part to zero
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            matrix[i][j] = 0.0;
        }
    }
}

std::chrono::microseconds measure_time_tbb(std::vector<std::vector<double>>& matrix, unsigned num_threads) {
    auto beg = std::chrono::high_resolution_clock::now();
    choleskyTBB(matrix, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
