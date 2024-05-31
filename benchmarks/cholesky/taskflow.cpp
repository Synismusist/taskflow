#include "cholesky.hpp"
#include <taskflow/taskflow.hpp>
#include <cmath>
#include <iostream>

void choleskyTaskflow(std::vector<std::vector<double>>& matrix, unsigned num_threads) {
    int n = matrix.size();
    tf::Executor executor(num_threads);
    tf::Taskflow taskflow;

    for (int j = 0; j < n; ++j) {
        auto diag_task = taskflow.emplace([&matrix, j]() {
            matrix[j][j] = sqrt(matrix[j][j]);
        });

        std::vector<tf::Task> col_tasks;
        for (int i = j + 1; i < n; ++i) {
            col_tasks.push_back(taskflow.emplace([&matrix, j, i]() {
                matrix[i][j] /= matrix[j][j];
            });
            diag_task.precede(col_tasks.back());
        }

        for_each(taskflow, col_tasks.begin(), col_tasks.end(), [&](auto& col_task) {
            for (int i = j + 1; i < n; ++i) {
                for (int k = j + 1; k <= i; ++k) {
                    taskflow.emplace([&matrix, j, i, k]() {
                        matrix[i][k] -= matrix[i][j] * matrix[k][j];
                    }).precede(col_task);
                }
            }
        });
    }

    executor.run(taskflow);

    // Set upper triangular part to zero
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            matrix[i][j] = 0.0;
        }
    }
}

std::chrono::microseconds measure_time_taskflow(std::vector<std::vector<double>>& matrix, unsigned num_threads) {
    auto beg = std::chrono::high_resolution_clock::now();
    choleskyTaskflow(matrix, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
