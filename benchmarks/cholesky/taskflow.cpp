#include "cholesky.hpp"
#include <taskflow/taskflow.hpp>
#include <cmath>
#include <iostream>

void choleskyTaskflow(std::vector<std::vector<double>>& matrix, 
                      unsigned num_threads) {
  // Counter or size type should always be non-negative
  size_t n = matrix.size();
  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  // Move col_tasks creation outside the loop
  std::vector<tf::Task> col_tasks(n);

  for (size_t j = 0; j < n; ++j) {
    auto diag_task = taskflow.emplace([&matrix, j]() {
      matrix[j][j] = sqrt(matrix[j][j]);
    });

    for (size_t i = j + 1; i < n; ++i) {
      col_tasks[i] = taskflow.emplace([&matrix, j, i]() {
        matrix[i][j] /= matrix[j][j];
      });
      diag_task.precede(col_tasks[i]);
    }

    for (size_t i = j + 1; i < n; ++i) {
      for (size_t k = j + 1; k <= i; ++k) {
        auto update_task = taskflow.emplace([&matrix, j, i, k]() {
          matrix[i][k] -= matrix[i][j] * matrix[k][j];
        });
        col_tasks[i].precede(update_task);
      }
    }
  }

  executor.run(taskflow).wait();

  // Set upper triangular part to zero
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      matrix[i][j] = 0.0;
    }
  }
}

std::chrono::microseconds measure_time_taskflow(
    std::vector<std::vector<double>>& matrix, 
    unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  choleskyTaskflow(matrix, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
