#include "cholesky.hpp"
#include <taskflow/taskflow.hpp>


void choleskyDynamicTaskflow(std::vector<std::vector<int>>& matrix, 
                      unsigned num_threads) {
  // Counter or size type should always be non-negative
  size_t n = matrix.size();
  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  // Dynamic creation of tasks with dependencies
  for (size_t j = 0; j < n; ++j) {
    // Diagonal task
    auto diag_task = executor.silent_dependent_async([&matrix, j]() {
      matrix[j][j] = sqrt(matrix[j][j]);
    });

    // Column tasks
    std::vector<tf::AsyncTask> col_tasks;
    for (size_t i = j + 1; i < n; ++i) {
      col_tasks.push_back(
        executor.silent_dependent_async([&matrix, j, i]() {
          matrix[i][j] /= matrix[j][j];
        }, diag_task)
      );
    }

    // Update tasks
    for (size_t i = j + 1; i < n; ++i) {
      for (size_t k = j + 1; k <= i; ++k) {
        executor.silent_dependent_async([&matrix, j, i, k]() {
          matrix[i][k] -= matrix[i][j] * matrix[k][j];
        }, col_tasks[i - (j + 1)]); // Adjust index for col_tasks
      }
    }
  }

  executor.wait_for_all();

  // Set upper triangular part to zero
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      matrix[i][j] = 0.0;
    }
  }
}

std::chrono::microseconds measure_time_taskflow_dynamic(
    std::vector<std::vector<int>>& matrix, 
    unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  choleskyDynamicTaskflow(matrix, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
