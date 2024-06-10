#include "cholesky.hpp"
#include <omp.h>

// Function for Cholesky decomposition using OpenMP
void choleskyOMP(std::vector<std::vector<double>>& matrix, 
                 unsigned num_threads) {
  int n = matrix.size();

  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single
    {
      for (int j = 0; j < n; j++) {
        // Diagonal element
        #pragma omp task depend(inout: matrix[j][j])
        {
          matrix[j][j] = sqrt(matrix[j][j]);
        }

        // Non-diagonal elements
        for (int i = j + 1; i < n; i++) {
          #pragma omp task depend(in: matrix[j][j]) \
                           depend(inout: matrix[i][j])
          {
            matrix[i][j] /= matrix[j][j];
          }
        }

        for (int i = j + 1; i < n; i++) {
          for (int k = j + 1; k <= i; k++) {
            #pragma omp task depend(in: matrix[i][j], matrix[k][j]) \
                             depend(inout: matrix[i][k])
            {
              matrix[i][k] -= matrix[i][j] * matrix[k][j];
            }
          }
        }
      }
    }
  }

  // Make it lower triangular
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      matrix[i][j] = 0.0;
    }
  }
}

// Measure time for OpenMP Cholesky decomposition
std::chrono::microseconds measure_time_OMP
    (const std::vector<std::vector<double>>& matrix, 
    unsigned num_threads) {

  auto beg = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<double>> matrix_copy = matrix;  // Make a copy of the matrix
  choleskyOMP(matrix_copy, num_threads);

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);

}
