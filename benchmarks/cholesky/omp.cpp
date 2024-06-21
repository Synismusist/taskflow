#include "cholesky.hpp"
#include <omp.h>

// Function for Cholesky decomposition using OpenMP
//TODO omp does not support vectors so instead do int **
//C Style for OMP if error 
void choleskyOMP(double** matrix, int n, unsigned num_threads) {

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

double** vectorToArray(const std::vector<std::vector<int>>& matrix) {
  int n = matrix.size();
  double** array = new double*[n];
  for (int i = 0; i < n; ++i) {
    array[i] = new double[n];
    for (int j = 0; j < n; ++j) {
      array[i][j] = static_cast<double>(matrix[i][j]);
    }
  }
  return array;
}

void deallocateArray(double** array, int n) {
  for (int i = 0; i < n; ++i) {
    delete[] array[i];
  }
  delete[] array;
}



// Measure time for OpenMP Cholesky decomposition
std::chrono::microseconds measure_time_omp
    (std::vector<std::vector<int>>& matrix, 
    unsigned num_threads) {

  double** matrix_copy = vectorToArray(matrix);  // Make a copy of the matrix
  int n = matrix.size();
  auto beg = std::chrono::high_resolution_clock::now();
  choleskyOMP(matrix_copy, n, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  deallocateArray(matrix_copy, n);
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);

}
