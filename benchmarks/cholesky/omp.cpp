#include "cholesky.hpp"
#include <omp.h>


// TODO: indent 2 space 
//       use up to 80 characteris per line
// you can take a look at Google C++ code style guideline: https://google.github.io/styleguide/cppguide.html

void choleskyOMP(std::vector<std::vector<double>>& matrix, unsigned num_threads) {
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

                // non-diag elements
                for (int i = j + 1; i < n; i++) {
                    #pragma omp task depend(in: matrix[j][j]) depend(inout: matrix[i][j])
                    {
                        matrix[i][j] /= matrix[j][j];
                    }
                }

                for (int i = j + 1; i < n; i++) {
                    for (int k = j + 1; k <= i; k++) {
                        #pragma omp task depend(in: matrix[i][j], matrix[k][j]) depend(inout: matrix[i][k])
                        {
                            matrix[i][k] -= matrix[i][j] * matrix[k][j];
                        }
                    }
                }
            }
        }
    }

    // make it L.T
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            matrix[i][j] = 0.0;
        }
    }
}

std::chrono::microseconds measure_time_OMP(
  const std::vector<std::vector<int>>& matrix, unsigned num_threads
){
    auto beg = std::chrono::high_resolution_clock::now();
    choleskyOMP(matrix, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
