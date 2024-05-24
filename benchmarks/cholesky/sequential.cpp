#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

std::vector<std::vector<double>> choleskyDecomposition(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0;

            if (j == i) { /
                for (int k = 0; k < j; ++k) {
                    sum += L[j][k] * L[j][k];
                }
                if (A[j][j] - sum <= 0) {
                    throw std::runtime_error("Matrix is not positive definite.");
                }
                L[j][j] = std::sqrt(A[j][j] - sum);
            } else { 
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

int main(){
    
}