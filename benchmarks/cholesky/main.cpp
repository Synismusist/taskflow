#include "cholesky.hpp"
#include <CLI11.hpp>

void cholesky(
  const std::string& model,
  unsigned num_threads,
  unsigned num_rounds
  ) 
{
  //creating matix of size from 10x10 to 100x100 with increments from 10s
  //matrix is a random symmetrix matrix and to do LL'
    
  for(size_t dimension = 10; dimension<=100; dimension+=10){

    // TODO: move this matrix outside the loop so you don't iteratively create and destroy
    //       the vector
    //       use vector::resize instead
    std::vector<std::vector<int>> matrix(dimension, std::vector<int>(dimension));
    
    // TODO: move this random device outside the loop
    //       the first two lines are very expensive
    std::random_device rndDev;
    std::mt19937 gen(rndDev());
    std::uniform_int_distribution<int> dis(0,100);
    
    for(size_t i=0; i<dimension; i++){
      matrix[i][i]=(dis(gen));
      for(size_t j=0; j<i; j++){
        int temp = dis(gen);
        matrix[i][j]=temp;
        matrix[j][i]=temp;
      }
    }
    
    // TODO: normally, compiler will figure the type for constants
    //       but you can explicitly specify the type 
    //       int = 0
    //       float = 0.0f
    //       double = 0.0
    double total_duration{0.0};

    for (unsigned i = 0; i < num_rounds; ++i) {
      if (model == "tf") {
        total_duration += measure_time_taskflow(matrix, num_threads).count();
      } else if (model == "tbb") {
        total_duration += measure_time_tbb(matrix, num_threads).count();
      } else if (model == "omp") {
        total_duration += measure_time_omp(matrix, num_threads).count();
      }
    }

    double avg_runtime_ms = total_duration / num_rounds;

    std::cout << std::setw(12) << dimension << std::setw(12) << avg_runtime_ms << std::endl;
  }
}
  
int main(int argc, char* argv[]) {
    CLI::App app{"CholeskyDecomp"};

    unsigned num_threads = 1;
    app.add_option("-t,--threads", num_threads, "Number of threads (default=1)");

    unsigned num_rounds = 1;
    app.add_option("-r,--rounds", num_rounds, "Number of benchmark rounds (default=1)");

    std::string model = "tf"; // Default to taskflow

    // TODO: name a new method "tf-dynamic"

    app.add_option("-m,--model", model,
                   "Parallelization method: tf|tbb|omp (default=tf)")
       ->check([](const std::string& p) {
           if (p != "tf" && p != "tbb" && p != "omp") {
               return "incorrect model it should be \"tbb\", \"omp\", or \"tf\"";
           }
           return "";
       });

    CLI11_PARSE(app, argc, argv);
    std::cout<<"Model: "<< model<<std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Rounds: " << num_rounds << std::endl;

    cholesky(model, num_threads, num_rounds);

    return 0;
}
