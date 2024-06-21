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

  std::vector<std::vector<int>> matrix(100, std::vector<int>(100));
  std::random_device rndDev;
  std::mt19937 gen(rndDev());
  std::uniform_int_distribution<int> dis(0,100);

  for(size_t dimension = 100; dimension>=10; dimension-=10){

    matrix.resize(dimension);

    for(auto& row: matrix){
      row.resize(dimension);
    }
    
    for(size_t i=0; i<dimension; i++){
      matrix[i][i]=(dis(gen));
      for(size_t j=0; j<i; j++){
        int temp = dis(gen);
        matrix[i][j]=temp;
        matrix[j][i]=temp;
      }
    }
    
    double total_duration{0.0};

    for (unsigned i = 0; i < num_rounds; ++i) {
      if (model == "tf") {
        total_duration += measure_time_taskflow(matrix, num_threads).count();
      } 
      /*else if (model == "tbb") {
        total_duration += measure_time_tbb(matrix, num_threads).count();
      }*/ 
      else if (model == "omp") {
        total_duration += measure_time_omp(matrix, num_threads).count();
      } else if (model == "tf-dynamic") {
        total_duration += measure_time_taskflow_dynamic(matrix, num_threads).count();
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


    app.add_option("-m,--model", model,
                   "Parallelization method: tf|tbb|omp|tf-dynamic (default=tf)")
       ->check([](const std::string& p) {
           if (p != "tf" && p != "tbb" && p != "omp" && p!= "tf-dynamic") {
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
