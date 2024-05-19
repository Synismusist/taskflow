#include <CLI11.hpp>
#include "box_blur.hpp"

void box_blur(
    const std::string& model,
    size_t blur_radius,
    unsigned num_threads,
    unsigned num_rounds) {

    for (size_t width = 100; width <= 1000; width += 100) { //benchmarking over 10 different image sizes
        size_t height = width; // Square image
        size_t channels = 3;   // RGB channels

        std::unique_ptr<unsigned char[]> image_data = std::make_unique<unsigned char[]>(width *height*channels);
        //random pixel assignment as the values dont really matter:
        std::random_device rndDev;
        std::mt19937 gen(rndDev());
        std::uniform_int_distribution<int> dis(0,255);


        for (int i = 0; i < width*height*channels; ++i) {
            image_data[i] = static_cast<unsigned char>(dis(gen));
        }


        double total_duration{0};

        for (unsigned i = 0; i < num_rounds; ++i) {
            if (model == "tf") {
                total_duration += measure_time_taskflow(image_data, width, height, channels, blur_radius, num_threads).count();
            } else if (model == "tbb") {
                total_duration += measure_time_tbb(image_data, width, height, channels, blur_radius, num_threads).count();
            } else if (model == "omp") {
                total_duration += measure_time_omp(image_data, width, height, channels, blur_radius, num_threads).count();
            }
        }


        double avg_runtime_ms = total_duration / num_rounds;

        std::cout << std::setw(12) << width
                  << std::setw(12) << avg_runtime_ms
                  << std::endl;
    }
}


int main(int argc, char* argv[]) {
    CLI::App app{"BoxBlur"};

    unsigned num_threads = 1;
    app.add_option("-t,--threads", num_threads, "Number of threads (default=1)");

    unsigned num_rounds = 1;
    app.add_option("-r,--rounds", num_rounds, "Number of benchmark rounds (default=1)");

    int blur_radius = 50;
    app.add_option("-b,--blur-radius", blur_radius, "Blur radius (default=50)");

    std::string model = "tf"; // Default to sequential
    app.add_option("-p,--model", model,
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
    std::cout << "Blur radius: " << blur_radius << std::endl;

    box_blur(model, blur_radius, num_threads, num_rounds);

    return 0;
}
