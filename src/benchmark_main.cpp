#include "workload_benchmark.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <workload_name> <mlir_file>" << std::endl;
        return 1;
    }

    std::string workload_name = argv[1];
    std::string mlir_file = argv[2];

    std::unique_ptr<BenchmarkBase> benchmark = BenchmarkFactory::getBenchmark(workload_name);
    if (!benchmark) {
        return 1;
    }

    if (!benchmark->loadMLIR(mlir_file)) {
        return 1;
    }

    benchmark->runWorkload();

    return 0;
}
