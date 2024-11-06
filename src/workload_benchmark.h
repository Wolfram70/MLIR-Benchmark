#ifndef WORKLOAD_BENCHMARK_H
#define WORKLOAD_BENCHMARK_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "/home/wolfram/mlir-gen/mlir-gen.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <memory>

#define EXECUTION_MODE "1024"

//module-serializer
class MLIRModuleSerializer {
public:
  // Constructor
  MLIRModuleSerializer(mlir::MLIRContext &context);

  // Load an MLIR module from a file
  mlir::LogicalResult load(const std::string &filename,
                           std::shared_ptr<mlir::ModuleOp> &module);

private:
  mlir::MLIRContext &context;
};

MLIRModuleSerializer::MLIRModuleSerializer(mlir::MLIRContext &context)
    : context(context) {}

// Load an MLIR module from a file
mlir::LogicalResult
MLIRModuleSerializer::load(const std::string &filename,
                           std::shared_ptr<mlir::ModuleOp> &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return mlir::failure();
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> owningModule =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!owningModule) {
    llvm::errs() << "Error can't load file " << filename << "\n";
    return mlir::failure();
  }

  module = std::make_shared<mlir::ModuleOp>(owningModule.release());
  return mlir::success();
}

// Benchmark base class
class BenchmarkBase {
protected:
    mlir::MLIRContext context;
    llvm::LLVMContext llvmContext;
    std::string workload_name;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::unique_ptr<llvm::Module> llvmModule;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
    execResults ExecutionResult;

public:
    BenchmarkBase(const std::string &name) : workload_name(name), context(mlir::MLIRContext()), llvmContext(llvm::LLVMContext()) {
        std::cout << "Creating benchmark for: " << workload_name << std::endl;

        //add func, affine, memref, arith, scf, cf dialect to the context
        context.loadDialect<mlir::func::FuncDialect, mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();

        mlir::registerBuiltinDialectTranslation(context);
    }

    bool lowerToLLVM() {
        // Lower the MLIR module to LLVM IR
        mlir::ConversionTarget target(context);
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addLegalOp<mlir::ModuleOp>();

        mlir::LLVMTypeConverter typeConverter(&context);

        mlir::RewritePatternSet patterns(&context);
        mlir::populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

        mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
        //apply full conversion
        if (failed(mlir::applyFullConversion(module.get(), target, frozenPatterns))) {
            std::cerr << "Lowered to LLVMIR." << std::endl;
            return false;
        }

        // Convert the MLIR module to LLVM IR
        llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    }

    // Parse the MLIR file
    bool loadMLIR(const std::string &filename) {
        std::ifstream mlirFile(filename);
        if (!mlirFile.is_open()) {
            std::cerr << "Error opening MLIR file: " << filename << std::endl;
            return false;
        }

        MLIRModuleSerializer serializer(context);
        std::shared_ptr<mlir::ModuleOp> module_ptr;
        if (serializer.load(filename, module_ptr).failed())
            return false;

        module = *module_ptr;

        lowerToLLVM();

        return true;
    }

    virtual void runWorkload() = 0;

    // Benchmarking methods
    void startTimer() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void endTimer() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    void printMetrics() {
        //sleep for 1 second to allow the system to update the metrics
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Execution Time: " << ExecutionResult.execTime << " ms" << std::endl;
        std::cout << "Memory Usage: " << ExecutionResult.memUsage << " MB" << std::endl;
        std::cout << "Cache Miss Rate: " << ExecutionResult.cacheMissRate * 100 << "%" << std::endl;
    }

    virtual ~BenchmarkBase() = default;
};

// Matrix multiplication benchmark class
class MatMulBenchmark : public BenchmarkBase {
public:
    MatMulBenchmark() : BenchmarkBase("Matrix Multiplication") {}

    void runWorkload() override {
        std::cout << "Running Matrix Multiplication workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "matmul", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Vector addition benchmark class
class VecAddBenchmark : public BenchmarkBase {
public:
    VecAddBenchmark() : BenchmarkBase("Vector Addition") {}

    void runWorkload() override {
        std::cout << "Running Matrix Multiplication workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "vecadd", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

//Batchnorm benchmark class
class BatchNormBenchmark : public BenchmarkBase {
public:
    BatchNormBenchmark() : BenchmarkBase("Batch Normalization") {}

    void runWorkload() override {
        std::cout << "Running Batch Normalization workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "batchnorm", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Convolution benchmark class
class ConvBenchmark : public BenchmarkBase {
public:
    ConvBenchmark() : BenchmarkBase("Convolution") {}

    void runWorkload() override {
        std::cout << "Running Convolution workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "conv", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Dot product benchmark class
class DotProductBenchmark : public BenchmarkBase {
public:
    DotProductBenchmark() : BenchmarkBase("Dot Product") {}

    void runWorkload() override {
        std::cout << "Running Dot Product workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "dotprod", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Elemadd benchmark class
class ElemAddBenchmark : public BenchmarkBase {
public:
    ElemAddBenchmark() : BenchmarkBase("Element-wise Addition") {}

    void runWorkload() override {
        std::cout << "Running Element-wise Addition workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "elemadd", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// FMA benchmark class
class FMABenchmark : public BenchmarkBase {
public:
    FMABenchmark() : BenchmarkBase("Fused Multiply-Add") {}

    void runWorkload() override {
        std::cout << "Running Fused Multiply-Add workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "fma", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// MatReduce benchmark class
class MatReduceBenchmark : public BenchmarkBase {
public:
    MatReduceBenchmark() : BenchmarkBase("Matrix Reduction") {}

    void runWorkload() override {
        std::cout << "Running Matrix Reduction workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "matreduction", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Softmax benchmark class
class SoftmaxBenchmark : public BenchmarkBase {
public:
    SoftmaxBenchmark() : BenchmarkBase("Softmax") {}

    void runWorkload() override {
        std::cout << "Running Softmax workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "softmax", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Matrix Transpose benchmark class
class MatTransposeBenchmark : public BenchmarkBase {
public:
    MatTransposeBenchmark() : BenchmarkBase("Matrix Transpose") {}

    void runWorkload() override {
        std::cout << "Running Matrix Transpose workload..." << std::endl;

        // auto executionEngine = mlir::ExecutionEngine::create(module.get());

        startTimer();
        ExecutionResult = ExecutionManager::finalizeExecution(workload_name, "mattranspose", EXECUTION_MODE);
        endTimer();

        printMetrics();
    }
};

// Factory to manage different benchmarks
class BenchmarkFactory {
public:
    static std::unique_ptr<BenchmarkBase> getBenchmark(const std::string &workload) {
        if (workload == "matmul") {
            return std::make_unique<MatMulBenchmark>();
        } else if (workload == "vecadd") {
            return std::make_unique<VecAddBenchmark>();
        }
        else if (workload == "batchnorm") {
            return std::make_unique<BatchNormBenchmark>();
        }
        else if (workload == "conv") {
            return std::make_unique<ConvBenchmark>();
        }
        else if (workload == "dotprod") {
            return std::make_unique<DotProductBenchmark>();
        }
        else if (workload == "elemadd") {
            return std::make_unique<ElemAddBenchmark>();
        }
        else if (workload == "fma") {
            return std::make_unique<FMABenchmark>();
        }
        else if (workload == "matreduction") {
            return std::make_unique<MatReduceBenchmark>();
        }
        else if (workload == "softmax") {
            return std::make_unique<SoftmaxBenchmark>();
        }
        else if (workload == "mattranspose") {
            return std::make_unique<MatTransposeBenchmark>();
        }
        else {
            std::cerr << "Unknown workload: " << workload << std::endl;
            return nullptr;
        }
    }
};

#endif // WORKLOAD_BENCHMARK_H
