#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

/**
 * @brief CUDA 错误检查宏 | CUDA Error Checking Macro
 *
 * 用于包装 CUDA API 调用并检查返回值。如果发生错误，打印详细的
 * 错误信息（包含文件名、行号、错误代码和描述）后退出程序。
 *
 * Wraps CUDA API calls and checks return values. If an error occurs,
 * prints detailed error information (filename, line number, error code
 * and description) then exits the program.
 *
 * 用法 | Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: [%d] %s\n",                     \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 核函数启动后错误检查宏 | CUDA Kernel Launch Error Checking Macro
 *
 * 在核函数启动后使用，检查启动配置错误和异步执行错误。
 * 同时调用 cudaDeviceSynchronize 以捕获异步执行错误。
 *
 * Use after kernel launch to check for launch configuration errors
 * and asynchronous execution errors. Also calls cudaDeviceSynchronize
 * to catch async execution errors.
 *
 * 用法 | Usage:
 *   my_kernel<<<grid, block>>>(args);
 *   CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL()                                                       \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: [%d] %s\n",       \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
        err = cudaDeviceSynchronize();                                            \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA kernel exec error at %s:%d: [%d] %s\n",         \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 异常式错误检查宏 | CUDA Exception-based Error Checking Macro
 *
 * 与 CUDA_CHECK 类似，但抛出 std::runtime_error 而非调用 exit()，
 * 使 RAII 析构函数能正常执行，避免资源泄漏。适用于需要错误恢复的场景。
 *
 * Similar to CUDA_CHECK but throws std::runtime_error instead of exit(),
 * allowing RAII destructors to run properly. Use when error recovery is needed.
 *
 * 用法 | Usage: CUDA_CHECK_THROW(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK_THROW(call)                                                    \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA error at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + ": [" +                                \
                std::to_string(static_cast<int>(err)) + "] " +                    \
                cudaGetErrorString(err));                                          \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 核函数异常式错误检查宏 | CUDA Kernel Exception-based Error Checking Macro
 *
 * 与 CUDA_CHECK_KERNEL 类似，但抛出异常。
 *
 * Similar to CUDA_CHECK_KERNEL but throws exceptions.
 */
#define CUDA_CHECK_KERNEL_THROW()                                                 \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA kernel launch error at ") + __FILE__ + ":" +    \
                std::to_string(__LINE__) + ": [" +                                \
                std::to_string(static_cast<int>(err)) + "] " +                    \
                cudaGetErrorString(err));                                          \
        }                                                                         \
        err = cudaDeviceSynchronize();                                            \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA kernel exec error at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + ": [" +                                \
                std::to_string(static_cast<int>(err)) + "] " +                    \
                cudaGetErrorString(err));                                          \
        }                                                                         \
    } while (0)
