#pragma once

#include "cuda_raii.h"

#ifdef HAVE_CUBLAS
#include <cublas_v2.h>

/**
 * @brief cuBLAS 状态码转字符串 | cuBLAS status code to string
 */
inline const char* cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

/**
 * @brief cuBLAS 错误检查宏（exit）| cuBLAS Error Checking Macro (exit)
 */
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cublasGetStatusString(status));           \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief cuBLAS 异常式错误检查宏 | cuBLAS Exception-based Error Checking Macro
 *
 * 与 CUBLAS_CHECK 类似，但抛出 std::runtime_error 而非调用 exit()，
 * 使 RAII 析构函数能正常执行。
 *
 * Similar to CUBLAS_CHECK but throws std::runtime_error instead of exit(),
 * allowing RAII destructors to run properly.
 */
#define CUBLAS_CHECK_THROW(call)                                                  \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(                                             \
                std::string("cuBLAS error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + ": " +                                 \
                cublasGetStatusString(status));                                    \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 cuBLAS Handle 包装类 | RAII-style cuBLAS Handle Wrapper Class
 */
class CublasHandle {
    struct Deleter { void operator()(cublasHandle_t h) const noexcept { cublasDestroy(h); } };
public:
    CublasHandle() {
        cublasHandle_t h = nullptr;
        CUBLAS_CHECK_THROW(cublasCreate(&h));
        handle_ = UniqueHandle<cublasHandle_t, nullptr, Deleter>(h);
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_.get(); }
    explicit operator cublasHandle_t() const noexcept { return handle_.get(); }

private:
    UniqueHandle<cublasHandle_t, nullptr, Deleter> handle_;
};

#endif // HAVE_CUBLAS
