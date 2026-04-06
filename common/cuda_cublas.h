#pragma once

#include "cuda_raii.h"

#ifdef HAVE_CUBLAS
#include <cublas_v2.h>

/**
 * @brief cuBLAS 错误检查宏 | cuBLAS Error Checking Macro
 *
 * NOTE: exit-style only. _THROW variant not yet provided — see cuda_check.h
 * for the pattern if you need exception-based cuBLAS error handling.
 */
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",                  \
                    __FILE__, __LINE__, static_cast<int>(status));                \
            exit(EXIT_FAILURE);                                                   \
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
        CUBLAS_CHECK(cublasCreate(&h));
        handle_ = UniqueHandle<cublasHandle_t, nullptr, Deleter>(h);
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_.get(); }
    explicit operator cublasHandle_t() const noexcept { return handle_.get(); }

private:
    UniqueHandle<cublasHandle_t, nullptr, Deleter> handle_;
};

#endif // HAVE_CUBLAS
