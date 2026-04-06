#pragma once

#include "cuda_raii.h"

#ifdef HAVE_CUDNN
#include <cudnn.h>

/**
 * @brief cuDNN 错误检查宏 | cuDNN Error Checking Macro
 *
 * NOTE: exit-style only. _THROW variant not yet provided — see cuda_check.h
 * for the pattern if you need exception-based cuDNN error handling.
 */
#define CUDNN_CHECK(call)                                                         \
    do {                                                                          \
        cudnnStatus_t status = (call);                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudnnGetErrorString(status));             \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 cuDNN Handle 包装类 | RAII-style cuDNN Handle Wrapper Class
 */
class CudnnHandle {
    struct Deleter { void operator()(cudnnHandle_t h) const noexcept { cudnnDestroy(h); } };
public:
    CudnnHandle() {
        cudnnHandle_t h = nullptr;
        CUDNN_CHECK(cudnnCreate(&h));
        handle_ = UniqueHandle<cudnnHandle_t, nullptr, Deleter>(h);
    }

    [[nodiscard]] cudnnHandle_t get() const noexcept { return handle_.get(); }
    explicit operator cudnnHandle_t() const noexcept { return handle_.get(); }

private:
    UniqueHandle<cudnnHandle_t, nullptr, Deleter> handle_;
};

/**
 * @brief RAII 风格的 cuDNN TensorDescriptor 包装类 | RAII-style cuDNN TensorDescriptor Wrapper Class
 */
class CudnnTensorDescriptor {
    struct Deleter { void operator()(cudnnTensorDescriptor_t d) const noexcept { cudnnDestroyTensorDescriptor(d); } };
public:
    CudnnTensorDescriptor() {
        cudnnTensorDescriptor_t d = nullptr;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&d));
        handle_ = UniqueHandle<cudnnTensorDescriptor_t, nullptr, Deleter>(d);
    }

    void set4d(cudnnTensorFormat_t format, cudnnDataType_t dataType,
               int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(handle_.get(), format, dataType, n, c, h, w));
    }

    [[nodiscard]] cudnnTensorDescriptor_t get() const noexcept { return handle_.get(); }
    explicit operator cudnnTensorDescriptor_t() const noexcept { return handle_.get(); }

private:
    UniqueHandle<cudnnTensorDescriptor_t, nullptr, Deleter> handle_;
};

#endif // HAVE_CUDNN
