#pragma once

#include "cuda_raii.h"

#ifdef HAVE_CUFFT
#include <cufft.h>

/**
 * @brief cuFFT 错误检查宏 | cuFFT Error Checking Macro
 *
 * NOTE: exit-style only. _THROW variant not yet provided — see cuda_check.h
 * for the pattern if you need exception-based cuFFT error handling.
 */
#define CUFFT_CHECK(call)                                                         \
    do {                                                                          \
        cufftResult status = (call);                                              \
        if (status != CUFFT_SUCCESS) {                                            \
            fprintf(stderr, "cuFFT error at %s:%d: status=%d\n",                   \
                    __FILE__, __LINE__, static_cast<int>(status));                \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 cuFFT Plan 包装类 | RAII-style cuFFT Plan Wrapper Class
 */
class CufftPlan {
    struct Deleter { void operator()(cufftHandle h) const noexcept { cufftDestroy(h); } };
public:
    CufftPlan() = default;

    void plan1d(int nx, cufftType type, int batch = 1) {
        handle_.reset();
        cufftHandle h = 0;
        CUFFT_CHECK(cufftPlan1d(&h, nx, type, batch));
        handle_ = UniqueHandle<cufftHandle, 0, Deleter>(h);
    }

    void plan2d(int nx, int ny, cufftType type) {
        handle_.reset();
        cufftHandle h = 0;
        CUFFT_CHECK(cufftPlan2d(&h, nx, ny, type));
        handle_ = UniqueHandle<cufftHandle, 0, Deleter>(h);
    }

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(handle_); }
    [[nodiscard]] cufftHandle get() const noexcept { return handle_.get(); }
    explicit operator cufftHandle() const noexcept { return handle_.get(); }

private:
    UniqueHandle<cufftHandle, 0, Deleter> handle_;
};

#endif // HAVE_CUFFT
