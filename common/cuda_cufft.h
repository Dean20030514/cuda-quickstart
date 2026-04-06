#pragma once

#include "cuda_raii.h"

#ifdef HAVE_CUFFT
#include <cufft.h>

/**
 * @brief cuFFT 结果码转字符串 | cuFFT result code to string
 */
inline const char* cufftGetResultString(cufftResult result) {
    switch (result) {
        case CUFFT_SUCCESS:            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:       return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:       return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:       return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:      return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:     return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:        return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:       return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:       return "CUFFT_INVALID_SIZE";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        default:                       return "CUFFT_UNKNOWN";
    }
}

/**
 * @brief cuFFT 错误检查宏（exit）| cuFFT Error Checking Macro (exit)
 */
#define CUFFT_CHECK(call)                                                         \
    do {                                                                          \
        cufftResult status = (call);                                              \
        if (status != CUFFT_SUCCESS) {                                            \
            fprintf(stderr, "cuFFT error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cufftGetResultString(status));            \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief cuFFT 异常式错误检查宏 | cuFFT Exception-based Error Checking Macro
 *
 * 与 CUFFT_CHECK 类似，但抛出 std::runtime_error 而非调用 exit()，
 * 使 RAII 析构函数能正常执行。
 *
 * Similar to CUFFT_CHECK but throws std::runtime_error instead of exit(),
 * allowing RAII destructors to run properly.
 */
#define CUFFT_CHECK_THROW(call)                                                   \
    do {                                                                          \
        cufftResult status = (call);                                              \
        if (status != CUFFT_SUCCESS) {                                            \
            throw std::runtime_error(                                             \
                std::string("cuFFT error at ") + __FILE__ + ":" +                 \
                std::to_string(__LINE__) + ": " +                                 \
                cufftGetResultString(status));                                     \
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
        CUFFT_CHECK_THROW(cufftPlan1d(&h, nx, type, batch));
        handle_ = UniqueHandle<cufftHandle, 0, Deleter>(h);
    }

    void plan2d(int nx, int ny, cufftType type) {
        handle_.reset();
        cufftHandle h = 0;
        CUFFT_CHECK_THROW(cufftPlan2d(&h, nx, ny, type));
        handle_ = UniqueHandle<cufftHandle, 0, Deleter>(h);
    }

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(handle_); }
    [[nodiscard]] cufftHandle get() const noexcept { return handle_.get(); }
    explicit operator cufftHandle() const noexcept { return handle_.get(); }

private:
    UniqueHandle<cufftHandle, 0, Deleter> handle_;
};

#endif // HAVE_CUFFT
