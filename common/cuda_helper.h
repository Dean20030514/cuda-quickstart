#pragma once

// cuda_helper.h — umbrella include for backward compatibility
//
// 向后兼容的统一头文件。新代码建议直接包含所需的子头文件：
//   cuda_check.h  — 错误检查宏 | Error checking macros
//   cuda_raii.h   — RAII 类 | RAII wrappers (CudaStream, CudaEvent, CudaDeviceMemory, ...)
//   cuda_utils.h  — 工具函数 | Utilities (calcGridSize, printDeviceInfo, measureBandwidth, elapsedMs)
//   cuda_cudnn.h  — cuDNN 包装 | cuDNN wrappers (requires HAVE_CUDNN)
//   cuda_cublas.h — cuBLAS 包装 | cuBLAS wrappers (requires HAVE_CUBLAS)
//   cuda_cufft.h  — cuFFT 包装 | cuFFT wrappers (requires HAVE_CUFFT)
//
// Umbrella header for backward compatibility. New code should include
// the specific sub-headers listed above.

#include "cuda_check.h"
#include "cuda_raii.h"
#include "cuda_utils.h"
#include "cuda_cudnn.h"
#include "cuda_cublas.h"
#include "cuda_cufft.h"
