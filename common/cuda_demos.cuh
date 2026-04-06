#pragma once

// NOTE: This header uses the <<<>>> kernel launch syntax on a template parameter.
// This is only valid when compiled with nvcc (not host-only compilers).
// All build paths in this project use nvcc, so this is safe.

#include "cuda_raii.h"
#include "cuda_utils.h"
#include <vector>
#include <numeric>
#include <cstdio>

// ============================================================================
// demoKernel — launch a kernel, time it, and verify add-one results
// ============================================================================

/**
 * @brief 启动 kernel 并验证 add-one 结果 | Launch kernel, time it, verify add-one results
 *
 * @tparam KernelFunc __global__ 函数类型 | __global__ function type
 * @param kernel 要启动的 kernel | The kernel to launch
 * @param N 元素数量 | Number of elements
 * @return true if verification passed
 */
template<typename KernelFunc>
inline bool demoKernel(KernelFunc kernel, int N = 1 << 20) {
    printf("\n--- Kernel Demo ---\n");

    std::vector<int> h(N);
    std::iota(h.begin(), h.end(), 0);

    CudaDeviceMemory<int> d(N);
    d.copyFromHost(h.data());

    const int block = 256;
    const int grid = calcGridSize(N, block);

    CudaEvent e0, e1;
    e0.record();
    kernel<<<grid, block>>>(d.get(), N);
    e1.record();
    CUDA_CHECK_KERNEL_THROW();

    float ms = elapsedMs(e0, e1);
    d.copyToHost(h.data());

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h[i] != i + 1) { ok = false; break; }
    }

    for (int i = 0; i < 16 && i < N; ++i) printf("%d ", h[i]);
    printf("...  (N=%d)\n", N);
    printf("  kernel elapsed: %.3f ms\n", ms);
    printf("  verification:   %s\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// demoAsyncStream — async pinned-memory round-trip
// ============================================================================

/**
 * @brief 异步 Stream 拷贝演示 | Async stream copy demo
 */
inline bool demoAsyncStream() {
    printf("\n--- Async Stream Demo ---\n");
    const int N = 1 << 18;

    CudaStream stream(cudaStreamNonBlocking);
    CudaPinnedMemory<float> h_in(N), h_out(N);
    CudaDeviceMemory<float> d_buf(N);

    for (size_t i = 0; i < static_cast<size_t>(N); ++i) h_in[i] = static_cast<float>(i);

    CudaEvent s0, s1;
    s0.record(stream.get());
    d_buf.copyFromHostAsync(h_in.get(), stream.get());
    d_buf.copyToHostAsync(h_out.get(), stream.get());
    s1.record(stream.get());
    stream.synchronize();

    float ms = elapsedMs(s0, s1);
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != static_cast<float>(i)) { ok = false; break; }
    }
    double sizeMB = static_cast<double>(N) * sizeof(float) / (1024.0 * 1024.0);
    printf("  Async round-trip %.1f MB: %.3f ms -> %s\n", sizeMB, ms, ok ? "PASSED" : "FAILED");
    return ok;
}
