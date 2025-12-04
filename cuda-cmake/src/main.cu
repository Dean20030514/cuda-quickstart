// cuda-cmake/src/main.cu - 使用公共头文件的 CUDA + cuDNN 示例
// cuda-cmake/src/main.cu - CUDA + cuDNN example using common header
#include "../../common/cuda_helper.h"
#include <vector>
#include <numeric>
#ifdef HAVE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

//==============================================================================
// Kernel
//==============================================================================

__global__ void add_one(int* __restrict__ a, int n) {
    // grid-stride loop 以适配任意 N
    // grid-stride loop to adapt to any N
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += 1;
    }
}

//==============================================================================
// Main
//==============================================================================

int main() {
    const int N = 1 << 16; // 使用更大的 N 演示性能；可根据需要调整 | Use larger N for performance demo; adjust as needed

    // 使用 std::vector 代替手动 malloc，更安全也更现代
    // Use std::vector instead of manual malloc, safer and more modern
    std::vector<int> h(N);
    std::iota(h.begin(), h.end(), 0); // 填充 0, 1, 2, ... | Fill with 0, 1, 2, ...

    // 使用 RAII 包装管理设备内存，自动释放
    // Use RAII wrapper to manage device memory, auto-release
    CudaDeviceMemory<int> d(N);
    d.copyFromHost(h.data());

    // 选择合理的 launch 配置
    // Choose reasonable launch configuration
    const int block = 256;
    const int grid = (N + block - 1) / block;

#ifdef HAVE_NVTX
    nvtx3::scoped_range r1{"add_one kernel"};
#endif

    // 使用 RAII 包装计时事件，确保资源正确释放
    // Use RAII wrapper for timing events, ensure proper resource release
    CudaEvent e0, e1;
    e0.record();
    add_one<<<grid, block>>>(d.get(), N);
    e1.record();

    // 检查 kernel 启动错误
    // Check kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = CudaEvent::elapsedMs(e0, e1);

    d.copyToHost(h.data());

    // 只打印前 16 个，避免大量输出干扰性能
    // Only print first 16 to avoid excessive output affecting performance
    for (int i = 0; i < 16 && i < N; ++i) printf("%d ", h[i]);
    printf("...  (N=%d)\n", N);
    printf("kernel elapsed: %.3f ms\n", ms);

#ifdef HAVE_CUDNN
    // cuDNN v9: use minimal validation (version + handle + descriptor)
    // cuDNN v9: 使用最小验证（版本 + handle + 描述符）
    size_t ver = cudnnGetVersion();
    printf("cuDNN detected, version %zu\n", ver);

    // 使用 RAII 包装，确保资源正确释放
    // Use RAII wrapper to ensure proper resource release
    CudnnHandle handle;
    CudnnTensorDescriptor xDesc;

    // Create a simple 1x1x4x4 tensor descriptor to validate headers/ABI
    // 创建简单的 1x1x4x4 张量描述符来验证头文件/ABI
    xDesc.set4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
    printf("cuDNN tensor descriptor created successfully.\n");
    printf("cuDNN 张量描述符创建成功。\n");
#else
    printf("(cuDNN not found at configure time; skipping cuDNN demo)\n");
    printf("(配置时未找到 cuDNN；跳过 cuDNN 演示)\n");
#endif

    return 0;
}
