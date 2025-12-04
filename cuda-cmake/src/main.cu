// cuda-cmake/src/main.cu - 使用公共头文件的 CUDA + cuDNN 示例
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += 1;
    }
}

//==============================================================================
// Main
//==============================================================================

int main() {
    const int N = 1 << 16; // 使用更大的 N 演示性能；可根据需要调整
    
    // 使用 std::vector 代替手动 malloc，更安全也更现代
    std::vector<int> h(N);
    std::iota(h.begin(), h.end(), 0); // 填充 0, 1, 2, ...

    // 使用 RAII 包装管理设备内存，自动释放
    CudaDeviceMemory<int> d(N);
    d.copyFromHost(h.data());

    // 选择合理的 launch 配置
    const int block = 256;
    const int grid = (N + block - 1) / block;

#ifdef HAVE_NVTX
    nvtx3::scoped_range r1{"add_one kernel"};
#endif

    // 使用 RAII 包装计时事件，确保资源正确释放
    CudaEvent e0, e1;
    e0.record();
    add_one<<<grid, block>>>(d.get(), N);
    e1.record();
    
    // 检查 kernel 启动错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms = CudaEvent::elapsedMs(e0, e1);

    d.copyToHost(h.data());

    // 只打印前 16 个，避免大量输出干扰性能
    for (int i = 0; i < 16 && i < N; ++i) printf("%d ", h[i]);
    printf("...  (N=%d)\n", N);
    printf("kernel elapsed: %.3f ms\n", ms);

#ifdef HAVE_CUDNN
    // cuDNN v9: use minimal validation (version + handle + descriptor)
    size_t ver = cudnnGetVersion();
    printf("cuDNN detected, version %zu\n", ver);

    // 使用 RAII 包装，确保资源正确释放
    CudnnHandle handle;
    CudnnTensorDescriptor xDesc;
    
    // Create a simple 1x1x4x4 tensor descriptor to validate headers/ABI
    xDesc.set4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
    printf("cuDNN tensor descriptor created successfully.\n");
#else
    printf("(cuDNN not found at configure time; skipping cuDNN demo)\n");
#endif

    return 0;
}
