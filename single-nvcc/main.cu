// single-nvcc/main.cu - 使用公共头文件的简单 CUDA 示例
#include "../common/cuda_helper.h"
#include <vector>
#include <numeric>

//==============================================================================
// Kernel
//==============================================================================

__global__ void add_one(int* __restrict__ a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += 1;
    }
}

//==============================================================================
// Main
//==============================================================================

int main() {
    const int N = 1 << 16;
    
    // 使用 std::vector 代替手动 malloc
    std::vector<int> h(N);
    std::iota(h.begin(), h.end(), 0);

    // 使用 RAII 包装管理设备内存
    CudaDeviceMemory<int> d(N);
    d.copyFromHost(h.data());
    
    const int block = 256;
    const int grid = (N + block - 1) / block;
    
    // 使用 RAII 包装计时
    CudaEvent e0, e1;
    e0.record();
    add_one<<<grid, block>>>(d.get(), N);
    e1.record();
    
    // 检查 kernel 启动错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms = CudaEvent::elapsedMs(e0, e1);

    d.copyToHost(h.data());

    for (int i = 0; i < 16 && i < N; ++i) printf("%d ", h[i]);
    printf("...  (N=%d)\n", N);
    printf("kernel elapsed: %.3f ms\n", ms);
    
    return 0;
}
