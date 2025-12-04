#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * @brief CUDA 错误检查宏
 * 
 * 用于包装 CUDA API 调用并检查返回值。如果发生错误，打印详细的
 * 错误信息（包含文件名、行号、错误代码和描述）后退出程序。
 * 
 * 用法：CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: [%d] %s\n",                     \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 核函数启动后错误检查宏
 * 
 * 在核函数启动后使用，检查启动配置错误和异步执行错误。
 * 
 * 用法：
 *   my_kernel<<<grid, block>>>(args);
 *   CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL()                                                       \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: [%d] %s\n",       \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 CUDA Event 包装类
 * 
 * 自动管理 cudaEvent_t 的生命周期，确保事件在离开作用域时被销毁，
 * 防止资源泄漏。
 */
class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    
    // 禁用拷贝
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // 启用移动
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(event_, stream)); }
    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }
    
    static float elapsedMs(const CudaEvent& start, const CudaEvent& end) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }
    
private:
    cudaEvent_t event_ = nullptr;
};

/**
 * @brief RAII 风格的 CUDA 设备内存包装类
 * 
 * 自动管理设备内存的分配和释放。
 * 
 * 用法：
 *   CudaDeviceMemory<float> d_data(1024);
 *   cudaMemcpy(d_data.get(), h_data, size, cudaMemcpyHostToDevice);
 */
template<typename T>
class CudaDeviceMemory {
public:
    explicit CudaDeviceMemory(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~CudaDeviceMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // 禁用拷贝
    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;
    
    // 启用移动
    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
    void copyFromHost(const T* src, size_t n = 0) {
        size_t copyCount = (n == 0) ? count_ : n;
        CUDA_CHECK(cudaMemcpy(ptr_, src, copyCount * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copyToHost(T* dst, size_t n = 0) const {
        size_t copyCount = (n == 0) ? count_ : n;
        CUDA_CHECK(cudaMemcpy(dst, ptr_, copyCount * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/**
 * @brief 简单的 grid-stride loop kernel
 * 
 * 用于处理任意大小 N 的数组，确保所有元素都被处理。
 */
__global__ inline void add_one_kernel(int* __restrict__ a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += 1;
    }
}

/**
 * @brief 计算合适的 grid 大小
 * 
 * @param n 总元素数量
 * @param blockSize 每个 block 的线程数
 * @return 需要的 grid 大小
 */
inline int calcGridSize(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

#ifdef HAVE_CUDNN
#include <cudnn.h>

/**
 * @brief cuDNN 错误检查宏
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
 * @brief RAII 风格的 cuDNN Handle 包装类
 */
class CudnnHandle {
public:
    CudnnHandle() { CUDNN_CHECK(cudnnCreate(&handle_)); }
    ~CudnnHandle() { if (handle_) cudnnDestroy(handle_); }
    
    CudnnHandle(const CudnnHandle&) = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;
    
    cudnnHandle_t get() const { return handle_; }
    operator cudnnHandle_t() const { return handle_; }
    
private:
    cudnnHandle_t handle_ = nullptr;
};

/**
 * @brief RAII 风格的 cuDNN TensorDescriptor 包装类
 */
class CudnnTensorDescriptor {
public:
    CudnnTensorDescriptor() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_)); }
    ~CudnnTensorDescriptor() { if (desc_) cudnnDestroyTensorDescriptor(desc_); }
    
    CudnnTensorDescriptor(const CudnnTensorDescriptor&) = delete;
    CudnnTensorDescriptor& operator=(const CudnnTensorDescriptor&) = delete;
    
    void set4d(cudnnTensorFormat_t format, cudnnDataType_t dataType,
               int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_, format, dataType, n, c, h, w));
    }
    
    cudnnTensorDescriptor_t get() const { return desc_; }
    operator cudnnTensorDescriptor_t() const { return desc_; }
    
private:
    cudnnTensorDescriptor_t desc_ = nullptr;
};

#endif // HAVE_CUDNN

#endif // CUDA_HELPER_H
