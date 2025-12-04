下面给你一套“开箱即用”的 VS Code + CUDA 项目创建方法（Windows 为主，Linux/WSL 同理）。按你熟悉程度选 A（超快上手，单文件）或 B（标准工程，CMake）。

---

# 0) 先装好这些（一次性）

1. **NVIDIA 驱动**（显卡正常工作）。
2. **CUDA Toolkit**（安装后应有 `nvcc`）。
3. **Windows 专用**：安装 **Microsoft C++ Build Tools / VS 2022（含 MSVC）**，否则 `nvcc` 找不到 `cl.exe`。
4. **VS Code 扩展**：

   * *C/C++*（ms-vscode.cpptools）
   * *CMake Tools*（ms-vscode.cmake-tools）—如果你选方案 B
   * *NVIDIA Nsight VS Code Edition*（调试 GPU 内核可用，选装）

> 验证：打开终端输入 `nvcc --version` 与 `nvidia-smi`。能正常输出版本/显卡信息即 OK。

---

# A) 超快上手（单文件 + tasks.json，用 `nvcc` 直接编译）

适合想先跑通一个最小 CUDA 内核的场景。

**目录结构**

```
cuda-quickstart/
  .vscode/
    tasks.json
  main.cu
```

**main.cu**

```cpp
#include <cstdio>

__global__ void add_one(int *a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] += 1;
}

int main() {
    const int N = 16;
    int h[N];
    for (int i = 0; i < N; ++i) h[i] = i;

    int *d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice);

    add_one<<<1, N>>>(d);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    for (int i = 0; i < N; ++i) printf("%d ", h[i]);
    printf("\n");
    return 0;
}
```

**.vscode/tasks.json**（Windows，MSVC 主机编译器）

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build with NVCC (Debug)",
      "type": "shell",
      "command": "nvcc",
      "args": [
        "-std=c++17",
        "-g", "-G", "-lineinfo",
        "-arch=sm_86",                  // 根据显卡改：20系=sm_75，30系=sm_86，40系(Ada)=sm_89
        "main.cu",
        "-o", "build\\main.exe"
      ],
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "Run",
      "type": "shell",
      "command": ".\\build\\main.exe",
      "dependsOn": "Build with NVCC (Debug)"
    }
  ]
}
```

> 运行：`Ctrl+Shift+P` → “Tasks: Run Task” → 选 **Run**。
> 常见坑：
>
> * 找不到 `cl.exe`：安装 *VS 2022 Build Tools*，或用 “Developer Command Prompt for VS” 打开 VS Code。
> * `-arch` 要匹配你的 GPU 架构（例：RTX 3060 用 `sm_86`）。

---

# B) 标准工程（CMake + CUDA，推荐日常开发）

更易维护、跨平台，VS Code 的 CMake Tools 一键配置/构建。

**目录结构**

```
cuda-cmake/
  CMakeLists.txt
  src/
    main.cu
```

**CMakeLists.txt**（要求 CMake ≥ 3.24 可用 `CUDA_ARCHITECTURES` 的 `native`）

```cmake
cmake_minimum_required(VERSION 3.24)
project(CudaDemo LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 自动使用本机 GPU 架构；若报不支持，可改成具体数值，如 75;86;89
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(cudatest src/main.cu)
target_link_libraries(cudatest PRIVATE CUDA::cudart)

# Debug 体验更好（行号、设备调试信息）
target_compile_options(cudatest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>)
```

**src/main.cu**（同上示例或你自己的代码）

**在 VS Code 中构建**

1. 打开文件夹 → 左下角 **CMake** 状态栏选择编译器套件（Windows 选 *Visual Studio 2022 Release - amd64* 或 *Clang-cl*）。
2. 点 **Configure** → **Build**。生成的可执行文件在 `build/`（或 `out/build/...`）目录。
3. 运行：`CMake: Run` 或在终端执行生成的可执行文件。

> 如果 `native` 架构不被识别，可改：
> `set(CMAKE_CUDA_ARCHITECTURES 75 86 89)`（根据你/同学的显卡列出多个）。

---

##（可选）GPU 调试

* 安装 **NVIDIA Nsight VS Code Edition** 扩展。
* 用它的调试配置启动，可在 `__global__` 内核里打断点、单步。
* 纯 VS Code 的 `cppdbg` 不能直接调试 GPU 线程，CPU 端可以。

---

## 常见问题速查

* **`nvcc` 找不到 / 不是内部命令**：把 `CUDA\\vX.Y\\bin` 加入 PATH，重开终端。
* **`cl.exe` not found**：安装 *MSVC*；或用 “Developer Command Prompt for VS 2022” 打开当前工程再编译。
* **算力不匹配报错**：`-arch=sm_XY` 改成与你显卡一致（20 系=75，30 系=86，40 系=89；老 10 系多为 61）。
* **MinGW 不支持**：Windows 下 `nvcc` 默认走 MSVC；不要用 MinGW 作为 host 编译器。
* **WSL2**：需要 Windows 上安装支持 WSL 的 NVIDIA 驱动；WSL 里安装 `cuda-toolkit` 后用 GCC/CMake 构建，VS Code 远程连接 WSL 即可。

---

需要我把上述示例打包成可直接打开的最小模板（含 CMake 与 tasks.json 两版）吗？我可以按你的显卡型号，把 `-arch` 和 `CUDA_ARCHITECTURES` 顺便配好。
