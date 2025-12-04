# 🚀 CUDA Quickstart

[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Windows](https://img.shields.io/badge/Platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![VS Code](https://img.shields.io/badge/IDE-VS%20Code-007ACC.svg)](https://code.visualstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

开箱即用的 CUDA 项目模板，支持 Windows + VS Code 开发环境。

## ✨ 特性

- 🎯 **两种方案**：单文件 nvcc 编译 / CMake 标准工程
- 🔧 **自动配置**：自动探测 GPU 架构，无需手动设置
- ⚡ **一键运行**：VS Code 任务或 PowerShell 脚本
- 🧠 **cuDNN 集成**：自动检测并启用 cuDNN
- 🛠️ **VS 兼容**：支持 VS 2022/2026，自动处理兼容性

## 📋 环境要求

| 组件 | 要求 |
|------|------|
| CUDA Toolkit | ≥ 12.0 |
| CMake | ≥ 3.24 |
| Visual Studio | 2022 Build Tools |
| VS Code 扩展 | C/C++、CMake Tools |

验证环境：

```powershell
nvcc --version    # 应显示 CUDA 版本
nvidia-smi        # 应显示 GPU 信息
```

## 🚀 快速开始

### 方案 A：single-nvcc（单文件，快速上手）

```powershell
cd single-nvcc
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1
```

### 方案 B：cuda-cmake（CMake，推荐日常开发）

```powershell
cd cuda-cmake
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1
```

### VS Code 运行

1. 打开仓库目录
2. `Ctrl+Shift+P` → "Tasks: Run Task" → 选择任务

## 📁 项目结构

```
cuda-quickstart/
├── common/cuda_helper.h           # 公共 CUDA 辅助函数
├── single-nvcc/                   # 方案 A
│   ├── main.cu
│   └── scripts/build_and_run.ps1
├── cuda-cmake/                    # 方案 B
│   ├── CMakeLists.txt
│   ├── src/main.cu
│   └── scripts/configure_build_run.ps1
└── scripts/global/                # 全局环境配置
    ├── enable_cuda_env.ps1        # 临时启用
    ├── install_ecuda_alias.ps1    # 安装快捷命令
    └── install_cuda_env_persistent.ps1  # 持久化
```

## ⚙️ 构建选项

### 配置模式

```powershell
# Debug（默认，含调试信息）
-Configuration Debug

# Release（优化编译）
-Configuration Release
```

### GPU 架构

```powershell
# 自动探测（默认）
# 不指定 -CudaArch/-Sm 参数

# 指定架构
-CudaArch "100"   # cuda-cmake
-Sm 100           # single-nvcc
```

### 性能优化

```powershell
# 启用 FastMath（牺牲精度换速度）
-FastMath
```

### 完整示例

```powershell
# cuda-cmake：Release + RTX 50 + FastMath
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1 -Configuration Release -CudaArch "100" -FastMath

# single-nvcc：Release + RTX 50 + FastMath
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1 -Configuration Release -Sm 100 -FastMath
```

## 🌐 全局环境配置

### 临时启用（当前会话）

```powershell
.\scripts\global\enable_cuda_env.ps1
```

### 安装 ecuda 快捷命令

```powershell
.\scripts\global\install_ecuda_alias.ps1

# 之后新终端直接使用
ecuda
```

### 持久化安装

```powershell
# 管理员（所有用户）
.\scripts\global\install_cuda_env_persistent.ps1 -Scope Machine

# 当前用户
.\scripts\global\install_cuda_env_persistent.ps1 -Scope User
```

## 📊 GPU 架构参考

| GPU 系列 | 架构 | SM 值 | 示例型号 |
|----------|------|-------|----------|
| GTX 16 / RTX 20 | Turing | 75 | GTX 1660, RTX 2080 |
| RTX 30 | Ampere | 86 | RTX 3060, 3080, 3090 |
| RTX 40 | Ada Lovelace | 89 | RTX 4060, 4080, 4090 |
| RTX 50 | Blackwell | 100 | RTX 5070, 5080, 5090 |
| H100 | Hopper | 90 | H100 |

> 💡 使用 `nvidia-smi` 查看 GPU 型号，对照上表选择 SM 值。

## ❓ 常见问题

<details>
<summary><b>nvcc 不是内部或外部命令</b></summary>

将 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` 加入 PATH，重启终端。

</details>

<details>
<summary><b>cl.exe not found</b></summary>

安装 VS 2022 Build Tools，或使用 "Developer Command Prompt for VS 2022" 打开 VS Code。

</details>

<details>
<summary><b>算力不匹配</b></summary>

修改 `-gencode` / `CMAKE_CUDA_ARCHITECTURES` / `-Sm` 为对应的 SM 值。

</details>

<details>
<summary><b>WSL2 支持</b></summary>

1. Windows 安装支持 WSL 的 NVIDIA 驱动
2. WSL 中安装 cuda-toolkit
3. 使用 GCC/CMake 构建

</details>

## 📄 License

[MIT License](LICENSE)

## 🔗 相关链接

- [CUDA Toolkit 下载](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 下载](https://developer.nvidia.com/cudnn)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [VS Code CUDA 调试](https://developer.nvidia.com/nsight-visual-studio-code-edition)
