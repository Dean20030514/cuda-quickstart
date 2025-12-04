# 🚀 CUDA Quickstart

[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Windows](https://img.shields.io/badge/Platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![VS Code](https://img.shields.io/badge/IDE-VS%20Code-007ACC.svg)](https://code.visualstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

开箱即用的 CUDA 项目模板，支持 Windows + VS Code 开发环境。

An out-of-the-box CUDA project template supporting Windows + VS Code development environment.

## ✨ 特性 | Features

- 🎯 **两种方案 | Two Options**：单文件 nvcc 编译 / CMake 标准工程 | Single-file nvcc compilation / CMake standard project
- 🔧 **自动配置 | Auto Configuration**：自动探测 GPU 架构，无需手动设置 | Automatic GPU architecture detection, no manual setup required
- ⚡ **一键运行 | One-Click Run**：VS Code 任务或 PowerShell 脚本 | VS Code tasks or PowerShell scripts
- 🧠 **cuDNN 集成 | cuDNN Integration**：自动检测并启用 cuDNN | Automatic cuDNN detection and enabling
- 🛠️ **VS 兼容 | VS Compatible**：支持 VS 2022/2026，自动处理兼容性 | Supports VS 2022/2026 with automatic compatibility handling

## 📋 环境要求 | Requirements

| 组件 Component | 要求 Requirement |
|----------------|------------------|
| CUDA Toolkit | ≥ 12.0 |
| CMake | ≥ 3.24 |
| Visual Studio | 2022 Build Tools |
| VS Code 扩展 Extensions | C/C++、CMake Tools |

验证环境 | Verify environment:

```powershell
nvcc --version    # 应显示 CUDA 版本 | Should display CUDA version
nvidia-smi        # 应显示 GPU 信息 | Should display GPU info
```

## 🚀 快速开始 | Quick Start

### 方案 A | Option A：single-nvcc（单文件，快速上手 | Single file, quick start）

```powershell
cd single-nvcc
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1
```

### 方案 B | Option B：cuda-cmake（CMake，推荐日常开发 | CMake, recommended for daily development）

```powershell
cd cuda-cmake
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1
```

### VS Code 运行 | Running in VS Code

1. 打开仓库目录 | Open the repository directory
2. `Ctrl+Shift+P` → "Tasks: Run Task" → 选择任务 | Select a task

## 📁 项目结构 | Project Structure

```
cuda-quickstart/
├── common/cuda_helper.h           # 公共 CUDA 辅助函数 | Common CUDA helper functions
├── single-nvcc/                   # 方案 A | Option A
│   ├── main.cu
│   └── scripts/build_and_run.ps1
├── cuda-cmake/                    # 方案 B | Option B
│   ├── CMakeLists.txt
│   ├── src/main.cu
│   └── scripts/configure_build_run.ps1
└── scripts/global/                # 全局环境配置 | Global environment configuration
    ├── enable_cuda_env.ps1        # 临时启用 | Temporary enable
    ├── install_ecuda_alias.ps1    # 安装快捷命令 | Install shortcut command
    └── install_cuda_env_persistent.ps1  # 持久化 | Persistent installation
```

## ⚙️ 构建选项 | Build Options

### 配置模式 | Configuration Mode

```powershell
# Debug（默认，含调试信息）| Debug (default, with debug info)
-Configuration Debug

# Release（优化编译）| Release (optimized compilation)
-Configuration Release
```

### GPU 架构 | GPU Architecture

```powershell
# 自动探测（默认）| Auto detection (default)
# 不指定 -CudaArch/-Sm 参数 | Do not specify -CudaArch/-Sm parameter

# 指定架构 | Specify architecture
-CudaArch "100"   # cuda-cmake
-Sm 100           # single-nvcc
```

### 性能优化 | Performance Optimization

```powershell
# 启用 FastMath（牺牲精度换速度）| Enable FastMath (trade precision for speed)
-FastMath
```

### 完整示例 | Complete Examples

```powershell
# cuda-cmake：Release + RTX 50 + FastMath
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1 -Configuration Release -CudaArch "100" -FastMath

# single-nvcc：Release + RTX 50 + FastMath
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1 -Configuration Release -Sm 100 -FastMath
```

## 🌐 全局环境配置 | Global Environment Configuration

### 临时启用（当前会话）| Temporary Enable (Current Session)

```powershell
.\scripts\global\enable_cuda_env.ps1
```

### 安装 ecuda 快捷命令 | Install ecuda Shortcut Command

```powershell
.\scripts\global\install_ecuda_alias.ps1

# 之后新终端直接使用 | Then use directly in new terminal
ecuda
```

### 持久化安装 | Persistent Installation

```powershell
# 管理员（所有用户）| Administrator (all users)
.\scripts\global\install_cuda_env_persistent.ps1 -Scope Machine

# 当前用户 | Current user
.\scripts\global\install_cuda_env_persistent.ps1 -Scope User
```

## 📊 GPU 架构参考 | GPU Architecture Reference

| GPU 系列 Series | 架构 Architecture | SM 值 Value | 示例型号 Example Models |
|-----------------|-------------------|-------------|-------------------------|
| GTX 16 / RTX 20 | Turing | 75 | GTX 1660, RTX 2080 |
| RTX 30 | Ampere | 86 | RTX 3060, 3080, 3090 |
| RTX 40 | Ada Lovelace | 89 | RTX 4060, 4080, 4090 |
| RTX 50 | Blackwell | 100 | RTX 5070, 5080, 5090 |
| H100 | Hopper | 90 | H100 |

> 💡 使用 `nvidia-smi` 查看 GPU 型号，对照上表选择 SM 值。
>
> 💡 Use `nvidia-smi` to check GPU model and select SM value from the table above.

## ❓ 常见问题 | FAQ

<details>
<summary><b>nvcc 不是内部或外部命令 | nvcc is not recognized as an internal or external command</b></summary>

将 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` 加入 PATH，重启终端。

Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` to PATH and restart the terminal.

</details>

<details>
<summary><b>cl.exe not found</b></summary>

安装 VS 2022 Build Tools，或使用 "Developer Command Prompt for VS 2022" 打开 VS Code。

Install VS 2022 Build Tools, or use "Developer Command Prompt for VS 2022" to open VS Code.

</details>

<details>
<summary><b>算力不匹配 | Compute capability mismatch</b></summary>

修改 `-gencode` / `CMAKE_CUDA_ARCHITECTURES` / `-Sm` 为对应的 SM 值。

Modify `-gencode` / `CMAKE_CUDA_ARCHITECTURES` / `-Sm` to the corresponding SM value.

</details>

<details>
<summary><b>WSL2 支持 | WSL2 Support</b></summary>

1. Windows 安装支持 WSL 的 NVIDIA 驱动 | Install WSL-compatible NVIDIA driver on Windows
2. WSL 中安装 cuda-toolkit | Install cuda-toolkit in WSL
3. 使用 GCC/CMake 构建 | Build with GCC/CMake

</details>

## 📄 License

[MIT License](LICENSE)

## 🔗 相关链接 | Related Links

- [CUDA Toolkit 下载 | Download](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 下载 | Download](https://developer.nvidia.com/cudnn)
- [CUDA 编程指南 | Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [VS Code CUDA 调试 | Debugging](https://developer.nvidia.com/nsight-visual-studio-code-edition)
