# CUDA Quickstart 项目规划 | Project Plan

## 项目目标 | Project Goal

提供开箱即用的 CUDA 开发模板，帮助开发者快速搭建 Windows + VS Code 环境下的 CUDA 项目。

Provide an out-of-the-box CUDA development template to help developers quickly set up CUDA projects in Windows + VS Code environment.

## 项目结构 | Project Structure

```
cuda-quickstart/
├── common/                    # 公共头文件 | Common headers
│   └── cuda_helper.h          # CUDA 辅助函数和宏 | CUDA helper functions and macros
├── single-nvcc/               # 方案 A：单文件 nvcc 编译 | Option A: Single-file nvcc compilation
│   ├── main.cu
│   ├── .vscode/
│   │   ├── tasks.json         # VS Code 构建任务 | VS Code build tasks
│   │   └── launch.json        # 调试配置 | Debug configuration
│   └── scripts/
│       └── build_and_run.ps1  # 自动构建脚本 | Auto build script
├── cuda-cmake/                # 方案 B：CMake 工程 | Option B: CMake project
│   ├── CMakeLists.txt
│   ├── src/main.cu
│   ├── .vscode/
│   │   ├── tasks.json
│   │   └── launch.json
│   └── scripts/
│       └── configure_build_run.ps1
└── scripts/                   # 全局环境配置脚本 | Global environment scripts
    ├── common/
    │   └── VsHelper.psm1      # VS 环境检测模块 | VS environment detection module
    └── global/
        ├── enable_cuda_env.ps1              # 临时启用 CUDA 环境 | Temporarily enable CUDA environment
        ├── install_ecuda_alias.ps1          # 安装 ecuda 快捷命令 | Install ecuda shortcut command
        ├── install_cuda_env_persistent.ps1  # 持久化安装 | Persistent installation
        └── remove_cuda_env_persistent.ps1   # 卸载 | Uninstall
```

## 功能特性 | Features

### ✅ 已实现 | Implemented

- [x] 单文件 nvcc 编译方案（single-nvcc）| Single-file nvcc compilation (single-nvcc)
- [x] CMake 标准工程方案（cuda-cmake）| CMake standard project (cuda-cmake)
- [x] 自动探测 GPU 架构（`native`）| Automatic GPU architecture detection (`native`)
- [x] 多架构 fatbin 支持（sm_75/86/89/90/100）| Multi-architecture fatbin support (sm_75/86/89/90/100)
- [x] Debug/Release 配置切换 | Debug/Release configuration switching
- [x] FastMath 优化选项 | FastMath optimization option
- [x] cuDNN 自动检测与集成 | Automatic cuDNN detection and integration
- [x] NVTX 标记支持 | NVTX marker support
- [x] VS 2022/2026 自动检测与兼容 | VS 2022/2026 auto-detection and compatibility
- [x] 全局 CUDA 环境配置脚本 | Global CUDA environment configuration scripts
- [x] `ecuda` 一键启用命令 | `ecuda` one-click enable command

### 🔧 技术要求 | Technical Requirements

| 组件 Component | 要求 Requirement |
|----------------|------------------|
| CUDA Toolkit | ≥ 12.0（推荐 Recommended 13.0）|
| CMake | ≥ 3.24（支持 Supports `native` 架构 architecture）|
| Visual Studio | 2022 Build Tools |
| Windows | 10/11 x64 |

### 📊 支持的 GPU 架构 | Supported GPU Architectures

| SM | 架构 Architecture | GPU 系列 Series |
|----|-------------------|-----------------|
| 75 | Turing | GTX 16xx, RTX 20xx |
| 86 | Ampere | RTX 30xx |
| 89 | Ada Lovelace | RTX 40xx |
| 90 | Hopper | H100 |
| 100 | Blackwell | RTX 50xx |

## 后续计划 | Future Plans

- [ ] 添加更多 CUDA 示例（矩阵乘法、归约等）| Add more CUDA examples (matrix multiplication, reduction, etc.)
- [ ] Linux/WSL 支持脚本 | Linux/WSL support scripts
- [ ] GitHub Actions CI/CD
- [ ] cuBLAS/cuFFT 集成示例 | cuBLAS/cuFFT integration examples
- [ ] Nsight 调试配置模板 | Nsight debugging configuration templates
