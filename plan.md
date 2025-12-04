# CUDA Quickstart 项目规划

## 项目目标

提供开箱即用的 CUDA 开发模板，帮助开发者快速搭建 Windows + VS Code 环境下的 CUDA 项目。

## 项目结构

```
cuda-quickstart/
├── common/                    # 公共头文件
│   └── cuda_helper.h          # CUDA 辅助函数和宏
├── single-nvcc/               # 方案 A：单文件 nvcc 编译
│   ├── main.cu
│   ├── .vscode/
│   │   ├── tasks.json         # VS Code 构建任务
│   │   └── launch.json        # 调试配置
│   └── scripts/
│       └── build_and_run.ps1  # 自动构建脚本
├── cuda-cmake/                # 方案 B：CMake 工程
│   ├── CMakeLists.txt
│   ├── src/main.cu
│   ├── .vscode/
│   │   ├── tasks.json
│   │   └── launch.json
│   └── scripts/
│       └── configure_build_run.ps1
└── scripts/                   # 全局环境配置脚本
    ├── common/
    │   └── VsHelper.psm1      # VS 环境检测模块
    └── global/
        ├── enable_cuda_env.ps1              # 临时启用 CUDA 环境
        ├── install_ecuda_alias.ps1          # 安装 ecuda 快捷命令
        ├── install_cuda_env_persistent.ps1  # 持久化安装
        └── remove_cuda_env_persistent.ps1   # 卸载
```

## 功能特性

### ✅ 已实现

- [x] 单文件 nvcc 编译方案（single-nvcc）
- [x] CMake 标准工程方案（cuda-cmake）
- [x] 自动探测 GPU 架构（`native`）
- [x] 多架构 fatbin 支持（sm_75/86/89/90/100）
- [x] Debug/Release 配置切换
- [x] FastMath 优化选项
- [x] cuDNN 自动检测与集成
- [x] NVTX 标记支持
- [x] VS 2022/2026 自动检测与兼容
- [x] 全局 CUDA 环境配置脚本
- [x] `ecuda` 一键启用命令

### 🔧 技术要求

| 组件 | 要求 |
|------|------|
| CUDA Toolkit | ≥ 12.0（推荐 13.0）|
| CMake | ≥ 3.24（支持 `native` 架构）|
| Visual Studio | 2022 Build Tools |
| Windows | 10/11 x64 |

### 📊 支持的 GPU 架构

| SM | 架构 | GPU 系列 |
|----|------|----------|
| 75 | Turing | GTX 16xx, RTX 20xx |
| 86 | Ampere | RTX 30xx |
| 89 | Ada Lovelace | RTX 40xx |
| 90 | Hopper | H100 |
| 100 | Blackwell | RTX 50xx |

## 后续计划

- [ ] 添加更多 CUDA 示例（矩阵乘法、归约等）
- [ ] Linux/WSL 支持脚本
- [ ] GitHub Actions CI/CD
- [ ] cuBLAS/cuFFT 集成示例
- [ ] Nsight 调试配置模板
