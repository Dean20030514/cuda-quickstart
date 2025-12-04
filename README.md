# CUDA Quickstart (VS Code + Windows)

本仓库包含两种最小可运行的 CUDA 项目模板（已提供“自动/可配置”的算力架构设置）：

- single-nvcc：单文件 + VS Code 任务，直接用 nvcc 编译运行。
- cuda-cmake：标准 CMake 工程，适合日常开发与跨平台。

请先安装并验证以下环境：

- NVIDIA 驱动、CUDA Toolkit（终端能运行 `nvcc --version`）
- Windows: 安装 Visual Studio 2022 Build Tools（提供 MSVC `cl.exe`）。
- VS Code 扩展：C/C++、CMake Tools（用于 CMake 方案）。

---

## 1) single-nvcc 方案

结构：

```text
single-nvcc/
  .vscode/tasks.json
  main.cu
```

运行：

1. VS Code 打开此仓库。
2. 按 Ctrl+Shift+P → “Tasks: Run Task” → 选 Run。
   - 会先用 nvcc 编译，随后运行 `build/main.exe`，你应看到 `1 2 3 ...` 的输出。

说明：

- 你可以直接运行“Run (Release, fatbin+fast-math)”任务：脚本会生成覆盖面广的 fatbin（sm_90/89/86/75 + 一份 PTX），开箱即用，适合不确定 GPU 型号时的演示；若你已知自己的 GPU 算力（如 89 对应 Ada/Lovelace），可用任务“Run (Release, sm=89)”生成更小的二进制并获得最佳性能。
- 常见问题：找不到 `cl.exe` → 安装 VS 2022 Build Tools，或使用 “Developer Command Prompt for VS 2022” 打开 VS Code。

---

## 2) cuda-cmake 方案

结构：

```text
cuda-cmake/
  CMakeLists.txt
  src/main.cu
```

构建与运行（推荐用 VS Code CMake Tools）：

1. 打开仓库后，状态栏选择编译器套件（Windows 选 Visual Studio 2022 - amd64）。
2. 点击 Configure → Build。
3. 生成的可执行文件位于构建目录（如 `build/` 或 `out/build/...`）。你可以用 CMake Tools 的 Run 运行，也可在终端执行。

说明：

- `CMakeLists.txt` 使用 `CMAKE_CUDA_ARCHITECTURES native`（需要 CMake ≥ 3.24），可自动探测本机 GPU 架构；Debug 模式启用 `-G -lineinfo`（便于调试），Release/RelWithDebInfo 默认开启 `-O3 --use_fast_math -Xptxas -O3` 以获取更优性能（可用 `-D CUDA_ENABLE_FAST_MATH=OFF` 关闭）。
- 链接 `CUDA::cudart`，若可用还会链接 `CUDA::nvtx3` 以便在 Nsight 中看到 NVTX 标记。
- 若系统已安装 cuDNN 并能被发现，则自动启用最小 cuDNN 校验。否则仅运行普通 CUDA kernel（会在终端提示跳过 cuDNN 示例）。

---

## 验证运行（PowerShell 可选命令）

你也可以在 PowerShell 中手动执行（确保环境变量已配置）：

```powershell
# 方式 A：nvcc（按需指定架构或使用 fatbin）
cd single-nvcc
nvcc -std=c++17 -g -G -lineinfo `
  -gencode=arch=compute_89,code=sm_89 ` # 替换为你的 SM，如 86/75
  -gencode=arch=compute_89,code=compute_89 ` # 可选：附带 PTX 便于前向兼容
  main.cu -o build\main.exe
.\build\main.exe

# 方式 B：CMake（示例：Ninja 或 VS 生成器均可）
cd ..\cuda-cmake
cmake -S . -B build -G "Ninja"
cmake --build build --config Debug
.\build\cudatest.exe
```

---

## 全局启用 CUDA 环境（任意工程，含 cuDNN）

如果你在其它项目文件夹里无法直接用 `nvcc`/`cl.exe`，有两种方式：

方式 A（临时，仅当前终端会话生效）：

```powershell
# 在任意工程目录中执行，启用后当次会话即可使用 nvcc 和 cl.exe
# 请将 <仓库根目录> 替换为实际路径
powershell -NoProfile -ExecutionPolicy Bypass -File <仓库根目录>\scripts\global\enable_cuda_env.ps1

# 验证
nvcc --version
cl.exe /?
```

说明：

- 脚本会优先选择 VS 2022 的开发者环境（Host x64），找不到时回退到 Insiders。
- 自动设置 `CUDA_PATH`，并把 CUDA 的 `bin` 加入当前会话 PATH。
- 自动设置 `CUDNN_ROOT = CUDA_PATH`，并把 CUDA 工具包目录下的 cuDNN `bin`（包含 `bin\13.0` / `bin\12.9` 等）加入 PATH，从而“总是优先用工具包里的 cuDNN”。
- 打开新的终端窗口后，如需再次使用 CUDA/cuDNN，请重新执行一次上述脚本，或改用 “Developer PowerShell for VS 2022” 作为默认终端。
- VS Code 建议：在状态栏用 CMake Tools 选择 “Visual Studio 2022 - amd64”，或把集成终端切到 “Developer PowerShell for VS 2022”。

想“一句命令启用”（ecuda）：可先安装快捷函数到 PowerShell 配置文件（仅需一次），之后新开终端直接输入 ecuda：

```powershell
# 安装 ecuda 快捷函数到 PowerShell Profile（当前用户）
# 请将 <仓库根目录> 替换为实际路径
powershell -NoProfile -ExecutionPolicy Bypass -File <仓库根目录>\scripts\global\install_ecuda_alias.ps1

# 打开新的 PowerShell 窗口：
ecuda             # 启用 CUDA+cuDNN 环境（可加 -Quiet -CudaPath -CuDnnRoot -VsArch）
```

方式 B（持久化，全局，对新开的所有终端生效）：

需要管理员权限（推荐 Machine 级）：

```powershell
# 以管理员身份打开 PowerShell
# 请将 <仓库根目录> 替换为实际路径
powershell -NoProfile -ExecutionPolicy Bypass -File <仓库根目录>\scripts\global\install_cuda_env_persistent.ps1 -Scope Machine
```

非管理员（仅当前用户）：

```powershell
# 请将 <仓库根目录> 替换为实际路径
powershell -NoProfile -ExecutionPolicy Bypass -File <仓库根目录>\scripts\global\install_cuda_env_persistent.ps1 -Scope User
```

默认行为：

- 自动探测并设置 `CUDA_PATH`（取最高版本的 CUDA 工具包目录）。
- 设置 `CUDNN_ROOT = CUDA_PATH`，优先使用 CUDA 工具包内置的 cuDNN（CUDA 12.4+ 通常自带）。
- 将 cuDNN 的版本化 bin 目录（如 `bin\13.0`、`bin\12.9`）以及 `bin` 目录加入系统 PATH（若尚未存在）。
- 广播环境变量更新；新开的终端会看到变更（已开启的终端需重开）。

卸载/回滚（可选，管理员执行 Machine 级）：

```powershell
# 请将 <仓库根目录> 替换为实际路径
powershell -NoProfile -ExecutionPolicy Bypass -File <仓库根目录>\scripts\global\remove_cuda_env_persistent.ps1 -Scope Machine -AlsoRemoveCudaPath -AlsoRemoveCuDnnRoot
```

注意：MSVC 的 INCLUDE/LIB 等编译环境不会被持久化，仍建议在构建时使用 VS 开发者命令行或在脚本里调用 VsDevCmd（本仓库的构建脚本已自动处理）。

---

## GPU Compute Capability 参考表

选择正确的 SM（Streaming Multiprocessor）架构对于 CUDA 程序的编译至关重要。以下是常用 NVIDIA GPU 的架构对照表：

| GPU 系列 | 架构代号 | SM 值 | 示例型号 |
|----------|----------|-------|----------|
| GeForce GTX 16 系列 | Turing | 75 | GTX 1650, 1660 |
| GeForce RTX 20 系列 | Turing | 75 | RTX 2060, 2070, 2080 |
| GeForce RTX 30 系列 | Ampere | 86 | RTX 3060, 3070, 3080, 3090 |
| GeForce RTX 40 系列 | Ada Lovelace | 89 | RTX 4060, 4070, 4080, 4090 |
| GeForce RTX 50 系列 | Blackwell | 100 | RTX 5070, 5080, 5090 |
| Tesla/Datacenter V100 | Volta | 70 | V100 |
| Tesla/Datacenter A100 | Ampere | 80 | A100 |
| Tesla/Datacenter H100 | Hopper | 90 | H100 |

> **提示**：使用 `nvidia-smi` 可查看你的 GPU 型号，然后对照上表选择对应的 SM 值。
> 若不确定，可使用 fatbin 模式（`-gencode` 多架构）或 `CMAKE_CUDA_ARCHITECTURES native` 自动探测。

---

## 常见问题速查

- `nvcc` 不是内部或外部命令：把 CUDA\vX.Y\bin 加入 PATH，重启终端。
- `cl.exe` not found：安装/修复 VS 2022 Build Tools（含 MSVC）；或用 VS 开发者命令提示符启动 VS Code。
- 算力不匹配：对 RTX 5070 应使用 100。如需更换其它 GPU，请把 `-gencode` / `CMAKE_CUDA_ARCHITECTURES` 调整为对应 SM。
- WSL2：请安装支持 WSL 的 NVIDIA 驱动，并在 WSL 里安装 cuda-toolkit，使用 GCC/CMake 构建。
- Visual Studio 2026 Insiders：CUDA 13 尚未标注支持 VS 2026。仓库脚本会优先选择 VS 2022 环境；若仅有 VS 2026，会为 nvcc/CMake 自动添加 `-allow-unsupported-compiler` 以临时绕过版本检查（功能可用，但更复杂项目建议使用 VS 2022 稳定版）。

---

如需切换到其它 GPU（如 RTX 3060/4070/2080 等），我可以把 `-gencode` / `CUDA_ARCHITECTURES` 改成对应的 SM，并保留 PTX 以获得前向兼容。

---

## cuDNN 集成与验证（可选）

本仓库的 CMake 会在配置阶段尝试查找 cuDNN：

- 头文件：`cudnn.h`
- 库：`cudnn.lib`

若你的系统未安装 cuDNN 开发包（仅有运行时 DLL），将无法链接 cuDNN，程序会提示"cuDNN not found… skipping demo"。如需启用：

1. 安装与 CUDA 版本匹配的 cuDNN（Windows x64，Developer/Library 包），并记下安装路径，例如：

   - `C:\Program Files\NVIDIA\CUDNN`（推荐）
   - 或你自定义的任意目录（需包含 `include\cudnn.h` 与 `lib\x64\cudnn.lib`）

2. 在 PowerShell 会话中设置环境变量并临时加入 PATH：

   ```powershell
   $env:CUDNN_ROOT = "C:\Program Files\NVIDIA\CUDNN"
   $env:Path = "$env:CUDNN_ROOT\bin;" + $env:Path   # 确保运行时 DLL 可被找到
   ```

3. 重新配置/构建并运行（可用仓库脚本）：

```powershell
# 请将 <仓库根目录> 替换为实际路径
powershell -NoProfile -ExecutionPolicy Bypass -File <仓库根目录>\cuda-cmake\scripts\configure_build_run.ps1
```

成功后，你将看到：

- 普通 CUDA kernel 的输出：`1 2 3 ... 16`
- 以及一段“cuDNN conv output (first N floats): ...” 的卷积结果。

如果仍找不到 cuDNN：

- 确认 `CUDNN_ROOT\include\cudnn.h` 与 `CUDNN_ROOT\lib\x64\cudnn.lib` 存在。
- 确认 `CUDNN_ROOT\bin` 下存在 `cudnn64*.dll`，并已加入当前会话 PATH。
- 也可以把上述两个步骤写入你的 PowerShell profile 以长期生效。
