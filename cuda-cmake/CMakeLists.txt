cmake_minimum_required(VERSION 3.24)
project(CudaDemo LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 自动探测本机 GPU 架构（需要 CMake ≥ 3.24）
set(CMAKE_CUDA_ARCHITECTURES native)

# Use the modern CUDAToolkit package for runtime linkage
find_package(CUDAToolkit REQUIRED)

add_executable(cudatest src/main.cu)

# 添加公共头文件目录
target_include_directories(cudatest PRIVATE ${CMAKE_SOURCE_DIR}/../common)

## ===== 构建优化选项 =====
# 可选：开启 fast-math（牺牲部分 IEEE 精度以换取更高性能）
option(CUDA_ENABLE_FAST_MATH "Enable --use_fast_math for CUDA files in Release/RelWithDebInfo" ON)
# 可选：在 MSVC 下使用静态运行时（/MT、/MTd）；默认 OFF 保持与多数外部库兼容
option(MSVC_STATIC_RUNTIME "Link static MSVC runtime (/MT)" OFF)

# Link against CUDA runtime
target_link_libraries(cudatest PRIVATE CUDA::cudart)

# Debug：便于调试的编译参数；Release：更激进的优化
target_compile_options(cudatest PRIVATE
	# CUDA Debug 给予行号与设备端调试符号
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G;-lineinfo>
    
	# CUDA Release/RelWithDebInfo：O3 +（可选）fast-math + 更优 ptxas
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
		-O3
		$<$<BOOL:${CUDA_ENABLE_FAST_MATH}>:--use_fast_math>
		# 注意：将 -Xptxas 的参数与等号连接为单个标志，防止 CMake/Ninja 在追加依赖项标志时
		# 将后续 -MD/-MT/-MF 误当作 -Xptxas 的参数，导致 nvcc 依赖文件选项解析失败。
		-Xptxas=-O3
	>
)

# MSVC 运行时可选静态链接
if (MSVC AND MSVC_STATIC_RUNTIME)
	# 仅对 host 端有效；CUDA host 编译同样遵循该设置
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Try to locate cuDNN and link if available
set(CUDNN_HINT_DIRS
	# Prefer CUDA Toolkit installation first
	"$ENV{CUDA_PATH}"
	"$ENV{CUDA_PATH_V13_0}"
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
	# Then user-provided CUDNN_ROOT and the standalone NVIDIA CUDNN installer
	"$ENV{CUDNN_ROOT}"
	"C:/Program Files/NVIDIA/CUDNN"
)

# Also consider versioned subfolders like C:/Program Files/NVIDIA/CUDNN/v9.14
file(GLOB CUDNN_VERSION_DIRS
	# Prefer versioned subfolders under CUDA Toolkit first
	"$ENV{CUDA_PATH}/v*"
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*"
	# Then versioned subfolders from user CUDNN_ROOT and standalone NVIDIA CUDNN
	"$ENV{CUDNN_ROOT}/v*"
	"C:/Program Files/NVIDIA/CUDNN/v*"
)

# 动态查找所有版本化的 cuDNN lib 目录（支持 cuDNN 9.x 的 CUDA 版本子目录结构）
file(GLOB CUDNN_LIB_VERSIONS
	"$ENV{CUDA_PATH}/lib/*/x64"
	"$ENV{CUDNN_ROOT}/lib/*/x64"
	"C:/Program Files/NVIDIA/CUDNN/lib/*/x64"
)

# cuDNN 9.x 新结构: CUDNN/v9.14/lib/13.0/x64 - 先找 v* 目录，再找 lib/*/x64
file(GLOB _CUDNN_V_DIRS "C:/Program Files/NVIDIA/CUDNN/v*")
foreach(_vdir ${_CUDNN_V_DIRS})
	file(GLOB _CUDNN_LIB_SUBDIRS "${_vdir}/lib/*/x64")
	list(APPEND CUDNN_LIB_VERSIONS ${_CUDNN_LIB_SUBDIRS})
endforeach()

find_library(CUDNN_LIBRARY
	NAMES cudnn
	HINTS ${CUDNN_HINT_DIRS} ${CUDNN_VERSION_DIRS} ${CUDNN_LIB_VERSIONS}
	PATH_SUFFIXES
		lib
		lib/x64
)

# 动态查找所有版本化的 cuDNN include 目录（支持 cuDNN 9.x 的 CUDA 版本子目录结构）
file(GLOB CUDNN_INCLUDE_VERSIONS
	"$ENV{CUDA_PATH}/include/*"
	"$ENV{CUDNN_ROOT}/include/*"
)

# cuDNN 9.x 新结构: CUDNN/v9.14/include/13.0 - 先找 v* 目录，再找 include/*
foreach(_vdir ${_CUDNN_V_DIRS})
	file(GLOB _CUDNN_INC_SUBDIRS "${_vdir}/include/*")
	list(APPEND CUDNN_INCLUDE_VERSIONS ${_CUDNN_INC_SUBDIRS})
endforeach()

find_path(CUDNN_INCLUDE_DIR
	NAMES cudnn.h
	HINTS ${CUDNN_HINT_DIRS} ${CUDNN_VERSION_DIRS} ${CUDNN_INCLUDE_VERSIONS}
	PATH_SUFFIXES
		include
)

if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
	message(STATUS "cuDNN found: lib=${CUDNN_LIBRARY}, include=${CUDNN_INCLUDE_DIR}")
	target_include_directories(cudatest PRIVATE ${CUDNN_INCLUDE_DIR})
	target_link_libraries(cudatest PRIVATE ${CUDNN_LIBRARY})
	target_compile_definitions(cudatest PRIVATE HAVE_CUDNN=1)
else()
	message(WARNING "cuDNN not found; building without cuDNN demo. Set CUDNN_ROOT to your cuDNN install root if needed.")
endif()

# 可选：NVTX 标记，便于 Nsight Systems/Compute 分析
if (TARGET CUDA::nvtx3)
	target_link_libraries(cudatest PRIVATE CUDA::nvtx3)
	target_compile_definitions(cudatest PRIVATE HAVE_NVTX=1)
endif()

# 安装规则
install(TARGETS cudatest RUNTIME DESTINATION bin)
