#!/usr/bin/env bash
# cuda-cmake/scripts/configure_build_run.sh - Linux CMake build script
# cuda-cmake/scripts/configure_build_run.sh - CMake 工程的 Linux 构建脚本
#
# Usage:
#   bash scripts/configure_build_run.sh                        # Debug, native arch
#   bash scripts/configure_build_run.sh -c Release              # Release mode
#   bash scripts/configure_build_run.sh -c Release -f           # Release + FastMath
#   bash scripts/configure_build_run.sh -a "89;86"              # Specify architectures
#   bash scripts/configure_build_run.sh --clean                 # Clean rebuild
#   bash scripts/configure_build_run.sh -b                      # Build only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/../../scripts/common/cuda_common.sh"

# ===== Parse arguments =====
CONFIGURATION="Debug"
CUDA_ARCH=""
FAST_MATH=false
BUILD_ONLY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)       CONFIGURATION="$2"; shift 2 ;;
    -a|--arch)         CUDA_ARCH="$2"; shift 2 ;;
    -f|--fast-math)    FAST_MATH=true; shift ;;
    -b|--build-only)   BUILD_ONLY=true; shift ;;
    --clean)           CLEAN=true; shift ;;
    -h|--help)
      echo "Usage: $0 [-c Debug|Release] [-a ARCH] [-f] [-b] [--clean]"
      echo "  -c, --config      Build configuration (Debug or Release, default: Debug)"
      echo "  -a, --arch        CMAKE_CUDA_ARCHITECTURES (e.g. '89' or '89;86'). Default: native"
      echo "  -f, --fast-math   Enable CUDA_ENABLE_FAST_MATH"
      echo "  -b, --build-only  Build only, do not run"
      echo "  --clean           Remove build directory before configuring"
      exit 0 ;;
    *) err "Unknown option: $1"; exit 1 ;;
  esac
done

check_cuda_installed

# ===== Determine generator and build directory =====
if command -v ninja &>/dev/null; then
  GENERATOR="Ninja"
  BUILD_PREFIX="build-ninja-"
else
  GENERATOR="Unix Makefiles"
  BUILD_PREFIX="build-make-"
fi
BUILD_DIR="${PROJECT_DIR}/${BUILD_PREFIX}${CONFIGURATION}"

if $CLEAN && [[ -d "$BUILD_DIR" ]]; then
  warn "[configure_build_run] Cleaning build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# ===== Build CMake configure arguments =====
CMAKE_ARGS=(
  -S "$PROJECT_DIR"
  -B "$BUILD_DIR"
  -G "$GENERATOR"
  -DCMAKE_BUILD_TYPE="$CONFIGURATION"
)
if [[ -n "$CUDA_ARCH" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi
if $FAST_MATH; then
  CMAKE_ARGS+=(-DCUDA_ENABLE_FAST_MATH=ON)
fi

# ===== Print status =====
info "[configure_build_run] Configuration: $CONFIGURATION"
info "[configure_build_run] Generator: $GENERATOR"
info "[configure_build_run] Build dir: $BUILD_DIR"
if $FAST_MATH; then warn "[configure_build_run] FastMath: ON"; fi

# ===== Configure, build, run =====
START_TIME=$(date +%s.%N)

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --config "$CONFIGURATION"

if ! $BUILD_ONLY; then
  "$BUILD_DIR/cudatest"
fi

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

ok "[configure_build_run] Completed in $(format_elapsed "$ELAPSED")"
