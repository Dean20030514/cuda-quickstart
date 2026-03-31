#!/usr/bin/env bash
# single-nvcc/scripts/build_and_run.sh - Linux build script for single-nvcc project
# single-nvcc/scripts/build_and_run.sh - 单文件 nvcc 编译的 Linux 构建脚本
#
# Usage:
#   bash scripts/build_and_run.sh                       # Debug, auto-detect GPU
#   bash scripts/build_and_run.sh -c Release             # Release mode
#   bash scripts/build_and_run.sh -s 89                  # Specify SM architecture
#   bash scripts/build_and_run.sh -c Release -f          # Release + FastMath
#   bash scripts/build_and_run.sh -b                     # Build only, don't run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/../../scripts/common/cuda_common.sh"

# ===== Parse arguments =====
CONFIGURATION="Debug"
SM=""
FAST_MATH=false
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)       CONFIGURATION="$2"; shift 2 ;;
    -s|--sm)           SM="$2"; shift 2 ;;
    -f|--fast-math)    FAST_MATH=true; shift ;;
    -b|--build-only)   BUILD_ONLY=true; shift ;;
    -h|--help)
      echo "Usage: $0 [-c Debug|Release] [-s SM] [-f] [-b]"
      echo "  -c, --config      Build configuration (Debug or Release, default: Debug)"
      echo "  -s, --sm          Target SM architecture (e.g. 89, 120). Auto-detected if omitted."
      echo "  -f, --fast-math   Enable --use_fast_math (Release recommended)"
      echo "  -b, --build-only  Build only, do not run"
      exit 0 ;;
    *) err "Unknown option: $1"; exit 1 ;;
  esac
done

check_cuda_installed

cd "$PROJECT_DIR"
mkdir -p build

# ===== Auto-detect GPU architecture =====
DETECTED_NATIVE=false
if [[ -z "$SM" ]]; then
  SM=$(detect_gpu_arch)
  if [[ -n "$SM" ]]; then
    DETECTED_NATIVE=true
  fi
fi

# ===== Generate gencode flags =====
if [[ -n "$SM" ]]; then
  GENCODE="-gencode=arch=compute_${SM},code=sm_${SM} -gencode=arch=compute_${SM},code=compute_${SM}"
else
  GENCODE="-gencode=arch=compute_120,code=sm_120"
  GENCODE+=" -gencode=arch=compute_100,code=sm_100"
  GENCODE+=" -gencode=arch=compute_90,code=sm_90"
  GENCODE+=" -gencode=arch=compute_89,code=sm_89"
  GENCODE+=" -gencode=arch=compute_86,code=sm_86"
  GENCODE+=" -gencode=arch=compute_120,code=compute_120"
fi

# ===== Build compile command =====
COMPILE_CMD="nvcc -std=c++17 ${GENCODE} main.cu -o build/main"
if [[ "$CONFIGURATION" == "Release" ]]; then
  COMPILE_CMD="$COMPILE_CMD -O3 -DNDEBUG"
  if $FAST_MATH; then
    COMPILE_CMD="$COMPILE_CMD --use_fast_math -Xptxas -O3"
  fi
else
  COMPILE_CMD="$COMPILE_CMD -g -G"
fi

# ===== Print status =====
info "[build_and_run] Configuration: $CONFIGURATION"
if $DETECTED_NATIVE; then
  warn "[build_and_run] Target SM: $SM (auto-detected native GPU)"
elif [[ -n "$SM" ]]; then
  warn "[build_and_run] Target SM: $SM"
else
  warn "[build_and_run] Target: multi-arch fatbin (nvidia-smi not available)"
fi
if $FAST_MATH; then warn "[build_and_run] FastMath: ON"; fi

# ===== Compile and run =====
START_TIME=$(date +%s.%N)

eval "$COMPILE_CMD"

if ! $BUILD_ONLY; then
  ./build/main
fi

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

ok "[build_and_run] Completed in $(format_elapsed "$ELAPSED")"
