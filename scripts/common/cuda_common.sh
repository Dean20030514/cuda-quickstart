#!/usr/bin/env bash
# cuda_common.sh - Shared utilities for Linux CUDA build scripts
# cuda_common.sh - Linux CUDA 构建脚本的公共工具函数
#
# Usage: source this file from other scripts
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/../../scripts/common/cuda_common.sh"

set -euo pipefail

# ===== Color helpers =====
_color_red='\033[0;31m'
_color_green='\033[0;32m'
_color_yellow='\033[1;33m'
_color_cyan='\033[0;36m'
_color_reset='\033[0m'

info()  { printf "${_color_cyan}%s${_color_reset}\n" "$*"; }
warn()  { printf "${_color_yellow}%s${_color_reset}\n" "$*"; }
ok()    { printf "${_color_green}%s${_color_reset}\n" "$*"; }
err()   { printf "${_color_red}%s${_color_reset}\n" "$*" >&2; }

# ===== check_cuda_installed =====
# Verify that nvcc is available on PATH
check_cuda_installed() {
  if ! command -v nvcc &>/dev/null; then
    err "nvcc not found on PATH. Please install CUDA Toolkit and ensure nvcc is in your PATH."
    exit 1
  fi
}

# ===== detect_gpu_arch =====
# Auto-detect the compute capability of the first GPU via nvidia-smi.
# Returns the SM number (e.g. "89" for compute_8.9) or empty string on failure.
detect_gpu_arch() {
  local cap
  if command -v nvidia-smi &>/dev/null; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
    if [[ -n "$cap" ]]; then
      # Convert "8.9" -> "89"
      echo "${cap//./}"
      return
    fi
  fi
  echo ""
}

# ===== format_elapsed =====
# Convert seconds to mm:ss.ff format
format_elapsed() {
  local secs="$1"
  local mins frac
  mins=$(echo "$secs" | awk '{printf "%02d", int($1/60)}')
  frac=$(echo "$secs" | awk '{printf "%05.2f", $1 - int($1/60)*60}')
  echo "${mins}:${frac}"
}
