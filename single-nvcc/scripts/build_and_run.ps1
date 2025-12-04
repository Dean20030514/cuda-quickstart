param(
    [switch]$BuildOnly,
    [ValidateSet('Debug','Release')]
    [string]$Configuration = 'Debug',
    # 指定 SM 架构（如 89、86、75）。不指定时默认生成通用 fatbin (sm_90/89/86/75 + PTX)。
    [string]$Sm,
    # 启用 --use_fast_math（Release/RelWithDebInfo 建议开启）
    [switch]$FastMath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# 导入公共模块
$modulePath = Join-Path $PSScriptRoot '..\..\scripts\common\VsHelper.psm1'
if (Test-Path $modulePath) {
    Import-Module $modulePath -Force
} else {
    throw "VsHelper.psm1 not found at: $modulePath"
}

$projectDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $projectDir
try {
    if (-not (Test-Path 'build')) { New-Item -ItemType Directory -Path 'build' | Out-Null }

    $vsdev = Find-VsDevCmd
    $cudaAllowFlag = Get-CudaUnsupportedCompilerFlag -VsDevCmdPath $vsdev

    # 生成 -gencode 片段
    if ($Sm) {
        $arch = $Sm.Trim()
        $gencode = "-gencode=arch=compute_${arch},code=sm_${arch} -gencode=arch=compute_${arch},code=compute_${arch}"
    } else {
        # 默认生成覆盖面广的 fatbin，支持主流 GPU；体积略大但便于演示
        $gencode = @(
            '-gencode=arch=compute_90,code=sm_90',
            '-gencode=arch=compute_89,code=sm_89',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_75,code=sm_75',
            # 附带一份较新的 PTX 以获得前向兼容
            '-gencode=arch=compute_89,code=compute_89'
        ) -join ' '
    }

    # Build the cmd command line to init env, compile, and optionally run
    $common = "nvcc -std=c++17 ${gencode} main.cu -o build\main.exe"
    if ($Configuration -eq 'Release') {
        $compileCmd = "$common -O3 -DNDEBUG"
        if ($FastMath) { $compileCmd = "$compileCmd --use_fast_math -Xptxas -O3" }
    } else {
        $compileCmd = "$common -g -G -lineinfo"
    }
    if (-not [string]::IsNullOrEmpty($cudaAllowFlag)) { $compileCmd = "$compileCmd $cudaAllowFlag" }
    if ($BuildOnly) {
        $cmd = 'call "{0}" -arch=amd64 && {1}' -f $vsdev, $compileCmd
    } else {
        $cmd = 'call "{0}" -arch=amd64 && {1} && build\main.exe' -f $vsdev, $compileCmd
    }

    Write-Host "[build_and_run] Using VsDevCmd: $vsdev" -ForegroundColor Cyan
    if (-not [string]::IsNullOrEmpty($cudaAllowFlag)) { Write-Host "[build_and_run] Using CUDA flag: $cudaAllowFlag (VS not 2022)" -ForegroundColor DarkYellow }
    cmd.exe /c $cmd
}
finally {
    Pop-Location
}
