param(
    [ValidateSet('Debug','Release')]
    [string]$Configuration = 'Debug',
    # 覆盖 CMAKE_CUDA_ARCHITECTURES（如 90;89;86），不指定则使用 CMakeLists 默认的 native
    [string]$CudaArch,
    # 是否启用 CMake 里的 CUDA_ENABLE_FAST_MATH（Release 推荐 ON）
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

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# Prefer Ninja if available (bundled with most CMake installs on Windows), fallback to NMake
$ninja = Get-Command ninja -ErrorAction SilentlyContinue
$generator = if ($ninja) { 'Ninja' } else { 'NMake Makefiles' }
$buildPrefix = if ($generator -eq 'Ninja') { 'build-ninja-' } else { 'build-nmake-' }
$buildDir = Join-Path $root ($buildPrefix + $Configuration)
if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

$vsdev = Find-VsDevCmd
Write-Host "[configure_build_run] Using VsDevCmd: $vsdev" -ForegroundColor Cyan

# 使用公共模块函数判断是否需要添加 unsupported-compiler 标志
$cudaFlag = Get-CudaUnsupportedCompilerFlag -VsDevCmdPath $vsdev

# Build the CMake configure command conditionally
$argsList = @(
    ('-S . -B "{0}" -G "{1}" -DCMAKE_BUILD_TYPE={2}' -f $buildDir, $generator, $Configuration)
)
if (-not [string]::IsNullOrEmpty($cudaFlag)) {
    $argsList += ('-D CMAKE_CUDA_FLAGS={0}' -f $cudaFlag)
}
if (-not [string]::IsNullOrEmpty($CudaArch)) {
    $argsList += ('-D CMAKE_CUDA_ARCHITECTURES={0}' -f $CudaArch)
}
if ($FastMath) {
    $argsList += '-D CUDA_ENABLE_FAST_MATH=ON'
}
$cmakeConfigure = 'cmake ' + ($argsList -join ' ')

# Build the command chain for CMD to ensure VsDevCmd environment is applied
$cfgCmd = @(
    ('call "{0}" -arch=amd64' -f $vsdev),
    '&&',
    $cmakeConfigure,
    '&&',
    ('cmake --build "{0}" --config {1}' -f $buildDir, $Configuration),
    '&&',
    ('pushd "{0}" && cudatest.exe && popd' -f $buildDir)
) -join ' '

Write-Host "[configure_build_run] Generator: $generator" -ForegroundColor Yellow
if (-not [string]::IsNullOrEmpty($cudaFlag)) { Write-Host "[configure_build_run] Using CUDA flag: $cudaFlag (VS not 2022)" -ForegroundColor DarkYellow }

Push-Location $root
try {
    cmd.exe /c $cfgCmd
}
finally {
    Pop-Location
}
