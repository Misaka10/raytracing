@echo off
echo ============================================
echo  RT Renderer - Full Build ^& Package
echo ============================================
echo.

cd /d "%~dp0"

set BUILD_GPU=0
if "%1"=="--gpu" set BUILD_GPU=1
if "%1"=="-g" set BUILD_GPU=1

echo [1/3] Building Rust path tracer (release)...
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust build failed!
    pause
    exit /b 1
)

if %BUILD_GPU%==1 (
    echo.
    echo [2/3] Building Rust path tracer with CUDA (GPU release)...
    cargo build --release --features cuda
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: GPU build failed. CPU binary will work, GPU mode disabled.
        echo          Install CUDA Toolkit 12.x and OptiX SDK 9.x, then set OPTIX_PATH.
        set BUILD_GPU=0
    )
)

echo.
echo [3/3] Packaging Electron app...
cd electron
if not exist "node_modules\" (
    echo       Installing npm dependencies...
    call npm install
)
call npx electron-builder --win portable
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Electron packaging failed. The Rust exe is still built.
    echo        Run 'cd electron ^&^& npm start' for development mode.
    pause
    exit /b 1
)

cd ..
echo.
echo ============================================
echo  Build complete!
echo  Output: electron\dist-pkg\win-unpacked\RT Renderer.exe
echo  Portable zip: electron\dist-pkg\ (see electron-builder output for filename)
echo.
if %BUILD_GPU%==1 (
    echo  GPU mode: ENABLED (OptiX RT Core + Tensor Core)
    echo  Usage: .\rt-next-week.exe --gpu --denoise -s 50 -o output.png
) else (
    echo  GPU mode: NOT included (use --gpu flag to build with CUDA)
    echo          .\build.bat --gpu   to include GPU support
)
echo ============================================
pause
