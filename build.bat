@echo off
echo ============================================
echo  RT Renderer - Full Build ^& Package
echo ============================================
echo.

cd /d "%~dp0"

set BUILD_GPU=0
if "%1"=="--gpu" set BUILD_GPU=1
if "%1"=="-g" set BUILD_GPU=1

if %BUILD_GPU%==1 (
    echo [1/2] Building Rust path tracer with CUDA (GPU release^)...
    cargo build --release --features cuda
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: GPU build failed!
        echo        Install CUDA Toolkit 13.1 and OptiX SDK 9.1.0, then set OPTIX_PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/2] Building Rust path tracer (CPU release^)...
    cargo build --release
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Rust build failed!
        pause
        exit /b 1
    )
)

echo.
echo [2/2] Packaging Electron app...
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
