@echo off
echo ============================================
echo  RT Renderer - Full Build & Package
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Building Rust path tracer (release)...
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust build failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Copying executable to project root...
copy /Y target\release\rt-next-week.exe . >nul
echo       Done.

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
echo  Output: electron\dist\RT Renderer 1.0.0.exe
echo ============================================
pause
