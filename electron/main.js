const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let mainWindow = null;
let renderProcess = null;

function getRustBinaryPath() {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, 'rt-next-week.exe');
    }
    return path.join(__dirname, '..', 'target', 'release', 'rt-next-week.exe');
}

function getCalibrationPath() {
    return path.join(app.getPath('userData'), 'calibration.json');
}

function getGpuCalibrationPath() {
    return path.join(app.getPath('userData'), 'calibration-gpu.json');
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 820,
        minWidth: 900,
        minHeight: 700,
        title: 'RT Renderer - Monte Carlo Path Tracer',
        icon: path.join(__dirname, 'assets', 'icon.ico'),
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
    mainWindow.setMenuBarVisibility(false);
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (renderProcess) {
        renderProcess.kill();
    }
    app.quit();
});

// Read calibration data
ipcMain.handle('read-calibration', async () => {
    const calPath = getCalibrationPath();
    if (fs.existsSync(calPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(calPath, 'utf8'));
            if (data.app_version !== app.getVersion()) {
                try { fs.unlinkSync(calPath); } catch (__) {}
                return null;
            }
            return data;
        } catch (_) {
            // Corrupted file — delete and re-calibrate
            try { fs.unlinkSync(calPath); } catch (__) {}
            return null;
        }
    }
    return null;
});

// Read GPU calibration data
ipcMain.handle('read-gpu-calibration', async () => {
    const calPath = getGpuCalibrationPath();
    if (fs.existsSync(calPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(calPath, 'utf8'));
            if (data.app_version !== app.getVersion()) {
                try { fs.unlinkSync(calPath); } catch (__) {}
                return null;
            }
            return data;
        } catch (_) {
            try { fs.unlinkSync(calPath); } catch (__) {}
            return null;
        }
    }
    return null;
});

// Check if GPU binary is available
// Check GPU availability using --check-gpu diagnostic flag (lightweight JSON probe)
ipcMain.handle('check-gpu', async () => {
    const binaryPath = getRustBinaryPath();
    if (!fs.existsSync(binaryPath)) {
        return { available: false, error: 'Binary not found', device_name: null };
    }

    return new Promise((resolve) => {
        const child = spawn(binaryPath, ['--check-gpu'], {
            stdio: ['ignore', 'pipe', 'pipe'],
            timeout: 15000,
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => { stdout += data.toString(); });
        child.stderr.on('data', (data) => { stderr += data.toString(); });

        child.on('error', () => {
            resolve({ available: false, error: 'Failed to spawn binary for GPU check', device_name: null });
        });

        child.on('close', (code) => {
            if (code === 0) {
                try {
                    // Extract the last JSON line from stdout (OptiX debug logs may precede it)
                    const lines = stdout.trim().split('\n');
                    let jsonLine = '';
                    for (let i = lines.length - 1; i >= 0; i--) {
                        const trimmed = lines[i].trim();
                        if (trimmed.startsWith('{')) {
                            jsonLine = trimmed;
                            break;
                        }
                    }
                    const result = JSON.parse(jsonLine || stdout.trim());
                    const deviceName = result.cuda?.device_name || result.optix?.device_name || null;
                    const available = result.status === 'ok';
                    const optixAvailable = result.optix?.available || false;
                    // Collect any warnings from the diagnostic
                    const warnings = result.cuda?.warnings || [];

                    resolve({
                        available,
                        device_name: deviceName,
                        optix_available: optixAvailable,
                        compute_capability: result.cuda?.compute_capability || null,
                        driver_version: result.cuda?.driver_version || null,
                        vram_mb: result.cuda?.vram_mb || null,
                        warnings,
                        error: result.status === 'ok' ? null
                            : (result.optix?.error || result.cuda?.error || 'Unknown GPU error'),
                        diagnostic: result,
                    });
                } catch (parseErr) {
                    resolve({
                        available: false,
                        error: 'Failed to parse GPU diagnostic JSON output',
                        device_name: null,
                        stderr_tail: stderr.slice(-500),
                        stdout_tail: stdout.slice(-500),
                    });
                }
            } else {
                const stderrLower = stderr.toLowerCase();
                if (stderrLower.includes('cuda feature') || stderrLower.includes('built without')) {
                    resolve({
                        available: false,
                        error: '二进制文件未启用 CUDA 功能。请使用 --features cuda 重新编译。',
                        device_name: null,
                    });
                } else if (code === null) {
                    resolve({
                        available: false,
                        error: 'GPU 检查超时 (15 秒)',
                        device_name: null,
                    });
                } else {
                    resolve({
                        available: false,
                        error: 'GPU check failed (exit ' + code + ')',
                        device_name: null,
                        stderr_tail: stderr.slice(-500),
                    });
                }
            }
        });
    });
});

// Run calibration benchmark
ipcMain.handle('run-calibration', async (_event, useGpu = false) => {
    const binaryPath = getRustBinaryPath();
    if (!fs.existsSync(binaryPath)) {
        return { error: `找不到 Rust 二进制文件: ${binaryPath}` };
    }

    const calOutput = path.join(app.getPath('temp'), 'rt_calibration.png');

    // 使用更大的校准负载以减少固定开销占比
    const width = useGpu ? 1280 : 320;
    const height = useGpu ? 720 : 180;
    const samples = useGpu ? 4 : 8;

    const args = [
        '--width', String(width),
        '--height', String(height),
        '--samples', String(samples),
        '--max-depth', '5',
        '--output', calOutput,
        '--seed', '0',
        '--calibrate',
    ];
    if (useGpu) {
        args.push('--gpu');
    }

    return new Promise((resolve) => {
        const child = spawn(binaryPath, args, { stdio: ['ignore', 'pipe', 'pipe'] });

        let stdoutOut = '';
        let stderrOut = '';

        child.stdout.on('data', (data) => { stdoutOut += data.toString(); });
        child.stderr.on('data', (data) => { stderrOut += data.toString(); });

        const timeout = setTimeout(() => {
            child.kill();
            resolve({ error: '校准超时 (60 秒)', fallback: useGpu ? 50000 : 500 });
        }, 60000);

        child.on('error', () => {
            clearTimeout(timeout);
            resolve({ error: '无法启动校准基准测试', fallback: useGpu ? 50000 : 500 });
        });

        child.on('close', (code) => {
            clearTimeout(timeout);
            if (code === 0) {
                // 从 stdout 解析校准 JSON
                let pixelSamplesPerMs = null;
                for (const line of stdoutOut.split('\n')) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    try {
                        const parsed = JSON.parse(trimmed);
                        if (typeof parsed.pixel_samples_per_ms === 'number') {
                            pixelSamplesPerMs = parsed.pixel_samples_per_ms;
                            break;
                        }
                    } catch (_) {}
                }
                // 如果 stdout 没有，尝试 stderr（GPU eprintln 调试输出可能混杂）
                if (pixelSamplesPerMs === null) {
                    for (const line of stderrOut.split('\n')) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        try {
                            const parsed = JSON.parse(trimmed);
                            if (typeof parsed.pixel_samples_per_ms === 'number') {
                                pixelSamplesPerMs = parsed.pixel_samples_per_ms;
                                break;
                            }
                        } catch (_) {}
                    }
                }

                if (pixelSamplesPerMs !== null) {
                    const calibration = {
                        pixel_samples_per_ms: Math.round(pixelSamplesPerMs * 100) / 100,
                        calibrated_at: new Date().toISOString(),
                        gpu: useGpu,
                        app_version: app.getVersion(),
                    };
                    const calPath = useGpu ? getGpuCalibrationPath() : getCalibrationPath();
                    try {
                        fs.writeFileSync(calPath, JSON.stringify(calibration, null, 2));
                    } catch (_) {}
                    try { fs.unlinkSync(calOutput); } catch (_) {}
                    resolve(calibration);
                } else {
                    resolve({ error: '校准输出中未找到有效数据', fallback: useGpu ? 50000 : 500 });
                }
            } else {
                resolve({ error: `校准失败 (退出码: ${code})`, fallback: useGpu ? 50000 : 500 });
            }
        });
    });
});

// Start render
ipcMain.handle('start-render', async (_event, config) => {
    if (renderProcess) {
        return { error: '正在渲染中，请等待或取消当前渲染。' };
    }

    const binaryPath = getRustBinaryPath();
    if (!fs.existsSync(binaryPath)) {
        return { error: `找不到渲染器二进制文件: ${binaryPath}\n请先编译: cargo build --release` };
    }

    const outputPath = config.output || path.join(app.getPath('temp'), 'rt_render_output.png');
    const args = [
        '--width', String(config.width),
        '--height', String(config.height),
        '--samples', String(config.samples),
        '--max-depth', String(config.maxDepth),
        '--output', outputPath,
        '--json',
    ];
    if (config.seed !== undefined && config.seed !== null && config.seed !== '') {
        args.push('--seed', String(config.seed));
    }
    if (config.gpu) {
        args.push('--gpu');
    }
    if (config.denoise) {
        args.push('--denoise');
    }

    return new Promise((resolve) => {
        renderProcess = spawn(binaryPath, args, {
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let buffer = '';
        let lastProgress = null;
        let stderrLines = [];

        renderProcess.stdout.on('data', (data) => {
            buffer += data.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                try {
                    const msg = JSON.parse(trimmed);
                    if (msg.type === 'progress') {
                        lastProgress = msg;
                    }
                    if (mainWindow && !mainWindow.isDestroyed()) {
                        mainWindow.webContents.send('render-progress', msg);
                    }
                } catch (_) {
                    if (mainWindow && !mainWindow.isDestroyed()) {
                        mainWindow.webContents.send('render-log', trimmed);
                    }
                }
            }
        });

        renderProcess.stderr.on('data', (data) => {
            const lines = data.toString().split('\n').filter(l => l.trim());
            stderrLines.push(...lines);
            for (const line of lines) {
                if (mainWindow && !mainWindow.isDestroyed()) {
                    mainWindow.webContents.send('render-log', line);
                }
            }
        });

        renderProcess.on('error', (err) => {
            renderProcess = null;
            resolve({ error: `渲染器启动失败: ${err.message}` });
        });

        renderProcess.on('close', (code) => {
            renderProcess = null;
            if (mainWindow && !mainWindow.isDestroyed()) {
                if (code === 0) {
                    mainWindow.webContents.send('render-done', {
                        output: outputPath,
                        progress: lastProgress,
                    });
                } else {
                    const errDetail = stderrLines.length > 0
                        ? '\nEngine output:\n' + stderrLines.slice(-10).join('\n')
                        : '';
                    const exitMsg = code === null
                        ? '渲染进程被信号终止'
                        : `渲染器异常退出 (退出码: ${code})`;
                    mainWindow.webContents.send('render-error', {
                        message: exitMsg + '\n请检查分辨率、采样数或可用内存。' + errDetail,
                    });
                }
            }
            resolve({ started: true });
        });
    });
});

// Cancel render
ipcMain.on('cancel-render', () => {
    if (renderProcess) {
        renderProcess.kill();
        renderProcess = null;
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('render-error', { message: '用户取消了渲染' });
        }
    }
});

// Get image as base64 data URL
ipcMain.handle('get-image-data', async (_event, imagePath) => {
    try {
        const data = fs.readFileSync(imagePath);
        const ext = path.extname(imagePath).toLowerCase();
        const mime = ext === '.png' ? 'image/png' : 'image/jpeg';
        const base64 = data.toString('base64');
        return { dataUrl: `data:${mime};base64,${base64}` };
    } catch (err) {
        return { error: `Failed to read rendered image: ${err.message}` };
    }
});
