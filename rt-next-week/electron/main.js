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

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 820,
        minWidth: 900,
        minHeight: 600,
        title: 'RT Renderer - Monte Carlo Path Tracer',
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
        return JSON.parse(fs.readFileSync(calPath, 'utf8'));
    }
    return null;
});

// Run calibration benchmark
ipcMain.handle('run-calibration', async () => {
    const binaryPath = getRustBinaryPath();
    if (!fs.existsSync(binaryPath)) {
        return { error: `Rust binary not found: ${binaryPath}` };
    }

    const calOutput = path.join(app.getPath('temp'), 'rt_calibration.png');
    const startTime = Date.now();

    return new Promise((resolve) => {
        const child = spawn(binaryPath, [
            '--width', '160',
            '--height', '90',
            '--samples', '16',
            '--max-depth', '5',
            '--output', calOutput,
            '--seed', '0',
            '--json',
        ], { stdio: ['ignore', 'pipe', 'pipe'] });

        child.on('close', (code) => {
            const elapsedMs = Date.now() - startTime;
            if (code === 0) {
                const pixelSamples = 160 * 90 * 16;
                const pixelSamplesPerMs = pixelSamples / Math.max(elapsedMs, 1);
                const calibration = {
                    pixel_samples_per_ms: Math.round(pixelSamplesPerMs * 100) / 100,
                    calibrated_at: new Date().toISOString(),
                };
                fs.writeFileSync(getCalibrationPath(), JSON.stringify(calibration, null, 2));
                try { fs.unlinkSync(calOutput); } catch (_) {}
                resolve(calibration);
            } else {
                resolve({ error: `性能校准失败（退出码: ${code}），将使用默认估算值`, fallback: 200 });
            }
        });
    });
});

// Start render
ipcMain.handle('start-render', async (_event, config) => {
    if (renderProcess) {
        return { error: '已有渲染任务正在进行中，请等待当前任务完成或取消后再试' };
    }

    const binaryPath = getRustBinaryPath();
    if (!fs.existsSync(binaryPath)) {
        return { error: `找不到渲染引擎程序: ${binaryPath}\n请先运行: cargo build --release` };
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
            resolve({ error: `启动渲染引擎失败: ${err.message}` });
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
                        ? '\n引擎输出:\n' + stderrLines.slice(-10).join('\n')
                        : '';
                    const exitMsg = code === null || code === null
                        ? '渲染进程被信号终止'
                        : `渲染引擎异常退出（退出码: ${code}）`;
                    mainWindow.webContents.send('render-error', {
                        message: exitMsg + '\n请检查分辨率、采样数等参数是否过大导致内存不足' + errDetail,
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
            mainWindow.webContents.send('render-error', { message: '渲染已被用户取消' });
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
        return { error: `读取渲染结果图片失败: ${err.message}` };
    }
});
