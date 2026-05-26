// RT 渲染器 — 前端渲染进程
// 负责：用户输入采集、渲染参数验证、进度显示、GPU 检测、硬件校准与预估
// DOM references
const $ = (sel) => document.querySelector(sel);
const widthSelect = $('#width-preset');
const widthInput = $('#width');
const heightInput = $('#height');
const aspectSelect = $('#aspect-ratio');
const samplesSlider = $('#samples');
const samplesVal = $('#samples-val');
const depthSlider = $('#max-depth');
const depthVal = $('#depth-val');
const seedInput = $('#seed');
const timeVal = $('#time-value');
const memoryVal = $('#memory-value');
const calStatus = $('#calibration-status');
const btnStart = $('#btn-start');
const btnCancel = $('#btn-cancel');
const errorDisplay = $('#error-display');
const placeholder = $('#placeholder');
const outputImage = $('#output-image');
const progressSection = $('#progress-section');
const progressBar = $('#progress-bar');
const progressPercent = $('#progress-percent');
const progressEta = $('#progress-eta');
const gpuStatus = $('#gpu-status');
const denoiseRow = $('#denoise-row');
const denoiseCheck = $('#denoise');
const btnPresetBack = $('#btn-preset-back');
const gpuLabel = $('#gpu-label');
const cpuRadio = document.querySelector('input[name="renderer"][value="cpu"]');
const gpuRadio = document.querySelector('input[name="renderer"][value="gpu"]');

// State
let calibration = null;       // CPU calibration
let gpuCalibration = null;    // GPU calibration
let gpuAvailable = false;
let renderStartTime = null;
let isRendering = false;

// Aspect ratio helpers
function getAspectRatio() {
    return parseFloat(aspectSelect.value);
}

function calcHeight(width, ratio) {
    return Math.max(1, Math.round(width / ratio));
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function isGpuMode() {
    return gpuRadio.checked;
}

// Update height when width or aspect changes
function updateHeight() {
    const w = parseInt(widthInput.value) || 3840;
    const ratio = getAspectRatio();
    heightInput.value = calcHeight(w, ratio);
}

function updateEstimates() {
    const w = parseInt(widthInput.value) || 3840;
    const h = parseInt(heightInput.value) || 2160;
    const spp = parseInt(samplesSlider.value) || 400;

    // Memory estimate
    const bytesPerPixel = isGpuMode() ? 48 : 6;
    const mb = (w * h * bytesPerPixel * 1.3) / (1024 * 1024);
    memoryVal.textContent = mb >= 1024 ? `~${(mb / 1024).toFixed(2)} GB` : `~${Math.round(mb)} MB`;

    if (mb > 500) {
        memoryVal.className = 'estimate-value warning-red';
    } else {
        memoryVal.className = 'estimate-value';
    }

    // Time estimate
    const cal = isGpuMode() ? gpuCalibration : calibration;
    if (cal && cal.pixel_samples_per_ms) {
        const pixelSamples = w * h * spp;
        const ms = pixelSamples / cal.pixel_samples_per_ms;
        timeVal.textContent = formatDuration(ms);
        if (ms > 7200000) {
            timeVal.className = 'estimate-value warning-red';
        } else if (ms > 1800000) {
            timeVal.className = 'estimate-value warning-yellow';
        } else {
            timeVal.className = 'estimate-value';
        }
    } else {
        timeVal.textContent = '--';
        timeVal.className = 'estimate-value';
    }
}

function formatDuration(ms) {
    if (ms < 1000) return '不到 1 秒';
    if (ms < 60000) return `~${Math.round(ms / 1000)}s`;
    const mins = Math.floor(ms / 60000);
    const hrs = Math.floor(mins / 60);
    const remainMins = mins % 60;
    if (hrs > 0) return `~${hrs}h ${remainMins}m`;
    return `~${mins}m`;
}

// Event: Renderer mode switch
cpuRadio.addEventListener('change', () => {
    denoiseRow.style.display = 'none';
    updateEstimates();
});

gpuRadio.addEventListener('change', () => {
    if (gpuAvailable) {
        denoiseRow.style.display = 'block';
    }
    updateEstimates();
});

// Event: width preset
widthSelect.addEventListener('change', () => {
    const val = widthSelect.value;
    if (val === 'custom') {
        widthSelect.style.display = 'none';
        widthInput.style.display = 'block';
        btnPresetBack.style.display = 'inline-block';
        widthInput.value = '1920';
        widthInput.focus();
        updateHeight();
    } else {
        widthSelect.style.display = 'inline-block';
        widthInput.style.display = 'none';
        btnPresetBack.style.display = 'none';
        widthInput.value = val;
        updateHeight();
    }
    updateEstimates();
});

// Event: back to preset from custom width
btnPresetBack.addEventListener('click', () => {
    widthSelect.value = '1920';
    widthSelect.style.display = 'inline-block';
    widthInput.style.display = 'none';
    btnPresetBack.style.display = 'none';
    updateHeight();
    updateEstimates();
});

// Event: custom width input
widthInput.addEventListener('input', () => {
    updateHeight();
    updateEstimates();
});

// Event: aspect ratio
aspectSelect.addEventListener('change', () => {
    updateHeight();
    updateEstimates();
});

// Event: samples slider
samplesSlider.addEventListener('input', () => {
    samplesVal.textContent = samplesSlider.value;
    updateEstimates();
    updatePresetButtons();
});

// Event: depth slider
depthSlider.addEventListener('input', () => {
    depthVal.textContent = depthSlider.value;
});

// Preset buttons
function updatePresetButtons() {
    const current = parseInt(samplesSlider.value);
    document.querySelectorAll('.preset').forEach((btn) => {
        btn.classList.toggle('active', parseInt(btn.dataset.samples) === current);
    });
}

document.querySelectorAll('.preset').forEach((btn) => {
    btn.addEventListener('click', () => {
        const val = parseInt(btn.dataset.samples);
        samplesSlider.value = val;
        samplesVal.textContent = val;
        updateEstimates();
        updatePresetButtons();
    });
});

// Start render
btnStart.addEventListener('click', async () => {
    if (isRendering) return;

    const width = parseInt(widthInput.value) || 3840;
    const height = parseInt(heightInput.value) || 2160;
    const samples = parseInt(samplesSlider.value) || 400;
    const maxDepth = parseInt(depthSlider.value) || 75;
    const seedRaw = seedInput.value.trim();
    const seed = seedRaw !== '' ? parseInt(seedRaw, 10) : undefined;
    const gpu = isGpuMode();
    const denoise = gpu && denoiseCheck.checked;

    if (isNaN(width) || width < 10 || width > 16384) {
        showError('宽度必须在 10 到 16384 之间');
        return;
    }
    if (isNaN(samples) || samples < 1) {
        showError('采样数必须至少为 1');
        return;
    }
    if (seedRaw !== '' && (isNaN(seed) || seed < 0)) {
        showError('种子必须为非负整数，留空则使用随机值');
        return;
    }

    setRenderingState(true);
    clearError();
    progressSection.style.display = 'block';
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressEta.textContent = '预计剩余: --';
    renderStartTime = Date.now();

    // Remove old event listeners
    if (window.electronAPI.removeAllListeners) {
        window.electronAPI.removeAllListeners();
    }

    // Listen for progress
    window.electronAPI.onProgress((msg) => {
        if (msg.type === 'start') {
            // rendering started
        } else if (msg.type === 'progress') {
            const pct = msg.total > 0 ? (msg.completed / msg.total * 100) : 0;
            progressBar.style.width = pct.toFixed(1) + '%';
            progressPercent.textContent = pct.toFixed(1) + '%';

            const elapsed = Date.now() - renderStartTime;
            if (msg.completed > 0 && elapsed > 500) {
                const totalMs = elapsed * (msg.total / msg.completed);
                const remaining = totalMs - elapsed;
                progressEta.textContent = '预计剩余: ' + formatDuration(remaining);
            }
        } else if (msg.type === 'done') {
            // handled by onDone
        }
    });

    // Listen for done
    window.electronAPI.onDone(async (msg) => {
        progressBar.style.width = '100%';
        progressPercent.textContent = '100%';
        progressEta.textContent = '完成';
        setRenderingState(false);

        if (msg.output) {
            displayImage(`rendered-file:///${msg.output}`);
        }
        progressSection.style.display = 'none';
    });

    // Listen for error
    window.electronAPI.onError((msg) => {
        setRenderingState(false);
        showError(msg.message || '发生未知错误');
        progressSection.style.display = 'none';
    });

    // Start
    const result = await window.electronAPI.startRender({
        width, height, samples, maxDepth, seed,
        output: '', // let backend pick temp path
        gpu,
        denoise,
    });

    if (result && result.error) {
        setRenderingState(false);
        showError(result.error);
        progressSection.style.display = 'none';
    }
});

// Cancel render
btnCancel.addEventListener('click', () => {
    if (isRendering) {
        window.electronAPI.cancelRender();
    }
});

// UI helpers
function setRenderingState(rendering) {
    isRendering = rendering;
    btnStart.disabled = rendering;
    btnStart.style.display = rendering ? 'none' : 'block';
    btnCancel.style.display = rendering ? 'block' : 'none';
    widthSelect.disabled = rendering;
    widthInput.disabled = rendering;
    aspectSelect.disabled = rendering;
    samplesSlider.disabled = rendering;
    depthSlider.disabled = rendering;
    seedInput.disabled = rendering;
    cpuRadio.disabled = rendering;
    gpuRadio.disabled = rendering;
    denoiseCheck.disabled = rendering;
    document.querySelectorAll('.preset').forEach(b => b.disabled = rendering);
}

function showError(msg) {
    errorDisplay.textContent = msg;
    errorDisplay.style.display = 'block';
}

function clearError() {
    errorDisplay.textContent = '';
    errorDisplay.style.display = 'none';
}

function displayImage(dataUrl) {
    placeholder.style.display = 'none';
    outputImage.style.display = 'block';
    outputImage.src = dataUrl;
}

// GPU detection
// GPU 可用性检测：更新设备名称、驱动、显存等状态显示
async function checkGpu() {
    gpuStatus.style.display = 'block';
    gpuStatus.textContent = '正在检查 GPU...';
    const result = await window.electronAPI.checkGpu();
    if (result && result.available) {
        gpuAvailable = true;
        const deviceName = result.device_name || 'NVIDIA GPU';
        const cc = result.compute_capability || '?';
        const driver = result.driver_version || '?';
        const vramGb = result.vram_mb ? (result.vram_mb / 1024).toFixed(1) : '?';
        const label = result.optix_available
            ? `GPU: ${deviceName} (CC ${cc}, ${vramGb} GB, 驱动 ${driver}) [RT Core + OptiX]`
            : `GPU: ${deviceName} (仅 CUDA，无 OptiX)`;
        gpuStatus.textContent = label;
        gpuStatus.className = 'hint gpu-ok';
        gpuLabel.style.opacity = '1';

        // Show warnings if any
        if (result.warnings && result.warnings.length > 0) {
            gpuStatus.textContent += '\n⚠ ' + result.warnings.join('\n⚠ ');
            gpuStatus.className = 'hint gpu-warn';
        }
    } else {
        gpuAvailable = false;
        gpuRadio.disabled = true;
        const errMsg = (result && result.error) ? result.error : 'GPU 不可用';
        gpuStatus.textContent = errMsg;
        gpuStatus.className = 'hint gpu-error';
        gpuLabel.style.opacity = '0.5';
        gpuLabel.title = 'GPU 不可用 — 请使用 --features cuda 编译或安装 CUDA 驱动';
    }
}

// Load calibration on startup
// 加载硬件性能校准数据
// 优先从缓存读取（版本号不匹配时自动失效），否则运行自测时
async function loadCalibration() {
    calStatus.textContent = '校准中...';
    try {
        // CPU calibration
        const existing = await window.electronAPI.readCalibration();
        if (existing && existing.pixel_samples_per_ms) {
            calibration = existing;
        } else {
            const result = await window.electronAPI.runCalibration(false);
            if (result && result.pixel_samples_per_ms) {
                calibration = result;
            } else if (result && result.fallback) {
                calibration = { pixel_samples_per_ms: result.fallback };
            } else {
                calibration = { pixel_samples_per_ms: 500 };
            }
        }

        // GPU calibration
        if (gpuAvailable) {
            const gpuExisting = await window.electronAPI.readGpuCalibration();
            if (gpuExisting && gpuExisting.pixel_samples_per_ms) {
                gpuCalibration = gpuExisting;
            } else {
                const gpuResult = await window.electronAPI.runCalibration(true);
                if (gpuResult && gpuResult.pixel_samples_per_ms) {
                    gpuCalibration = gpuResult;
                } else if (gpuResult && gpuResult.fallback) {
                    gpuCalibration = { pixel_samples_per_ms: gpuResult.fallback };
                } else {
                    gpuCalibration = { pixel_samples_per_ms: 50000 };
                }
            }
        }

        // Show calibration info
        const cpuSpeed = calibration ? calibration.pixel_samples_per_ms.toFixed(0) : '?';
        if (gpuCalibration && gpuCalibration.pixel_samples_per_ms) {
            const gpuSpeed = gpuCalibration.pixel_samples_per_ms.toFixed(0);
            calStatus.textContent = `CPU: ${cpuSpeed} px-samples/ms | GPU: ${gpuSpeed} px-samples/ms`;
        } else {
            calStatus.textContent = `已校准: ${cpuSpeed} px-samples/ms`;
        }
    } catch (err) {
        calStatus.textContent = '校准不可用';
        calibration = { pixel_samples_per_ms: 500 };
        if (gpuAvailable) {
            gpuCalibration = { pixel_samples_per_ms: 50000 };
        }
    }
    updateEstimates();
}

// Init
checkGpu().then(() => loadCalibration()).catch(() => {
    calStatus.textContent = '校准不可用';
    calibration = { pixel_samples_per_ms: 500 };
    updateEstimates();
});
updateEstimates();
