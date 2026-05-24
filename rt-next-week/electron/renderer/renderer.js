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
    const mb = (w * h * 6 * 1.3) / (1024 * 1024);
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
    if (ms < 1000) return '< 1s';
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
        widthInput.value = '1920';
        widthInput.focus();
        updateHeight();
    } else {
        widthInput.value = val;
        updateHeight();
    }
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
        showError('Width must be between 10 and 16384');
        return;
    }
    if (isNaN(samples) || samples < 1) {
        showError('Samples must be at least 1');
        return;
    }
    if (seedRaw !== '' && (isNaN(seed) || seed < 0)) {
        showError('Seed must be a non-negative integer or empty for random');
        return;
    }

    setRenderingState(true);
    clearError();
    progressSection.style.display = 'block';
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressEta.textContent = 'ETA: --';
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
                progressEta.textContent = 'ETA: ' + formatDuration(remaining);
            }
        } else if (msg.type === 'done') {
            // handled by onDone
        }
    });

    // Listen for done
    window.electronAPI.onDone(async (msg) => {
        progressBar.style.width = '100%';
        progressPercent.textContent = '100%';
        progressEta.textContent = 'Done';
        setRenderingState(false);

        if (msg.output) {
            const result = await window.electronAPI.getImageData(msg.output);
            if (result.dataUrl) {
                displayImage(result.dataUrl);
            }
        }
        progressSection.style.display = 'none';
    });

    // Listen for error
    window.electronAPI.onError((msg) => {
        setRenderingState(false);
        showError(msg.message || 'An unknown error occurred');
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
async function checkGpu() {
    gpuStatus.style.display = 'block';
    gpuStatus.textContent = 'Checking GPU...';
    const result = await window.electronAPI.checkGpu();
    if (result && result.available) {
        gpuAvailable = true;
        const deviceName = result.device_name || 'NVIDIA GPU';
        const label = result.optix_available
            ? `GPU: ${deviceName} (RT Core)`
            : `GPU: ${deviceName} (CUDA only, no OptiX)`;
        gpuStatus.textContent = label;
        gpuStatus.className = 'hint gpu-ok';
        gpuLabel.style.opacity = '1';
    } else {
        gpuAvailable = false;
        gpuRadio.disabled = true;
        const errMsg = (result && result.error) ? result.error : 'GPU not available';
        gpuStatus.textContent = errMsg;
        gpuStatus.className = 'hint gpu-error';
        gpuLabel.style.opacity = '0.5';
        gpuLabel.title = 'GPU unavailable — build with --features cuda or install CUDA driver';
    }
}

// Load calibration on startup
async function loadCalibration() {
    calStatus.textContent = 'Calibrating...';

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
            calibration = { pixel_samples_per_ms: 200 };
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
                gpuCalibration = { pixel_samples_per_ms: 10000 };
            }
        }
    }

    // Show calibration info
    const cpuSpeed = calibration ? calibration.pixel_samples_per_ms.toFixed(0) : '?';
    if (gpuCalibration && gpuCalibration.pixel_samples_per_ms) {
        const gpuSpeed = gpuCalibration.pixel_samples_per_ms.toFixed(0);
        calStatus.textContent = `CPU: ${cpuSpeed} px-ms/s | GPU: ${gpuSpeed} px-ms/s`;
    } else {
        calStatus.textContent = `Calibrated: ${cpuSpeed} px-samples/ms`;
    }
    updateEstimates();
}

// Init
checkGpu().then(() => loadCalibration());
updateEstimates();
