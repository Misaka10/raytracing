const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    startRender: (config) => ipcRenderer.invoke('start-render', config),
    cancelRender: () => ipcRenderer.send('cancel-render'),
    onProgress: (callback) => {
        ipcRenderer.on('render-progress', (_event, msg) => callback(msg));
    },
    onDone: (callback) => {
        ipcRenderer.on('render-done', (_event, msg) => callback(msg));
    },
    onError: (callback) => {
        ipcRenderer.on('render-error', (_event, msg) => callback(msg));
    },
    onLog: (callback) => {
        ipcRenderer.on('render-log', (_event, msg) => callback(msg));
    },
    runCalibration: (useGpu) => ipcRenderer.invoke('run-calibration', useGpu),
    checkGpu: () => ipcRenderer.invoke('check-gpu'),
    readCalibration: () => ipcRenderer.invoke('read-calibration'),
    readGpuCalibration: () => ipcRenderer.invoke('read-gpu-calibration'),
    removeAllListeners: () => {
        ipcRenderer.removeAllListeners('render-progress');
        ipcRenderer.removeAllListeners('render-done');
        ipcRenderer.removeAllListeners('render-error');
        ipcRenderer.removeAllListeners('render-log');
    },
});
