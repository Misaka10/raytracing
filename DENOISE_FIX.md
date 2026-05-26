# Tensor Core 降噪修复总结

## 版本

v1.1.3 → v1.1.4

## 问题现象

OptiX AI HDR 降噪器（Tensor Core 加速）输出图像降噪效果差：边缘模糊、纹理丢失、颜色偏暗。

## 根因分析

经过代码审查发现三个独立问题：

### 1. 缺少引导缓冲区（主要问题）

**位置**：`src/cuda/optix_bridge.cu` 第 788-790 行

```c
opts.guideAlbedo = 0;
opts.guideNormal = 0;
```

降噪器仅使用颜色信息（3 通道 RGB），未提供反照率 (albedo) 和法线 (normal) 引导。引导缓冲区的作用：

| 引导类型 | 作用 | 缺失后果 |
|---------|------|---------|
| Albedo | 告知降噪器表面反射率，区分纹理细节与噪点 | 墙壁颜色纹理被过度模糊 |
| Normal | 告知降噪器几何边界位置，保持边缘锐利 | 盒子边缘、球体轮廓出现重影 |

### 2. HDR 强度硬编码

**位置**：`src/cuda/optix_bridge.cu` 第 857 行

```c
params.hdrIntensity = 1.0f;
```

OptiX 降噪器内部会将 HDR 值归一化后送入神经网络。硬编码 1.0 禁用了自动强度计算，而 Cornell Box 光源强度为 15.0，导致 HDR 范围被压缩，整体偏暗。

### 3. 降噪结果未下载到主机（隐蔽 bug）

**位置**：`src/camera.rs` 与 `optix_bridge.cu` 的调用时序

```
render() 过程:
  1. GPU 渲染 → d_output
  2. cuMemcpyDtoH → 下载到主机 output 缓冲区   ← 此时数据未降噪
  3. denoise()
       a. 降噪 d_output → d_denoisedOutput
       b. cuMemcpyDtoD → d_output              ← 仅写回 GPU
       c. 缺少 cuMemcpyDtoH！                    ← 主机 output 仍是旧数据
  4. save_png_gpu(output)                       ← 保存的是未降噪图像
```

`bridge.render()` 在降噪前就把数据下载到了主机，之后 `bridge.denoise()` 只修改了 GPU 端数据，导致保存的 PNG 实际是未降噪版本。现场表现为降噪与非降噪输出 MD5 完全一致。

## 修复方案

### 改动文件清单

| 文件 | 改动 |
|------|------|
| `src/cuda/shaders/common.h` | `LaunchParams` 增加 `albedo_buffer`、`guide_normal_buffer` 两个指针 |
| `src/cuda/shaders/raygen.cu` | 首次命中时根据材质类型写入引导数据 |
| `src/cuda/optix_bridge.cu` | 7 处改动（见下） |
| `src/cuda/optix_bridge.h` | `denoise()` 签名增加 `float* output` 参数 |
| `src/cuda/optix.rs` | Rust FFI 绑定同步更新 |
| `src/camera.rs` | 调用点传递 `output` 缓冲区 |
| `Cargo.toml` | 版本号 1.1.3 → 1.1.4 |
| `electron/package.json` | 版本号 1.1.3 → 1.1.4 |

### optix_bridge.cu 的 7 处改动

| 改动点 | 说明 |
|--------|------|
| `GpuLaunchParams` 结构体 | 增加 `albedo_buffer`、`guide_normal_buffer` 指针 |
| `OptiXBridge` 结构体 | 增加 `d_albedoBuffer`、`d_guideNormalBuffer` 设备指针 + `denoiserWidth/Height` 跟踪字段 |
| `init()` | 零初始化新字段 |
| `destroy()` | 释放两个新缓冲区 |
| `create_pipeline()` | 分配 albedo/normal 缓冲区 + `cuMemsetD8` 清零 |
| `render()` | 设置 `params.albedo_buffer`、`params.guide_normal_buffer` |
| `denoise()` | **重写**：引导启用 + 分辨率变更检测 + 移除硬编码 hdrIntensity + 引导层创建 + DtoH 下载 |

### 引导缓冲区数据流

```
raygen.cu (depth==0, sj==0, si==0):
  ├── 根据材质类型计算 albedo
  │     Lambertian/Metal/Isotropic → mat.albedo
  │     Dielectric/DiffuseLight   → (1, 1, 1)
  ├── 写入 albedo_buffer[pixel_idx]
  └── 写入 guide_normal_buffer[pixel_idx] (world-space)

optix_bridge_denoise():
  ├── OptixDenoiserGuideLayer { albedo, normal }
  ├── 9 通道输入: 3 颜色 + 3 albedo + 3 normal
  └── 降噪后 cuMemcpyDtoH 下载到主机 output
```

## 验证结果

| 验证项 | 结果 |
|--------|------|
| 编译 (`--features cuda`) | 通过 |
| 88 个单元测试 | 全部通过 |
| GPU 诊断 (`--check-gpu`) | RTX 5080, CUDA 13.2, OptiX 可用 |
| 降噪输入通道 | 3 → **9** (颜色+反照率+法线) |
| 降噪前后文件大小 | 756KB → 462KB (-39%) |
| 降噪前后 MD5 | 不同（确认实际生效） |
| Electron 打包 | `RT Renderer 1.1.4.exe` (75MB) |
