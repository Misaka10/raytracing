> [:arrow_down: 跳转到英文版 (Jump to English)](#rt-renderer--physically-based-monte-carlo-path-tracer-en)

# RT Renderer — 基于物理的蒙特卡洛路径追踪器

Peter Shirley《Ray Tracing: The Next Week》的 Rust 移植版，支持 NVIDIA OptiX GPU 加速与 Electron 桌面前端。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [命令行用法](#命令行用法)
- [架构](#架构)
  - [模块地图](#模块地图)
  - [CPU 渲染管线](#cpu-渲染管线)
  - [GPU 渲染管线](#gpu-渲染管线)
  - [场景构建](#场景构建)
  - [多重重要性采样 (MIS)](#多重重要性采样-mis)
  - [材质系统](#材质系统)
  - [PDF 系统](#pdf-系统)
- [构建系统](#构建系统)
- [GPU 诊断](#gpu-诊断)
- [Electron 前端](#electron-前端)
- [测试](#测试)
- [打包](#打包)
- [系统要求](#系统要求)

---

## 概述

RT Renderer 是一个基于物理的路径追踪器，实现了《Ray Tracing: The Next Week》中的技术，支持两种渲染后端：

| 后端 | 技术 | 性能 |
|------|------|------|
| CPU | Rust + rayon 并行 | ~200 px/sample/ms（16核） |
| GPU | CUDA + NVIDIA OptiX 9.1 + RT Core BVH | ~10,000 px/sample/ms（高端 NVIDIA GPU） |

两种后端在相同场景和随机种子下产生视觉上完全一致的输出（仅因 RNG 噪声有细微差异）。

核心特性：
- Cornell box 场景，包含箱体、玻璃球、面光源
- 多重重要性采样（50/50 BSDF + 光源混合）
- RT Core 硬件加速 BVH 遍历
- OptiX AI 降噪器（Tensor Core，可选）
- 重心坐标插值顶点法线，实现光滑球体
- 球体立体角采样用于 MIS
- 通过 `--seed` 实现确定性渲染
- Electron 桌面界面，支持进度可视化

---

## 快速开始

```sh
# 一键构建：CPU 编译 + Electron 打包
.\build.bat

# 一键构建：CPU + GPU 编译 + Electron 打包
.\build.bat --gpu

# CPU 渲染（默认：4K 分辨率，400 spp，75 次反弹）
cargo build --release
./target/release/rt-next-week.exe --output scene.png

# GPU 渲染（需要 CUDA 13.1 + OptiX 9.1 SDK）
cargo build --release --features cuda
./target/release/rt-next-week.exe --gpu --output scene.png

# GPU 诊断
./target/release/rt-next-week.exe --check-gpu

# 运行测试
cargo test --features cuda
```

---

## 命令行用法

```
rt-next-week.exe [选项]

选项：
  --width <N>         图像宽度（默认：3840）
  --height <N>        图像高度（默认：2160，或根据宽高比推导）
  --aspect-ratio <R>  宽高比（默认：1.777 = 16:9）
  --samples <N>       每像素采样数，分层 sqrt(N)xsqrt(N)（默认：400）
  --max-depth <N>     最大光线反弹次数（默认：75）
  --output <PATH>     输出 PNG 路径（默认：output.png）
  --seed <N>          随机种子，用于确定性渲染
  --gpu               使用 GPU（OptiX RT Core）后端
  --denoise           启用 OptiX AI 降噪器（仅 GPU）
  --json              输出 JSON 进度行用于 IPC（Electron 使用）
  --check-gpu         GPU 诊断：检测驱动、设备、OptiX 后退出
  --calibrate         自测时校准模式：抑制进度输出，输出吞吐量 JSON 到 stdout
```

---

## 架构

### 模块地图

```
src/
├── main.rs              — CLI 入口，Cornell box 场景构建
├── lib.rs               — 模块声明，feature-gated cuda 模块
├── camera.rs            — CPU 渲染循环（rayon）+ GPU 渲染入口 + PNG 输出
├── vec3.rs              — Vec3 (x,y,z)，Point3，Color 别名；SIMD f64 布局
├── ray.rs               — Ray { orig, dir, tm } — 参数化光线，支持运动模糊
├── rng.rs               — SmallRng (ChaCha12) PRNG 封装；线程局部便利接口
├── interval.rs          — [min, max] 区间运算（clamp, expand, surrounds）
├── aabb.rs              — 轴对齐包围盒
├── bvh.rs               — BVH 树（O(log n) 碰撞检测，空间中位数分割）
├── hittable.rs          — HitRecord，Hittable 枚举（所有几何体变体）
├── hittable_list.rs     — 扁平对象列表（场景根节点 + 光源列表）
├── sphere.rs            — 解析球体：碰撞、pdf_value、random（立体角）；支持静止 + 运动球体（运动模糊，通过 Ray 中心）
├── quad.rs              — 四边形：碰撞、pdf_value、random（均匀面积）
├── quad_box.rs          — make_box() 从最小/最大角点构建（6 个四边形）
├── constant_medium.rs   — 体积雾（随机距离采样）
├── material.rs          — Material 枚举 + scatter + scattering_pdf
├── texture.rs           — Texture 枚举（SolidColor, Checker, Image, Noise）
├── onb.rs               — 标准正交基（Hughes-Moeller 方法）；局部 ↔ 世界方向变换
├── pdf.rs               — PDF 枚举（Sphere, Cosine, Mixture）
├── perlin.rs            — 3D Perlin 噪声（hermite 平滑）+ turbulence（fBM）
├── color_io.rs          — linear_to_gamma，pixel_to_8bit/10bit/16bit 编码（含 NaN 保护）
└── cuda/
    ├── mod.rs           — CUDA 特性门控
    ├── optix.rs         — Rust FFI 到 optix_bridge C API + GPU 诊断
    ├── optix_bridge.h   — 桥接库 C 头文件
    ├── optix_bridge.cu  — C/CUDA 桥接：OptiX 初始化、BVH 构建、渲染、降噪
    ├── scene.rs         — GpuScene: Hittable -> 三角网格 + 顶点法线
    └── shaders/
        ├── common.h     — GpuFloat3, GpuMaterialData, CameraParams, LaunchParams
        ├── raygen.cu    — 光线生成着色器（MIS 路径追踪循环）
        ├── closesthit.cu — 命中着色器（重心坐标法线插值）
        ├── miss.cu      — 未命中着色器（背景颜色）
        ├── materials.h  — scatter_lambertian/metal/dielectric/isotropic
        ├── pdf.h        — 余弦 PDF 值，混合 PDF
        └── random.h     — XORShift128+ GPU 随机数生成器
```

### CPU 渲染管线

入口：`camera.rs` -> `Camera::render()`

```
对每个像素（rayon 并行）：
  对每个子像素采样（sqrt_spp x sqrt_spp）：
    1. Camera::get_ray() — 分层采样 + 散焦模糊
    2. ray_color() — 递归路径追踪
  累加，按 pixel_samples_scale 缩放
  通过 linear_to_gamma + pixel_to_10bit 转换为 10 位 gamma
保存为 16 位 PNG
```

`ray_color()` 递归逻辑：
1. 通过 BVH 进行碰撞检测：`world.hit(ray, [0.001, inf])` -> `HitRecord`
2. 未命中 -> 返回黑色（封闭的 Cornell box）
3. `material.emitted()` -> 发光贡献（仅 DiffuseLight 非零）
4. `material.scatter()` -> `ScatterRecord`：
   - **DiffuseLight**：返回 false -> 仅发光，路径终止
   - **Metal/Dielectric**：`skip_pdf=true` -> 直接用 `attenuation * ray_color(reflected_ray)` 递归
   - **Lambertian/Isotropic**：`skip_pdf=false` -> 进入下方 MIS 路径
5. MIS：50% 光源列表采样 / 50% BSDF 采样
6. `pdf_val = 0.5 * lights.pdf_value(scattered) + 0.5 * bsdf_pdf.value(scattered)`
7. 递归：`sample_color = ray_color(scattered_ray, depth-1)`
8. 返回：`emission + attenuation * scattering_pdf * sample_color / pdf_val`

### GPU 渲染管线

入口：`camera.rs` -> `Camera::render_gpu()`

**阶段 1 — 场景上传（CPU 端）：**
```
Hittable 树 -> GpuScene::from_world()
  ├── 细分球体：32x32 经纬网格 -> 2048 个三角形
  ├── 细分四边形：每个四边形 2 个三角形
  ├── 计算顶点法线（球体为解析法线，四边形为面法线）
  ├── 去重材质 -> GpuMaterialData 缓冲区
  └── 构建每个三角形的材质索引
```

**阶段 2 — GPU 设置（optix_bridge.cu）：**
```
上传顶点/法线/索引/材质 -> GPU 缓冲区
构建 RT Core BVH（硬件加速结构）
创建 OptiX 管线（raygen + closesthit + miss）
```

**阶段 3 — 光线生成（raygen.cu）：**
```
对每个像素：
  对每个子像素采样（sqrt_spp x sqrt_spp）：
    1. 分层相机光线 + 散焦模糊
    2. 路径追踪循环（最多 max_depth 次迭代）：
       a. optixTrace() -> RT Core BVH 遍历
       b. 未命中 -> 添加背景，退出
       c. 命中 -> 读取重心插值法线 + 材质
       d. DiffuseLight + 正面 -> 添加发光，退出
       e. scatter() -> ScatterResult
       f. skip_pdf（metal/dielectric） -> 直接递归
       g. MIS：50% BRDF / 50% 碰撞体采样
          - 碰撞体：50% 光源矩形 / 50% 球体立体角
       h. pdf_val = 0.5*BSDF + 0.5*hittable_pdf
       i. throughput *= attenuation * scattering_pdf / pdf_val
    3. 累加、缩放、钳制、写入帧缓冲
```

**阶段 4 — 降噪（可选，需要 Tensor Core）：**
```
OptiX AI HDR 降噪器 -> 降噪输出缓冲
```

**阶段 5 — 回读与保存：**
```
输出缓冲从 GPU 复制到 CPU
PNG 编码：linear_to_gamma -> 10 位 -> 16 位（与 CPU 一致）
```

### GPU 数据布局

主机 ↔ GPU 结构体布局必须完全一致。`common.h` 和 `optix_bridge.cu` 中的静态断言在编译时验证大小。

| 结构体 | 大小 | 字段 |
|--------|------|------|
| `GpuFloat3` | 12 字节 | `float x, y, z`（4 字节对齐） |
| `GpuMaterialData` | 36 字节 | `u32 mat_type` + `GpuFloat3 albedo` + `f32 fuzz` + `f32 ir` + `GpuFloat3 emission` |
| `CameraParams` | 148 字节 | lookfrom, lookat, vup [GpuFloat3]; vfov, aspect_ratio, defocus_angle, focus_dist [f32]; u, v, w, pixel00_loc, pixel_delta_u, pixel_delta_v, defocus_disk_u, defocus_disk_v [GpuFloat3] |

材质类型 ID：0=Lambertian, 1=Metal, 2=Dielectric, 3=DiffuseLight, 4=Isotropic。

### 场景构建

Cornell box 场景在 `main.rs` 中定义：

```
墙壁（5 个四边形）：
  左墙：  红色   (0.65, 0.05, 0.05)
  右墙：  绿色   (0.12, 0.45, 0.15)
  地板：  白色   (0.73, 0.73, 0.73)
  天花板：白色   (0.73, 0.73, 0.73)
  后墙：  白色   (0.73, 0.73, 0.73)

光源（四边形）：
  位置：(213, 554, 227)，大小 130x105
  材质：DiffuseLight，发光强度 (15, 15, 15)

箱体：
  6 个四边形，从 (0,0,0) 到 (165, 330, 165)，白色
  绕 Y 轴旋转 15 deg
  平移至 (265, 0, 295)

玻璃球：
  球心：(190, 90, 190)，半径：90
  材质：Dielectric，折射率 1.5

相机：
  位置：(278, 278, -800)，看向 (278, 278, 0)
  视场角：40 deg，无散焦模糊
```

光源采样列表（独立于世界几何体）：
- 光源四边形，带空（黑色）Lambertian 材质 — 用于方向采样
- 玻璃球，带空（黑色）Lambertian 材质 — 用于方向采样

### 多重重要性采样 (MIS)

路径追踪器使用 50/50 混合 MIS 来降低同时采样直接光照和间接反弹时的方差。

**MIS 权重计算：**

```
pdf_val = 0.5 * scattering_pdf + 0.5 * hittable_pdf

其中：
  scattering_pdf = cos(theta) / PI        （余弦加权半球）
  hittable_pdf   = 0.5 * light_pdf + 0.5 * sphere_pdf
  light_pdf      = dist^2 / (cos_light * area)  （光线命中光源矩形时）
  sphere_pdf     = 1.0 / solid_angle            （光线命中玻璃球时）

throughput *= attenuation * scattering_pdf / pdf_val
```

**策略选择（50/50）：**
- **策略 1（BSDF）**：从余弦加权半球采样方向。计算该方向的碰撞体 PDF。
- **策略 2（碰撞体）**：50% 采样光源矩形上的点，50% 通过立体角球体采样方向。

**球体立体角采样**（与 CPU `random_to_sphere` 一致）：
1. 从命中点指向球心的方向 -> 构建 ONB
2. 在 [cos_theta_max, 1] 范围内均匀采样 z，其中 cos_theta_max = sqrt(1 - r^2/d^2)
3. 在 [0, 2pi] 范围内均匀采样 phi
4. 通过 ONB 变换局部向量 (sqrt(1-z^2)*cos_phi, sqrt(1-z^2)*sin_phi, z)

### 材质系统

| 材质 | scatter() 返回 | skip_pdf | scattering_pdf | 策略 |
|------|---------------|----------|----------------|------|
| Lambertian | true | false | cos(theta)/pi | 余弦半球 |
| Metal | true | true | N/A | 完美/模糊反射 |
| Dielectric | true | true | N/A | 折射或 Schlick 反射 |
| DiffuseLight | **false** | N/A | N/A | 仅发光，路径终止 |
| Isotropic | true | false | 1/(4pi) | 均匀球体 |

**Metal 散射**（与 CPU 行为一致）：
```rust
reflected = reflect(ray).unit_vector() + fuzz * random_unit_vector()
// 不做归一化 — 模糊随距离增加
```

**Dielectric 散射：**
```rust
refraction_ratio = front_face ? 1.0/ir : ir
if cannot_refract || schlick_reflectance(cos_theta, ratio) > rand():
    reflect()      // 全内反射或概率反射
else:
    refract()      // Snell 定律
```

### PDF 系统

```
Pdf 枚举：
├── Sphere       -> value: 1/(4pi),         generate: random_unit_vector
├── Cosine(Onb)  -> value: cos(theta)/pi,   generate: ONB x random_cosine_direction
└── Mixture(p0,p1) -> value: p0与p1的平均值,  generate: 随机选择 p0 或 p1
```

### 纹理系统

纹理提供表面颜色变化。`Texture` 枚举有四种变体：

| 变体 | 字段 | 描述 |
|------|------|------|
| `SolidColor` | `Color` | 在所有 UV/点返回恒定颜色 |
| `Checker` | `inv_scale, even, odd` | 3D 程序化棋盘格（两个纹理在 3D 空间中交替） |
| `Image` | `data, width, height` | 通过 `image` crate 从文件加载的 2D 图像纹理 |
| `Noise` | `noise: Perlin, scale` | Perlin 噪声大理石条纹图案，可配置缩放 |

CPU 端 `BsdfPdf` 按材质构建：
- Lambertian -> `Pdf::Cosine(&normal)`
- Isotropic -> `Pdf::Sphere()`

`lights.pdf_value()` 对光源列表中所有光源（四边形 + 球体）取平均：
```rust
hittable_list.pdf_value() = avg(quad.pdf_value(), sphere.pdf_value())
```

---

## 构建系统

### Cargo + build.rs

标准 Rust 编译通过 Cargo 完成。当启用 `--features cuda` 时，`build.rs` 会：

1. 定位 CUDA Toolkit（nvcc）和 OptiX SDK（optix.h）
2. 使用 `std::thread::scope` + NVCC **并行编译** 3 个 `.cu` 着色器到 `.ptx`
3. 将 PTX ISA 版本从 9.1 降级到 8.5（CUDA 13.x 生成 9.1，但 OptiX 9.1 SDK 无法解析）
4. 将 `optix_bridge.cu` 编译为静态库（`.lib`）
5. 链接：`optix_bridge.lib`（静态）+ `cudart.lib` + `cuda.lib`（驱动动态库）

### 多线程编译

| 组件 | 并行方式 |
|------|---------|
| Cargo（rustc） | 每个 crate 并行（默认：CPU 核心数） |
| rustc 后端 | `codegen-units=1`（`.cargo/config.toml`）|
| NVCC 着色器 | `std::thread::scope` — 3 个着色器并发编译 |
| LTO | `thin` — 跨 crate 内联，无串行瓶颈 |

配置：`.cargo/config.toml`
```toml
[build]
rustflags = ["-C", "target-feature=+crt-static", "-C", "link-arg=/STACK:16777216"]

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = true
```

### PTX 架构

着色器使用 `--gpu-architecture=compute_75`（Turing）编译。PTX 是一种中间表示——NVIDIA 驱动程序在运行时将其 JIT 编译为实际的 GPU ISA。兼容从 Turing (RTX 20) 到 Blackwell (RTX 50) 的 GPU。

---

## GPU 诊断

```sh
./rt-next-week.exe --check-gpu
```

向 stdout 输出 JSON，`status` 可能的值：
- `"ok"` — GPU 完全可用
- `"no_optix"` — CUDA 驱动可用但 OptiX 初始化失败
- `"no_cuda_driver"` — 未检测到 CUDA 驱动
- `"not_compiled"` — 二进制未启用 CUDA 功能编译

```json
{
  "status": "ok",
  "cuda": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 4090",
    "driver_version": "13.2",
    "compute_capability": "8.9",
    "vram_mb": 16302,
    "device_count": 1,
    "warnings": null,
    "error": null
  },
  "optix": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 4090",
    "error": null
  }
}
```

自动警告：
- 驱动版本 < R560 -> "Driver too old: NVIDIA R560+ required for OptiX 9.x"
- 计算能力 < 7.5 -> "GPU may not run all shaders correctly"

---

## Electron 前端

位置：`electron/`

```
electron/
├── main.js          — Electron 主进程，IPC 处理器，进程管理
├── preload.js       — 上下文桥接：向渲染器暴露安全 API
├── package.json     — 依赖：electron, electron-builder
├── electron-builder.yml — 构建配置（便携版目标）
├── assets/
│   ├── icon.ico     — 应用图标
│   └── w700d1q75cms.jpg — 背景图
└── renderer/
    ├── index.html   — UI 布局
    ├── renderer.js  — 渲染逻辑：校准、进度、GPU 状态
    └── style.css    — 暗色主题样式
```

**IPC 通道：**

| 通道 | 方向 | 用途 |
|------|------|------|
| `check-gpu` | 渲染器 -> 主进程 | 运行 `--check-gpu`，返回解析后的 JSON |
| `read-calibration` | 渲染器 -> 主进程 | 加载缓存的 CPU 校准数据 |
| `read-gpu-calibration` | 渲染器 -> 主进程 | 加载缓存的 GPU 校准数据 |
| `run-calibration` | 渲染器 -> 主进程 | CPU: 320x180, 8 spp / GPU: 1280x720, 4 spp 基准渲染 |
| `start-render` | 渲染器 -> 主进程 | 开始完整分辨率渲染 |
| `cancel-render` | 渲染器 -> 主进程 | 终止正在运行的渲染进程 |
| `render-progress` | 主进程 -> 渲染器 | 进度更新（已完成/总像素数） |
| `render-done` | 主进程 -> 渲染器 | 渲染完成，附带输出路径 |
| `render-error` | 主进程 -> 渲染器 | 渲染错误，附带错误信息 |
| `render-log` | 主进程 -> 渲染器 | 原始 stderr 输出行 |

输出图片通过自定义 `rendered-file://` 协议加载（绕过 Node.js 512MB base64 数据 URL 限制）。
协议处理器从 URL 路径中剥离 `?t=...` 缓存破坏参数，使用 `Cache-Control: no-store` 防止重渲染后显示旧图。

**GPU 状态显示**（renderer.js 中）：
- 启动时通过 `--check-gpu` 检测 GPU 可用性
- 显示：设备名称、计算能力、显存、驱动版本
- 驱动过旧或 GPU 性能不足时显示警告
- 根据校准基准调整时间估算

---

## 测试

共 91 个单元测试，覆盖所有模块。运行方式：

```sh
cargo test --features cuda
```

主要测试分类：

| 模块 | 测试数 | 验证内容 |
|------|--------|---------|
| `vec3` | 17 | 算术运算、点积/叉积、单位向量、随机辅助函数、反射/折射 |
| `interval` | 7 | Contains、Surrounds、Clamp、Expand、add_offset |
| `aabb` | 5 | 构建、碰撞检测、包围盒合并、add_offset |
| `bvh` | 6 | 命中/未命中、最近命中顺序、包围盒覆盖、PDF 正值性 |
| `sphere` | 4 | 命中球心、未命中、包围盒、pdf_value |
| `quad` | 4 | 命中中心、平行未命中、边界、pdf_value |
| `camera` | 11 | 宽高比、高度、种子确定性、JSON 进度、gamma 一致性、零采样/极端宽高比防御 |
| `cuda::scene` | 13 | 球体/四边形细分、法线、材质转换、变换穿透、结构体大小/偏移 |
| `cuda::optix` | 1 | CameraParams 大小断言（148 字节） |
| `color_io` | 12 | Gamma 校正、所有位深（8/10/16 位）、NaN 保护、单调性 |
| `pdf` | 6 | Sphere/Cosine/Mixture 的值和生成、半球方向 |
| `perlin` | 3 | 噪声范围、确定性输出、turb 范围 |
| `ray` | 2 | at() 方法 |
| `material` | （隐式）| 通过相机和场景集成测试 |

---

## 打包

Electron 应用打包为便携版（免安装）ZIP：

```sh
cd electron
npm install
npm run dist          # 完整构建：electron-builder -> electron/dist-pkg/
```

手动重新打包（仅更新前端或二进制文件时）：
```sh
# 从 git 跟踪的源码构建 app.asar
mkdir _asar_src
cp electron/main.js electron/preload.js electron/package.json _asar_src/
cp -r electron/renderer _asar_src/
cd _asar_src && npx asar pack . ../electron/app.asar

# 更新 ZIP
python -c "
import zipfile
# 替换 ZIP 中的 resources/app.asar 和 resources/rt-next-week.exe
"
```

输出：`electron/dist-pkg/`（便携版 ZIP，约 110 MB）

内容：
- `RT Renderer.exe` — Electron 可执行文件
- `resources/app.asar` — 前端（JS、CSS、HTML）
- `resources/rt-next-week.exe` — Rust 渲染引擎
- `*.dll` — Chromium/Electron 运行时依赖

---

## 系统要求

| 组件 | 开发环境 | 运行环境 |
|------|---------|---------|
| Rust | 1.78+ | — |
| CUDA Toolkit | 13.1 | — |
| OptiX SDK | 9.1.0 | — |
| Visual Studio | 2022（Build Tools）| VC++ Redist 2015-2022 |
| Node.js | 20+（Electron 需要）| — |
| NVIDIA 驱动 | R560+ | R560+（包含 OptiX 9.x 运行时）|
| NVIDIA GPU | 计算能力 7.5+ | RTX 20 系列或更新 |
| 操作系统 | Windows 10/11 | Windows 10/11 |

---
> [:arrow_down: 跳转到中文版 (Jump to Chinese)](#rt-renderer--%E5%9F%BA%E4%BA%8E%E7%89%A9%E7%90%86%E7%9A%84%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E8%B7%AF%E5%BE%84%E8%BF%BD%E8%B8%AA%E5%99%A8)

# RT Renderer — Physically Based Monte Carlo Path Tracer

Rust port of Peter Shirley's *Ray Tracing: The Next Week* with NVIDIA OptiX GPU acceleration and Electron desktop frontend.

## Table of Contents

- [Overview](#overview-en)
- [Quick Start](#quick-start-en)
- [CLI Usage](#cli-usage-en)
- [Architecture](#architecture-en)
  - [Module Map](#module-map-en)
  - [CPU Rendering Pipeline](#cpu-rendering-pipeline-en)
  - [GPU Rendering Pipeline](#gpu-rendering-pipeline-en)
  - [Scene Construction](#scene-construction-en)
  - [Multiple Importance Sampling (MIS)](#multiple-importance-sampling-mis-en)
  - [Material System](#material-system-en)
  - [PDF System](#pdf-system-en)
- [Build System](#build-system-en)
- [GPU Diagnostics](#gpu-diagnostics-en)
- [Electron Frontend](#electron-frontend-en)
- [Testing](#testing-en)
- [Packaging](#packaging-en)
- [Requirements](#requirements-en)

---

## Overview

RT Renderer is a physically based path tracer implementing the techniques from *Ray Tracing: The Next Week*. It supports two rendering backends:

| Backend | Technology | Performance |
|---------|-----------|-------------|
| CPU | Rust + rayon parallel | ~200 px/sample/ms (16-core) |
| GPU | CUDA + NVIDIA OptiX 9.1 + RT Core BVH | ~10,000 px/sample/ms (high-end NVIDIA GPU) |

Both backends produce visually identical output given the same scene and seed (differ only by RNG noise).

Key features:
- Cornell box scene with box, glass sphere, area light
- Multiple Importance Sampling (50/50 BSDF + light mixture)
- RT Core hardware-accelerated BVH traversal
- OptiX AI denoiser (Tensor Core, optional)
- Barycentric-interpolated vertex normals for smooth spheres
- Solid-angle sphere sampling for MIS
- Deterministic rendering with `--seed`
- Electron desktop UI with progress visualization

---

## Quick Start

```sh
# One-click: CPU build + Electron package
.\build.bat

# One-click: CPU + GPU build + Electron package
.\build.bat --gpu

# CPU render (default: 4K, 400 spp, 75 bounces)
cargo build --release
./target/release/rt-next-week.exe --output scene.png

# GPU render (requires CUDA 13.1 + OptiX 9.1 SDK)
cargo build --release --features cuda
./target/release/rt-next-week.exe --gpu --output scene.png

# GPU diagnostics
./target/release/rt-next-week.exe --check-gpu

# Run tests
cargo test --features cuda
```

---

## CLI Usage

```
rt-next-week.exe [OPTIONS]

Options:
  --width <N>         Image width (default: 3840)
  --height <N>        Image height (default: 2160, or derived from aspect)
  --aspect-ratio <R>  Aspect ratio (default: 1.777 = 16:9)
  --samples <N>       Samples per pixel, stratified sqrt(N)xsqrt(N) (default: 400)
  --max-depth <N>     Maximum ray bounces (default: 75)
  --output <PATH>     Output PNG path (default: output.png)
  --seed <N>          Random seed for deterministic rendering
  --gpu               Use GPU (OptiX RT Core) backend
  --denoise           Enable OptiX AI denoiser (GPU only)
  --json              Output JSON progress lines for IPC (used by Electron)
  --check-gpu         GPU diagnostics: probe driver, device, OptiX, then exit
  --calibrate         Self-timed calibration: suppress progress, output throughput JSON to stdout
```

---

## Architecture

### Module Map

```
src/
├── main.rs              — CLI entry point, Cornell box scene construction
├── lib.rs               — Module declarations, feature-gated cuda module
├── camera.rs            — CPU render loop (rayon) + GPU render entry + PNG output
├── vec3.rs              — Vec3 (x,y,z), Point3, Color aliases; SIMD f64 layout
├── ray.rs               — Ray { orig, dir, tm } — parametric ray with motion blur support
├── rng.rs               — SmallRng (ChaCha12) PRNG wrapper; thread-local convenience
├── interval.rs          — [min, max] interval math (clamp, expand, surrounds)
├── aabb.rs              — Axis-Aligned Bounding Box
├── bvh.rs               — BVH tree (O(log n) hit test, spatial median split)
├── hittable.rs          — HitRecord, Hittable enum (all geometry variants)
├── hittable_list.rs     — Flat object list (scene root + light list)
├── sphere.rs            — Analytic sphere: hit, pdf_value, random (solid-angle); stationary + moving (motion blur via Ray center)
├── quad.rs              — Quadrilateral: hit, pdf_value, random (uniform area)
├── quad_box.rs          — make_box() from min/max corners (6 quads)
├── constant_medium.rs   — Volumetric fog (random distance sampling)
├── material.rs          — Material enum + scatter + scattering_pdf
├── texture.rs           — Texture enum (SolidColor, Checker, Image, Noise)
├── onb.rs               — Orthonormal basis (Hughes-Moeller method); local ↔ world direction transform
├── pdf.rs               — PDF enum (Sphere, Cosine, Mixture)
├── perlin.rs            — 3D Perlin noise (hermite smoothing) + turbulence (fBM)
├── color_io.rs          — linear_to_gamma, pixel_to_8bit/10bit/16bit encoding with NaN guard
└── cuda/
    ├── mod.rs           — CUDA feature gate
    ├── optix.rs         — Rust FFI to optix_bridge C API + GPU diagnostics
    ├── optix_bridge.h   — C header for bridge library
    ├── optix_bridge.cu  — C/CUDA bridge: OptiX init, BVH build, render, denoiser
    ├── scene.rs         — GpuScene: Hittable -> triangle mesh + vertex normals
    └── shaders/
        ├── common.h     — GpuFloat3, GpuMaterialData, CameraParams, LaunchParams
        ├── raygen.cu    — Ray generation shader (MIS path tracing loop)
        ├── closesthit.cu — Hit shader (barycentric normal interpolation)
        ├── miss.cu      — Miss shader (background color)
        ├── materials.h  — scatter_lambertian/metal/dielectric/isotropic
        ├── pdf.h        — Cosine PDF value, mixture PDF
        └── random.h     — XORShift128+ GPU RNG
```

### CPU Rendering Pipeline

Entry point: `camera.rs` -> `Camera::render()`

```
For each pixel (rayon parallel):
  For each sub-pixel sample (sqrt_spp x sqrt_spp):
    1. Camera::get_ray() — stratified sample + defocus blur
    2. ray_color() — recursive path tracing
  Accumulate, scale by pixel_samples_scale
  Convert to 10-bit gamma via linear_to_gamma + pixel_to_10bit
Save as 16-bit PNG
```

`ray_color()` recursive logic:
1. Hit test via BVH: `world.hit(ray, [0.001, inf])` -> `HitRecord`
2. Miss -> return black (enclosed Cornell box)
3. `material.emitted()` -> emission contribution (non-zero only for DiffuseLight)
4. `material.scatter()` -> `ScatterRecord`:
   - **DiffuseLight**: returns false -> only emission, path ends
   - **Metal/Dielectric**: `skip_pdf=true` -> recurse directly with `attenuation * ray_color(reflected_ray)`
   - **Lambertian/Isotropic**: `skip_pdf=false` -> MIS path below
5. MIS: 50% light-list sampling / 50% BSDF sampling
6. `pdf_val = 0.5 * lights.pdf_value(scattered) + 0.5 * bsdf_pdf.value(scattered)`
7. Recurse: `sample_color = ray_color(scattered_ray, depth-1)`
8. Return: `emission + attenuation * scattering_pdf * sample_color / pdf_val`

### GPU Rendering Pipeline

Entry point: `camera.rs` -> `Camera::render_gpu()`

**Phase 1 — Scene Upload (CPU side):**
```
Hittable tree -> GpuScene::from_world()
  ├── Tessellate spheres: 32x32 lat/lon grid -> 2048 triangles
  ├── Tessellate quads: 2 triangles per quad
  ├── Compute vertex normals (analytic for spheres, face normal for quads)
  ├── Deduplicate materials -> GpuMaterialData buffer
  └── Build per-triangle material index
```

**Phase 2 — GPU Setup (optix_bridge.cu):**
```
Upload vertices/normals/indices/materials -> GPU buffers
Build RT Core BVH (hardware acceleration structure)
Create OptiX pipeline (raygen + closesthit + miss)
```

**Phase 3 — Ray Generation (raygen.cu):**
```
For each pixel:
  For each sub-pixel sample (sqrt_spp x sqrt_spp):
    1. Stratified camera ray + defocus blur
    2. Path tracing loop (max_depth iterations):
       a. optixTrace() -> RT Core BVH traversal
       b. Miss -> add background, break
       c. Hit -> read barycentric-interpolated normal + material
       d. DiffuseLight + front_face -> add emission, break
       e. scatter() -> ScatterResult
       f. skip_pdf (metal/dielectric) -> direct recursion
       g. MIS: 50% BRDF / 50% hittable sampling
          - Hittable: 50% light rectangle / 50% sphere solid-angle
       h. pdf_val = 0.5*BSDF + 0.5*hittable_pdf
       i. throughput *= attenuation * scattering_pdf / pdf_val
    3. Accumulate, scale, clamp, write to framebuffer
```

**Phase 4 — Denoiser (optional, Tensor Core):**
```
OptiX AI HDR denoiser -> denoised output buffer
```

**Phase 5 — Readback & Save:**
```
Copy output buffer GPU -> CPU
PNG encoding: linear_to_gamma -> 10-bit -> 16-bit (same as CPU)
```

### GPU Data Layout

Host ↔ GPU struct layout must match exactly. Static assertions in `common.h` and `optix_bridge.cu` verify sizes at compile time.

| Struct | Size | Fields |
|--------|------|--------|
| `GpuFloat3` | 12 bytes | `float x, y, z` (4-byte alignment) |
| `GpuMaterialData` | 36 bytes | `u32 mat_type` + `GpuFloat3 albedo` + `f32 fuzz` + `f32 ir` + `GpuFloat3 emission` |
| `CameraParams` | 148 bytes | lookfrom, lookat, vup [GpuFloat3]; vfov, aspect_ratio, defocus_angle, focus_dist [f32]; u, v, w, pixel00_loc, pixel_delta_u, pixel_delta_v, defocus_disk_u, defocus_disk_v [GpuFloat3] |

Material type IDs: 0=Lambertian, 1=Metal, 2=Dielectric, 3=DiffuseLight, 4=Isotropic.

### Scene Construction

The Cornell box scene is defined in `main.rs`:

```
Walls (5 quads):
  Left:   red    (0.65, 0.05, 0.05)
  Right:  green  (0.12, 0.45, 0.15)
  Floor:  white  (0.73, 0.73, 0.73)
  Ceiling: white (0.73, 0.73, 0.73)
  Back:   white  (0.73, 0.73, 0.73)

Light (quad):
  Position: (213, 554, 227), size 130x105
  Material: DiffuseLight, emission (15, 15, 15)

Box:
  6 quads from (0,0,0) to (165, 330, 165), white
  Rotated 15 deg around Y axis
  Translated to (265, 0, 295)

Glass sphere:
  Center: (190, 90, 190), radius: 90
  Material: Dielectric, IOR 1.5

Camera:
  Position: (278, 278, -800), looking at (278, 278, 0)
  FOV: 40 deg, no defocus blur
```

Light sampling list (separate from world geometry):
- Light quad with empty (black) Lambertian material — for direction sampling
- Glass sphere with empty (black) Lambertian material — for direction sampling

### Multiple Importance Sampling (MIS)

The path tracer uses 50/50 mixture MIS to reduce variance when sampling both direct lighting and indirect bounces.

**MIS weight calculation:**

```
pdf_val = 0.5 * scattering_pdf + 0.5 * hittable_pdf

where:
  scattering_pdf = cos(theta) / PI        (cosine-weighted hemisphere)
  hittable_pdf   = 0.5 * light_pdf + 0.5 * sphere_pdf
  light_pdf      = dist^2 / (cos_light * area)   (if ray hits light rect)
  sphere_pdf     = 1.0 / solid_angle             (if ray hits glass sphere)

throughput *= attenuation * scattering_pdf / pdf_val
```

**Strategy selection (50/50):**
- **Strategy 1 (BSDF)**: Sample direction from cosine-weighted hemisphere. Compute hittable PDF for that direction.
- **Strategy 2 (Hittable)**: 50% sample point on light rectangle, 50% sample direction via solid-angle sphere sampling.

**Sphere solid-angle sampling** (matching CPU `random_to_sphere`):
1. Direction from hit point toward sphere center -> build ONB
2. Sample z uniformly in [cos_theta_max, 1] where cos_theta_max = sqrt(1 - r^2/d^2)
3. Sample phi uniformly in [0, 2pi]
4. Transform local (sqrt(1-z^2)*cos_phi, sqrt(1-z^2)*sin_phi, z) via ONB

### Material System

| Material | scatter() returns | skip_pdf | scattering_pdf | Strategy |
|----------|-------------------|----------|----------------|----------|
| Lambertian | true | false | cos(theta)/pi | Cosine hemisphere |
| Metal | true | true | N/A | Perfect/fuzzed reflection |
| Dielectric | true | true | N/A | Refraction or Schlick reflection |
| DiffuseLight | **false** | N/A | N/A | Only emission, path terminates |
| Isotropic | true | false | 1/(4pi) | Uniform sphere |

**Metal scatter** (CPU behavior):
```rust
reflected = reflect(ray).unit_vector() + fuzz * random_unit_vector()
// NOT normalized — blur increases with distance
```

**Dielectric scatter:**
```rust
refraction_ratio = front_face ? 1.0/ir : ir
if cannot_refract || schlick_reflectance(cos_theta, ratio) > rand():
    reflect()      // total internal reflection or probabilistic
else:
    refract()      // Snell's law
```

### PDF System

```
Pdf enum:
├── Sphere       -> value: 1/(4pi),         generate: random_unit_vector
├── Cosine(Onb)  -> value: cos(theta)/pi,   generate: ONB x random_cosine_direction
└── Mixture(p0,p1) -> value: avg of p0,p1,  generate: random pick p0 or p1
```

### Texture System

Textures provide surface color variation. The `Texture` enum has four variants:

| Variant | Fields | Description |
|---------|--------|-------------|
| `SolidColor` | `Color` | Constant color at all UV/points |
| `Checker` | `inv_scale, even, odd` | 3D procedural checkerboard (two textures alternating in 3D space) |
| `Image` | `data, width, height` | 2D image texture loaded from file via the `image` crate |
| `Noise` | `noise: Perlin, scale` | Perlin noise marble-like pattern with configurable scale |

The CPU `BsdfPdf` is constructed per material:
- Lambertian -> `Pdf::Cosine(&normal)`
- Isotropic -> `Pdf::Sphere()`

`lights.pdf_value()` averages over all lights in the list (quad + sphere):
```rust
hittable_list.pdf_value() = avg(quad.pdf_value(), sphere.pdf_value())
```

---

## Build System

### Cargo + build.rs

Normal Rust compilation via Cargo. When `--features cuda` is enabled, `build.rs`:

1. Locates CUDA Toolkit (nvcc) and OptiX SDK (optix.h)
2. Compiles 3 `.cu` shaders to `.ptx` **in parallel** using `std::thread::scope` + NVCC
3. Patches PTX ISA version from 9.1 -> 8.5 (CUDA 13.x generates 9.1 which OptiX 9.1 SDK rejects)
4. Compiles `optix_bridge.cu` to a static library (`.lib`)
5. Links: `optix_bridge.lib` (static) + `cudart.lib` + `cuda.lib` (dynamic from driver)

### Multi-threaded compilation

| Component | Parallelism |
|-----------|-------------|
| Cargo (rustc) | Per-crate parallelism (default: CPU cores) |
| rustc backend | `codegen-units=1` (`.cargo/config.toml`) |
| NVCC shaders | `std::thread::scope` — 3 shaders compiled concurrently |
| LTO | `thin` — cross-crate inlining without serial bottleneck |

Config: `.cargo/config.toml`
```toml
[build]
rustflags = ["-C", "target-feature=+crt-static", "-C", "link-arg=/STACK:16777216"]

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = true
```

### PTX architecture

Shaders compiled with `--gpu-architecture=compute_75` (Turing). PTX is an intermediate representation — the NVIDIA driver JIT-compiles it to the actual GPU ISA at runtime. Compatible with Turing (RTX 20) through Blackwell (RTX 50) GPUs.

---

## GPU Diagnostics

```sh
./rt-next-week.exe --check-gpu
```

Outputs JSON to stdout. Possible `status` values:
- `"ok"` — GPU fully operational
- `"no_optix"` — CUDA driver available but OptiX init failed
- `"no_cuda_driver"` — No CUDA driver detected
- `"not_compiled"` — Binary built without CUDA feature

```json
{
  "status": "ok",
  "cuda": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 4090",
    "driver_version": "13.2",
    "compute_capability": "8.9",
    "vram_mb": 16302,
    "device_count": 1,
    "warnings": null,
    "error": null
  },
  "optix": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 4090",
    "error": null
  }
}
```

Automatic warnings:
- Driver < R560 -> "Driver too old: NVIDIA R560+ required for OptiX 9.x"
- Compute capability < 7.5 -> "GPU may not run all shaders correctly"

---

## Electron Frontend

Location: `electron/`

```
electron/
├── main.js          — Electron main process, IPC handlers, spawn management
├── preload.js       — Context bridge: exposes safe API to renderer
├── package.json     — Dependencies: electron, electron-builder
├── electron-builder.yml — Build config (portable target)
├── assets/
│   ├── icon.ico     — App icon
│   └── w700d1q75cms.jpg — Background image
└── renderer/
    ├── index.html   — UI layout
    ├── renderer.js  — Render logic: calibration, progress, GPU status
    └── style.css    — Dark theme styling
```

**IPC channels:**

| Channel | Direction | Purpose |
|---------|-----------|---------|
| `check-gpu` | renderer -> main | Run `--check-gpu`, return parsed JSON |
| `read-calibration` | renderer -> main | Load cached CPU calibration |
| `read-gpu-calibration` | renderer -> main | Load cached GPU calibration |
| `run-calibration` | renderer -> main | CPU: 320x180 at 8 spp / GPU: 1280x720 at 4 spp benchmark |
| `start-render` | renderer -> main | Start full-resolution render |
| `cancel-render` | renderer -> main | Kill running render process |
| `render-progress` | main -> renderer | Progress update (completed/total pixels) |
| `render-done` | main -> renderer | Render complete with output path |
| `render-error` | main -> renderer | Render error with message |
| `render-log` | main -> renderer | Raw stderr output lines |

Output images are served via custom `rendered-file://` protocol (bypasses Node.js 512MB string limit on base64 data URLs).
Protocol handler strips `?t=...` cache buster from URL path, serves with `Cache-Control: no-store` to prevent stale image display after re-render.

**GPU status display** (in renderer.js):
- Checks GPU availability on startup via `--check-gpu`
- Shows: device name, compute capability, VRAM, driver version
- Shows warnings if driver is old or GPU capability is low
- Adjusts time estimate based on calibration benchmark

---

## Testing

91 unit tests across all modules. Run with:

```sh
cargo test --features cuda
```

Key test categories:

| Module | Tests | What they verify |
|--------|-------|-----------------|
| `vec3` | 17 | Arithmetic, dot/cross, unit vector, RNG helpers, reflect/refract |
| `interval` | 7 | Contains, surrounds, clamp, expand, add_offset |
| `aabb` | 5 | Construction, hit test, box union, add_offset |
| `bvh` | 6 | Hit/miss, closest-hit ordering, bounding box coverage, PDF positivity |
| `sphere` | 4 | Hit center, miss, bbox, pdf_value |
| `quad` | 4 | Hit center, parallel miss, bounds, pdf_value |
| `camera` | 11 | Aspect ratio, height, seed determinism, JSON progress, gamma consistency, clamp guards |
| `cuda::scene` | 13 | Sphere/quads tessellation, normals, material conversion, transform penetration, struct sizes/offsets |
| `cuda::optix` | 1 | CameraParams size assertion (148 bytes) |
| `color_io` | 12 | Gamma correction, all bit depths (8/10/16-bit), NaN guard, monotonicity |
| `pdf` | 6 | Sphere/cosine/mixture value and generate, hemisphere direction |
| `perlin` | 3 | Noise range, deterministic output, turb range |
| `ray` | 2 | at() method |
| `material` | (implicit) | Via camera + scene integration tests |

---

## Packaging

The Electron app is packaged as a portable (no-install) ZIP:

```sh
cd electron
npm install
npm run dist          # Full build: electron-builder -> electron/dist-pkg/
```

Manual repack (for updating only frontend or binary):
```sh
# Build app.asar from git-tracked source
mkdir _asar_src
cp electron/main.js electron/preload.js electron/package.json _asar_src/
cp -r electron/renderer _asar_src/
cd _asar_src && npx asar pack . ../electron/app.asar

# Update ZIP
python -c "
import zipfile
# Replace resources/app.asar and resources/rt-next-week.exe in ZIP
"
```

Output: `electron/dist-pkg/` (portable ZIP, ~110 MB)

Contents:
- `RT Renderer.exe` — Electron executable
- `resources/app.asar` — Frontend (JS, CSS, HTML)
- `resources/rt-next-week.exe` — Rust rendering engine
- `*.dll` — Chromium/Electron runtime dependencies

---

## Requirements

| Component | Development | Runtime |
|-----------|-------------|---------|
| Rust | 1.78+ | — |
| CUDA Toolkit | 13.1 | — |
| OptiX SDK | 9.1.0 | — |
| Visual Studio | 2022 (Build Tools) | VC++ Redist 2015-2022 |
| Node.js | 20+ (for Electron) | — |
| NVIDIA Driver | R560+ | R560+ (includes OptiX 9.x runtime) |
| NVIDIA GPU | Any CC 7.5+ | RTX 20-series or newer |
| OS | Windows 10/11 | Windows 10/11 |

