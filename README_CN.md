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
| CPU | Rust + rayon 并行 | ~200 px·sample/ms（16核） |
| GPU | CUDA + NVIDIA OptiX 9.1 + RT Core BVH | ~10,000 px·sample/ms（RTX 5080） |

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
  --samples <N>       每像素采样数，分层 sqrt(N)×sqrt(N)（默认：400）
  --max-depth <N>     最大光线反弹次数（默认：75）
  --output <PATH>     输出 PNG 路径（默认：output.png）
  --seed <N>          随机种子，用于确定性渲染
  --gpu               使用 GPU（OptiX RT Core）后端
  --denoise           启用 OptiX AI 降噪器（仅 GPU）
  --json              输出 JSON 进度行用于 IPC（Electron 使用）
  --check-gpu         GPU 诊断：检测驱动、设备、OptiX 后退出
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
├── ray.rs               — Ray { origin, direction, time }
├── interval.rs          — [min, max] 区间运算（clamp, expand, surrounds）
├── aabb.rs              — 轴对齐包围盒
├── bvh.rs               — BVH 树（O(log n) 碰撞检测，空间中位数分割）
├── hittable.rs          — HitRecord，Hittable 枚举（所有几何体变体）
├── hittable_list.rs     — 扁平对象列表（场景根节点 + 光源列表）
├── sphere.rs            — 解析球体：碰撞、pdf_value、random（立体角）
├── quad.rs              — 四边形：碰撞、pdf_value、random（均匀面积）
├── quad_box.rs          — make_box() 从最小/最大角点构建（6 个四边形）
├── constant_medium.rs   — 体积雾（随机距离采样）
├── material.rs          — Material 枚举 + scatter + scattering_pdf
├── texture.rs           — Texture 枚举（SolidColor）
├── onb.rs               — 标准正交基，用于余弦半球采样
├── pdf.rs               — PDF 枚举（Sphere, Cosine, Mixture）
├── perlin.rs            — 3D Perlin 噪声 + turbulence
├── color_io.rs          — linear_to_gamma，pixel_to_10bit/16bit 编码
└── cuda/
    ├── mod.rs           — CUDA 特性门控
    ├── optix.rs         — Rust FFI 到 optix_bridge C API + GPU 诊断
    ├── optix_bridge.h   — 桥接库 C 头文件
    ├── optix_bridge.cu  — C/CUDA 桥接：OptiX 初始化、BVH 构建、渲染、降噪
    ├── scene.rs         — GpuScene: Hittable → 三角网格 + 顶点法线
    └── shaders/
        ├── common.h     — GpuFloat3, GpuMaterialData, CameraParams, LaunchParams
        ├── raygen.cu    — 光线生成着色器（MIS 路径追踪循环）
        ├── closesthit.cu — 命中着色器（重心坐标法线插值）
        ├── miss.cu      — 未命中着色器（背景颜色）
        ├── materials.h  — scatter_lambertian/metal/dielectric/isotropic
        ├── pdf.h        — 余弦 PDF 值，混合 PDF
        └── random.h     — 基于 PCG 的 GPU 随机数生成器
```

### CPU 渲染管线

入口：`camera.rs` → `Camera::render()`

```
对每个像素（rayon 并行）：
  对每个子像素采样（sqrt_spp × sqrt_spp）：
    1. Camera::get_ray() — 分层采样 + 散焦模糊
    2. ray_color() — 递归路径追踪
  累加，按 pixel_samples_scale 缩放
  通过 linear_to_gamma + pixel_to_10bit 转换为 10 位 gamma
保存为 16 位 PNG
```

`ray_color()` 递归逻辑：
1. 通过 BVH 进行碰撞检测：`world.hit(ray, [0.001, ∞])` → `HitRecord`
2. 未命中 → 返回黑色（封闭的 Cornell box）
3. `material.emitted()` → 发光贡献（仅 DiffuseLight 非零）
4. `material.scatter()` → `ScatterRecord`：
   - **DiffuseLight**：返回 false → 仅发光，路径终止
   - **Metal/Dielectric**：`skip_pdf=true` → 直接用 `attenuation * ray_color(reflected_ray)` 递归
   - **Lambertian/Isotropic**：`skip_pdf=false` → 进入下方 MIS 路径
5. MIS：50% 光源列表采样 / 50% BSDF 采样
6. `pdf_val = 0.5 * lights.pdf_value(scattered) + 0.5 * bsdf_pdf.value(scattered)`
7. 递归：`sample_color = ray_color(scattered_ray, depth-1)`
8. 返回：`emission + attenuation * scattering_pdf * sample_color / pdf_val`

### GPU 渲染管线

入口：`camera.rs` → `Camera::render_gpu()`

**阶段 1 — 场景上传（CPU 端）：**
```
Hittable 树 → GpuScene::from_world()
  ├── 细分球体：32×32 经纬网格 → 2048 个三角形
  ├── 细分四边形：每个四边形 2 个三角形
  ├── 计算顶点法线（球体为解析法线，四边形为面法线）
  ├── 去重材质 → GpuMaterialData 缓冲区
  └── 构建每个三角形的材质索引
```

**阶段 2 — GPU 设置（optix_bridge.cu）：**
```
上传顶点/法线/索引/材质 → GPU 缓冲区
构建 RT Core BVH（硬件加速结构）
创建 OptiX 管线（raygen + closesthit + miss）
```

**阶段 3 — 光线生成（raygen.cu）：**
```
对每个像素：
  对每个子像素采样（sqrt_spp × sqrt_spp）：
    1. 分层相机光线 + 散焦模糊
    2. 路径追踪循环（最多 max_depth 次迭代）：
       a. optixTrace() → RT Core BVH 遍历
       b. 未命中 → 添加背景，退出
       c. 命中 → 读取重心插值法线 + 材质
       d. DiffuseLight + 正面 → 添加发光，退出
       e. scatter() → ScatterResult
       f. skip_pdf（metal/dielectric） → 直接递归
       g. MIS：50% BRDF / 50% 碰撞体采样
          - 碰撞体：50% 光源矩形 / 50% 球体立体角
       h. pdf_val = 0.5*BSDF + 0.5*hittable_pdf
       i. throughput *= attenuation * scattering_pdf / pdf_val
    3. 累加、缩放、钳制、写入帧缓冲
```

**阶段 4 — 降噪（可选，需要 Tensor Core）：**
```
OptiX AI HDR 降噪器 → 降噪输出缓冲
```

**阶段 5 — 回读与保存：**
```
输出缓冲从 GPU 复制到 CPU
PNG 编码：linear_to_gamma → 10 位 → 16 位（与 CPU 一致）
```

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
  位置：(213, 554, 227)，大小 130×105
  材质：DiffuseLight，发光强度 (15, 15, 15)

箱体：
  6 个四边形，从 (0,0,0) 到 (165, 330, 165)，白色
  绕 Y 轴旋转 15°
  平移至 (265, 0, 295)

玻璃球：
  球心：(190, 90, 190)，半径：90
  材质：Dielectric，折射率 1.5

相机：
  位置：(278, 278, -800)，看向 (278, 278, 0)
  视场角：40°，无散焦模糊
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
  light_pdf      = dist² / (cos_light * area)  （光线命中光源矩形时）
  sphere_pdf     = 1.0 / solid_angle            （光线命中玻璃球时）

throughput *= attenuation * scattering_pdf / pdf_val
```

**策略选择（50/50）：**
- **策略 1（BSDF）**：从余弦加权半球采样方向。计算该方向的碰撞体 PDF。
- **策略 2（碰撞体）**：50% 采样光源矩形上的点，50% 通过立体角球体采样方向。

**球体立体角采样**（与 CPU `random_to_sphere` 一致）：
1. 从命中点指向球心的方向 → 构建 ONB
2. 在 [cos_θ_max, 1] 范围内均匀采样 z，其中 cos_θ_max = √(1 − r²/d²)
3. 在 [0, 2π] 范围内均匀采样 φ
4. 通过 ONB 变换局部向量 (√(1−z²)·cos φ, √(1−z²)·sin φ, z)

### 材质系统

| 材质 | scatter() 返回 | skip_pdf | scattering_pdf | 策略 |
|------|---------------|----------|----------------|------|
| Lambertian | true | false | cos(θ)/π | 余弦半球 |
| Metal | true | true | N/A | 完美/模糊反射 |
| Dielectric | true | true | N/A | 折射或 Schlick 反射 |
| DiffuseLight | **false** | N/A | N/A | 仅发光，路径终止 |
| Isotropic | true | false | 1/(4π) | 均匀球体 |

**Metal 散射**（与 CPU 行为一致）：
```rust
reflected = reflect(ray).unit_vector() + fuzz * random_unit_vector()
// 不做归一化 — 模糊随距离增加
```

**Dielectric 散射：**
```rust
refraction_ratio = front_face ? 1.0/ir : ir
if cannot_refract || schlick_reflectance(cos_θ, ratio) > rand():
    reflect()      // 全内反射或概率反射
else:
    refract()      // Snell 定律
```

### PDF 系统

```
Pdf 枚举：
├── Sphere       → value: 1/(4π),          generate: random_unit_vector
├── Cosine(Onb)  → value: cos(θ)/π,        generate: ONB × random_cosine_direction
└── Mixture(p0,p1) → value: p0与p1的平均值,  generate: 随机选择 p0 或 p1
```

CPU 端 `BsdfPdf` 按材质构建：
- Lambertian → `Pdf::Cosine(&normal)`
- Isotropic → `Pdf::Sphere()`

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
| rustc 后端 | `codegen-units=16`（`.cargo/config.toml`） |
| NVCC 着色器 | `std::thread::scope` — 3 个着色器并发编译 |
| LTO | 已禁用（`lto=false`）— 避免串行链接瓶颈 |

配置：`.cargo/config.toml`
```toml
[build]
rustflags = ["-C", "target-cpu=native", "-C", "link-arg=/STACK:16777216"]

[profile.release]
codegen-units = 16
lto = false
```

### PTX 架构

着色器使用 `--gpu-architecture=compute_75`（Turing）编译。PTX 是一种中间表示——NVIDIA 驱动程序在运行时将其 JIT 编译为实际的 GPU ISA。兼容从 Turing (RTX 20) 到 Blackwell (RTX 50) 的 GPU。

---

## GPU 诊断

```sh
./rt-next-week.exe --check-gpu
```

向 stdout 输出 JSON：
```json
{
  "status": "ok",
  "cuda": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 5080",
    "driver_version": "13.2",
    "compute_capability": "12.0",
    "vram_mb": 16302,
    "device_count": 1,
    "warnings": null,
    "error": null
  },
  "optix": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 5080",
    "error": null
  }
}
```

自动警告：
- 驱动版本 < R560 → "Driver too old: NVIDIA R560+ required for OptiX 9.x"
- 计算能力 < 7.5 → "GPU may not run all shaders correctly"

---

## Electron 前端

位置：`electron/`

```
electron/
├── main.js          — Electron 主进程，IPC 处理器，进程管理
├── preload.js       — 上下文桥接：向渲染器暴露安全 API
├── package.json     — 依赖：electron, electron-builder
├── electron-builder.yml — 构建配置（便携版目标）
└── renderer/
    ├── index.html   — UI 布局
    ├── renderer.js  — 渲染逻辑：校准、进度、GPU 状态
    └── style.css    — 暗色主题样式
```

**IPC 通道：**

| 通道 | 方向 | 用途 |
|------|------|------|
| `check-gpu` | 渲染器 → 主进程 | 运行 `--check-gpu`，返回解析后的 JSON |
| `read-calibration` | 渲染器 → 主进程 | 加载缓存的 CPU 校准数据 |
| `read-gpu-calibration` | 渲染器 → 主进程 | 加载缓存的 GPU 校准数据 |
| `run-calibration` | 渲染器 → 主进程 | 运行 160×90 基准渲染 |
| `start-render` | 渲染器 → 主进程 | 开始完整分辨率渲染 |
| `cancel-render` | 渲染器 → 主进程 | 终止正在运行的渲染进程 |
| `get-image-data` | 渲染器 → 主进程 | 将输出 PNG 读取为 base64 数据 URL |
| `render-progress` | 主进程 → 渲染器 | 进度更新（已完成/总像素数） |
| `render-done` | 主进程 → 渲染器 | 渲染完成，附带输出路径 |
| `render-error` | 主进程 → 渲染器 | 渲染错误，附带错误信息 |
| `render-log` | 主进程 → 渲染器 | 原始 stderr 输出行 |

**GPU 状态显示**（renderer.js 中）：
- 启动时通过 `--check-gpu` 检测 GPU 可用性
- 显示：设备名称、计算能力、显存、驱动版本
- 驱动过旧或 GPU 性能不足时显示警告
- 根据校准基准调整时间估算

---

## 测试

共 88 个单元测试，覆盖所有模块。运行方式：

```sh
cargo test --features cuda
```

主要测试分类：

| 模块 | 测试数 | 验证内容 |
|------|--------|---------|
| `vec3` | 15 | 算术运算、点积/叉积、单位向量、随机辅助函数 |
| `interval` | 6 | Contains、Surrounds、Clamp、Expand |
| `aabb` | 4 | 构建、碰撞检测、包围盒合并 |
| `bvh` | 4 | 命中/未命中、包围盒覆盖、PDF 正值性 |
| `sphere` | 3 | 命中球心、未命中、包围盒、pdf_value |
| `quad` | 4 | 命中中心、平行未命中、边界、pdf_value |
| `camera` | 7 | 宽高比、种子确定性、gamma 一致性 |
| `cuda::scene` | 11 | 细分、材质转换、法线、结构体大小 |
| `cuda::optix` | 1 | CameraParams 大小断言（148 字节） |
| `color_io` | 8 | Gamma 校正、不同位深的像素编码 |
| `pdf` | 5 | Sphere/Cosine/Mixture 的值和生成 |
| `perlin` | 2 | 噪声范围、确定性输出 |
| `ray` | 2 | at() 方法 |
| `material` | （隐式）| 通过相机和场景集成测试 |

---

## 打包

Electron 应用打包为便携版（免安装）ZIP：

```sh
cd electron
npm install
npm run dist          # 完整构建：electron-builder → dist-pkg/
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

输出：`RT Renderer 2.0.1 GPU Portable.zip`（约 110 MB）

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
