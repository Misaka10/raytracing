# RT Renderer — Physically Based Monte Carlo Path Tracer

Rust port of Peter Shirley's *Ray Tracing: The Next Week* with NVIDIA OptiX GPU acceleration and Electron desktop frontend.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Architecture](#architecture)
  - [Module Map](#module-map)
  - [CPU Rendering Pipeline](#cpu-rendering-pipeline)
  - [GPU Rendering Pipeline](#gpu-rendering-pipeline)
  - [Scene Construction](#scene-construction)
  - [Multiple Importance Sampling (MIS)](#multiple-importance-sampling-mis)
  - [Material System](#material-system)
  - [PDF System](#pdf-system)
- [Build System](#build-system)
- [GPU Diagnostics](#gpu-diagnostics)
- [Electron Frontend](#electron-frontend)
- [Testing](#testing)
- [Packaging](#packaging)
- [Requirements](#requirements)

---

## Overview

RT Renderer is a physically based path tracer implementing the techniques from *Ray Tracing: The Next Week*. It supports two rendering backends:

| Backend | Technology | Performance |
|---------|-----------|-------------|
| CPU | Rust + rayon parallel | ~200 px·sample/ms (16-core) |
| GPU | CUDA + NVIDIA OptiX 9.1 + RT Core BVH | ~10,000 px·sample/ms (RTX 5080) |

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
  --samples <N>       Samples per pixel, stratified sqrt(N)×sqrt(N) (default: 400)
  --max-depth <N>     Maximum ray bounces (default: 75)
  --output <PATH>     Output PNG path (default: output.png)
  --seed <N>          Random seed for deterministic rendering
  --gpu               Use GPU (OptiX RT Core) backend
  --denoise           Enable OptiX AI denoiser (GPU only)
  --json              Output JSON progress lines for IPC (used by Electron)
  --check-gpu         GPU diagnostics: probe driver, device, OptiX, then exit
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
├── ray.rs               — Ray { origin, direction, time }
├── interval.rs          — [min, max] interval math (clamp, expand, surrounds)
├── aabb.rs              — Axis-Aligned Bounding Box
├── bvh.rs               — BVH tree (O(log n) hit test, spatial median split)
├── hittable.rs          — HitRecord, Hittable enum (all geometry variants)
├── hittable_list.rs     — Flat object list (scene root + light list)
├── sphere.rs            — Analytic sphere: hit, pdf_value, random (solid-angle)
├── quad.rs              — Quadrilateral: hit, pdf_value, random (uniform area)
├── quad_box.rs          — make_box() from min/max corners (6 quads)
├── constant_medium.rs   — Volumetric fog (random distance sampling)
├── material.rs          — Material enum + scatter + scattering_pdf
├── texture.rs           — Texture enum (SolidColor)
├── onb.rs               — Orthonormal basis for cosine hemisphere sampling
├── pdf.rs               — PDF enum (Sphere, Cosine, Mixture)
├── perlin.rs            — 3D Perlin noise + turbulence
├── color_io.rs          — linear_to_gamma, pixel_to_10bit/16bit encoding
└── cuda/
    ├── mod.rs           — CUDA feature gate
    ├── optix.rs         — Rust FFI to optix_bridge C API + GPU diagnostics
    ├── optix_bridge.h   — C header for bridge library
    ├── optix_bridge.cu  — C/CUDA bridge: OptiX init, BVH build, render, denoiser
    ├── scene.rs         — GpuScene: Hittable → triangle mesh + vertex normals
    └── shaders/
        ├── common.h     — GpuFloat3, GpuMaterialData, CameraParams, LaunchParams
        ├── raygen.cu    — Ray generation shader (MIS path tracing loop)
        ├── closesthit.cu — Hit shader (barycentric normal interpolation)
        ├── miss.cu      — Miss shader (background color)
        ├── materials.h  — scatter_lambertian/metal/dielectric/isotropic
        ├── pdf.h        — Cosine PDF value, mixture PDF
        └── random.h     — PCG-based GPU RNG
```

### CPU Rendering Pipeline

Entry point: `camera.rs` → `Camera::render()`

```
For each pixel (rayon parallel):
  For each sub-pixel sample (sqrt_spp × sqrt_spp):
    1. Camera::get_ray() — stratified sample + defocus blur
    2. ray_color() — recursive path tracing
  Accumulate, scale by pixel_samples_scale
  Convert to 10-bit gamma via linear_to_gamma + pixel_to_10bit
Save as 16-bit PNG
```

`ray_color()` recursive logic:
1. Hit test via BVH: `world.hit(ray, [0.001, ∞])` → `HitRecord`
2. Miss → return black (enclosed Cornell box)
3. `material.emitted()` → emission contribution (non-zero only for DiffuseLight)
4. `material.scatter()` → `ScatterRecord`:
   - **DiffuseLight**: returns false → only emission, path ends
   - **Metal/Dielectric**: `skip_pdf=true` → recurse directly with `attenuation * ray_color(reflected_ray)`
   - **Lambertian/Isotropic**: `skip_pdf=false` → MIS path below
5. MIS: 50% light-list sampling / 50% BSDF sampling
6. `pdf_val = 0.5 * lights.pdf_value(scattered) + 0.5 * bsdf_pdf.value(scattered)`
7. Recurse: `sample_color = ray_color(scattered_ray, depth-1)`
8. Return: `emission + attenuation * scattering_pdf * sample_color / pdf_val`

### GPU Rendering Pipeline

Entry point: `camera.rs` → `Camera::render_gpu()`

**Phase 1 — Scene Upload (CPU side):**
```
Hittable tree → GpuScene::from_world()
  ├── Tessellate spheres: 32×32 lat/lon grid → 2048 triangles
  ├── Tessellate quads: 2 triangles per quad
  ├── Compute vertex normals (analytic for spheres, face normal for quads)
  ├── Deduplicate materials → GpuMaterialData buffer
  └── Build per-triangle material index
```

**Phase 2 — GPU Setup (optix_bridge.cu):**
```
Upload vertices/normals/indices/materials → GPU buffers
Build RT Core BVH (hardware acceleration structure)
Create OptiX pipeline (raygen + closesthit + miss)
```

**Phase 3 — Ray Generation (raygen.cu):**
```
For each pixel:
  For each sub-pixel sample (sqrt_spp × sqrt_spp):
    1. Stratified camera ray + defocus blur
    2. Path tracing loop (max_depth iterations):
       a. optixTrace() → RT Core BVH traversal
       b. Miss → add background, break
       c. Hit → read barycentric-interpolated normal + material
       d. DiffuseLight + front_face → add emission, break
       e. scatter() → ScatterResult
       f. skip_pdf (metal/dielectric) → direct recursion
       g. MIS: 50% BRDF / 50% hittable sampling
          - Hittable: 50% light rectangle / 50% sphere solid-angle
       h. pdf_val = 0.5*BSDF + 0.5*hittable_pdf
       i. throughput *= attenuation * scattering_pdf / pdf_val
    3. Accumulate, scale, clamp, write to framebuffer
```

**Phase 4 — Denoiser (optional, Tensor Core):**
```
OptiX AI HDR denoiser → denoised output buffer
```

**Phase 5 — Readback & Save:**
```
Copy output buffer GPU → CPU
PNG encoding: linear_to_gamma → 10-bit → 16-bit (same as CPU)
```

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
  Position: (213, 554, 227), size 130×105
  Material: DiffuseLight, emission (15, 15, 15)

Box:
  6 quads from (0,0,0) to (165, 330, 165), white
  Rotated 15° around Y axis
  Translated to (265, 0, 295)

Glass sphere:
  Center: (190, 90, 190), radius: 90
  Material: Dielectric, IOR 1.5

Camera:
  Position: (278, 278, -800), looking at (278, 278, 0)
  FOV: 40°, no defocus blur
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
  light_pdf      = dist² / (cos_light * area)   (if ray hits light rect)
  sphere_pdf     = 1.0 / solid_angle             (if ray hits glass sphere)

throughput *= attenuation * scattering_pdf / pdf_val
```

**Strategy selection (50/50):**
- **Strategy 1 (BSDF)**: Sample direction from cosine-weighted hemisphere. Compute hittable PDF for that direction.
- **Strategy 2 (Hittable)**: 50% sample point on light rectangle, 50% sample direction via solid-angle sphere sampling.

**Sphere solid-angle sampling** (matching CPU `random_to_sphere`):
1. Direction from hit point toward sphere center → build ONB
2. Sample z uniformly in [cos_θ_max, 1] where cos_θ_max = √(1 − r²/d²)
3. Sample φ uniformly in [0, 2π]
4. Transform local (√(1−z²)·cos φ, √(1−z²)·sin φ, z) via ONB

### Material System

| Material | scatter() returns | skip_pdf | scattering_pdf | Strategy |
|----------|-------------------|----------|----------------|----------|
| Lambertian | true | false | cos(θ)/π | Cosine hemisphere |
| Metal | true | true | N/A | Perfect/fuzzed reflection |
| Dielectric | true | true | N/A | Refraction or Schlick reflection |
| DiffuseLight | **false** | N/A | N/A | Only emission, path terminates |
| Isotropic | true | false | 1/(4π) | Uniform sphere |

**Metal scatter** (CPU behavior):
```rust
reflected = reflect(ray).unit_vector() + fuzz * random_unit_vector()
// NOT normalized — blur increases with distance
```

**Dielectric scatter:**
```rust
refraction_ratio = front_face ? 1.0/ir : ir
if cannot_refract || schlick_reflectance(cos_θ, ratio) > rand():
    reflect()      // total internal reflection or probabilistic
else:
    refract()      // Snell's law
```

### PDF System

```
Pdf enum:
├── Sphere       → value: 1/(4π),          generate: random_unit_vector
├── Cosine(Onb)  → value: cos(θ)/π,        generate: ONB × random_cosine_direction
└── Mixture(p0,p1) → value: avg of p0,p1,  generate: random pick p0 or p1
```

CPU `BsdfPdf` is constructed per material:
- Lambertian → `Pdf::Cosine(&normal)`
- Isotropic → `Pdf::Sphere()`

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
3. Patches PTX ISA version from 9.1 → 8.5 (CUDA 13.x generates 9.1 which OptiX 9.1 SDK rejects)
4. Compiles `optix_bridge.cu` to a static library (`.lib`)
5. Links: `optix_bridge.lib` (static) + `cudart.lib` + `cuda.lib` (dynamic from driver)

### Multi-threaded compilation

| Component | Parallelism |
|-----------|-------------|
| Cargo (rustc) | Per-crate parallelism (default: CPU cores) |
| rustc backend | `codegen-units=16` (`.cargo/config.toml`) |
| NVCC shaders | `std::thread::scope` — 3 shaders compiled concurrently |
| LTO | Disabled (`lto=false`) — avoids serial link bottleneck |

Config: `.cargo/config.toml`
```toml
[build]
rustflags = ["-C", "target-cpu=native", "-C", "link-arg=/STACK:16777216"]

[profile.release]
codegen-units = 16
lto = false
```

### PTX architecture

Shaders compiled with `--gpu-architecture=compute_75` (Turing). PTX is an intermediate representation — the NVIDIA driver JIT-compiles it to the actual GPU ISA at runtime. Compatible with Turing (RTX 20) through Blackwell (RTX 50) GPUs.

---

## GPU Diagnostics

```sh
./rt-next-week.exe --check-gpu
```

Outputs JSON to stdout:
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

Automatic warnings:
- Driver < R560 → "Driver too old: NVIDIA R560+ required for OptiX 9.x"
- Compute capability < 7.5 → "GPU may not run all shaders correctly"

---

## Electron Frontend

Location: `electron/`

```
electron/
├── main.js          — Electron main process, IPC handlers, spawn management
├── preload.js       — Context bridge: exposes safe API to renderer
├── package.json     — Dependencies: electron, electron-builder
├── electron-builder.yml — Build config (portable target)
└── renderer/
    ├── index.html   — UI layout
    ├── renderer.js  — Render logic: calibration, progress, GPU status
    └── style.css    — Dark theme styling
```

**IPC channels:**

| Channel | Direction | Purpose |
|---------|-----------|---------|
| `check-gpu` | renderer → main | Run `--check-gpu`, return parsed JSON |
| `read-calibration` | renderer → main | Load cached CPU calibration |
| `read-gpu-calibration` | renderer → main | Load cached GPU calibration |
| `run-calibration` | renderer → main | Run 160×90 benchmark render |
| `start-render` | renderer → main | Start full-resolution render |
| `cancel-render` | renderer → main | Kill running render process |
| `get-image-data` | renderer → main | Read output PNG as base64 data URL |
| `render-progress` | main → renderer | Progress update (completed/total pixels) |
| `render-done` | main → renderer | Render complete with output path |
| `render-error` | main → renderer | Render error with message |
| `render-log` | main → renderer | Raw stderr output lines |

**GPU status display** (in renderer.js):
- Checks GPU availability on startup via `--check-gpu`
- Shows: device name, compute capability, VRAM, driver version
- Shows warnings if driver is old or GPU capability is low
- Adjusts time estimate based on calibration benchmark

---

## Testing

88 unit tests across all modules. Run with:

```sh
cargo test --features cuda
```

Key test categories:

| Module | Tests | What they verify |
|--------|-------|-----------------|
| `vec3` | 15 | Arithmetic, dot/cross, unit vector, RNG helpers |
| `interval` | 6 | Contains, surrounds, clamp, expand |
| `aabb` | 4 | Construction, hit test, box union |
| `bvh` | 4 | Hit/miss, bounding box coverage, PDF positivity |
| `sphere` | 3 | Hit center, miss, bbox, pdf_value |
| `quad` | 4 | Hit center, parallel miss, bounds, pdf_value |
| `camera` | 7 | Aspect ratio, seed determinism, gamma consistency |
| `cuda::scene` | 11 | Tessellation, material conversion, normals, struct sizes |
| `cuda::optix` | 1 | CameraParams size assertion (148 bytes) |
| `color_io` | 8 | Gamma correction, pixel encoding at various bit depths |
| `pdf` | 5 | Sphere/cosine/mixture value and generation |
| `perlin` | 2 | Noise range, deterministic output |
| `ray` | 2 | at() method |
| `material` | (implicit) | Via camera + scene integration tests |

---

## Packaging

The Electron app is packaged as a portable (no-install) ZIP:

```sh
cd electron
npm install
npm run dist          # Full build: electron-builder → dist-pkg/
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

Output: `RT Renderer 2.0.1 GPU Portable.zip` (≈110 MB)

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
