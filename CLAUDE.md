# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Build

```sh
# One-click: CPU build + Electron package
.\build.bat

# One-click: CPU + GPU build + Electron package
.\build.bat --gpu

# CPU-only (fast)
cargo build --release

# GPU (requires CUDA 13.1 + OptiX 9.1.0 SDK)
cargo build --release --features cuda

# GPU diagnostics (JSON: driver version, CC, VRAM, OptiX status)
.\target\release\rt-next-week.exe --check-gpu

# Run all tests
cargo test --features cuda

# Run specific test module
cargo test --lib cuda::scene::tests --features cuda
```

Build script (`build.rs`) compiles 3 `.cu` shaders to PTX **in parallel** via `std::thread::scope` + NVCC, then patches PTX ISA 9.1 → 8.5 for OptiX 9.x compatibility. `.cargo/config.toml` sets `codegen-units=16` for parallel rustc backend with `lto=false` to avoid serial link bottleneck.

### Electron packaging

```sh
cd electron
npm install

# Rebuild full Electron portable package
npm run dist

# Quick fix: repack only app.asar (frontend changes)
npx asar extract "electron/dist-pkg/win-unpacked/resources/app.asar" app-src/
# ... edit files in app-src/ ...
npx asar pack app-src resources/app.asar
# Then update resources/rt-next-week.exe and resources/app.asar in ZIP
```

### --check-gpu JSON output

```json
{
  "status": "ok",
  "cuda": {
    "available": true, "device_name": "NVIDIA GeForce RTX 5080",
    "driver_version": "13.2", "compute_capability": "12.0",
    "vram_mb": 16302, "device_count": 1,
    "warnings": null, "error": null
  },
  "optix": { "available": true, "device_name": "...", "error": null }
}
```

Automatic warnings: driver < R560, compute capability < 7.5.

## Architecture

Rust port of Peter Shirley's *Ray Tracing: The Next Week* path tracer with NVIDIA OptiX GPU acceleration.

### Module map

```
src/
├── main.rs           — CLI entry point; Cornell box scene construction
├── lib.rs            — Module declarations; feature-gated cuda module
├── camera.rs         — CPU render loop (rayon parallel) + GPU render entry point
├── vec3.rs           — Vec3 (x,y,z), Point3, Color aliases; SIMD f64 layout
├── ray.rs            — Ray { origin, direction }
├── rng.rs            — Seedable RNG wrapper (deterministic rendering)
├── interval.rs       — [min, max] interval math
├── aabb.rs           — Axis-aligned bounding box
├── bvh.rs            — BvhNode (leaf/split recursive tree)
├── hittable.rs       — HitRecord, Hittable enum (all geometry variants)
├── hittable_list.rs  — Flat object list (scene root + light list)
├── sphere.rs         — Analytic sphere intersection
├── quad.rs           — Quadrilateral intersection + PDF value
├── quad_box.rs       — make_box() from min/max corners
├── constant_medium.rs — Volumetric fog
├── material.rs       — Material enum (Lambertian, Metal, Dielectric, DiffuseLight, Isotropic)
├── texture.rs        — Texture enum (SolidColor)
├── onb.rs            — Orthonormal basis for cosine hemisphere sampling
├── pdf.rs            — Mixture, Cosine, Hittable, Sphere PDFs
├── perlin.rs         — Procedural noise
├── color_io.rs       — linear_to_gamma, pixel_to_10bit/16bit encoding
└── cuda/
    ├── mod.rs         — CUDA feature gate
    ├── optix.rs       — Rust FFI to optix_bridge C API
    ├── optix_bridge.h — C header for bridge library
    ├── optix_bridge.cu — C bridge: CUDA/OptiX init, BVH build, render launch, denoiser
    ├── scene.rs       — GpuScene: Hittable → triangle mesh conversion + vertex normals
    └── shaders/
        ├── common.h   — GpuFloat3, GpuMaterialData, CameraParams, LaunchParams, helpers
        ├── raygen.cu  — Ray generation shader with MIS path tracing
        ├── closesthit.cu — Hit shader with barycentric normal interpolation
        ├── miss.cu     — Miss shader (background color)
        ├── materials.h — Lambertian, Metal, Dielectric, Isotropic scatter functions
        ├── pdf.h       — Cosine PDF; mixture PDF value
        └── random.h    — PCG-based GPU RNG
```

### CPU → GPU scene flow

1. `main.rs` builds the Cornell box scene as `Hittable` tree
2. `render_gpu()` in `camera.rs` calls `GpuScene::from_world()` to flatten the tree and tessellate into triangles
3. `GpuScene` stores: vertices, normals (per-vertex, barycentric-interpolated in shader), indices, per-triangle material IDs, deduplicated materials
4. `optix_bridge` uploads all buffers to GPU, builds RT Core BVH, launches raygen

### GPU rendering pipeline

- **Ray gen**: Stratified samples, path tracing loop with MIS (50/50 BRDF + light/sphere sampling), full recursion (no Russian roulette, matching CPU)
- **Closest hit**: Barycentric normal interpolation from per-vertex normals (smooth spheres + flat quads)
- **Miss**: Returns background color
- **Denoiser**: OptiX AI HDR denoiser (Tensor Core), post-render pass

### Data layout (critical)

- `GpuFloat3 = {float x, y, z}` — 12 bytes, 4-byte alignment. GPU and host must match.
- `GpuMaterialData` = 36 bytes (u32 + GpuFloat3 + f32 + f32 + GpuFloat3)
- `CameraParams` = 148 bytes (11 × GpuFloat3 + 4 × f32)
- Static asserts in `common.h` and `optix_bridge.cu` verify sizes match

### Scene tessellation

- **Spheres**: 32 × 32 lat/lon grid = 2048 triangles, vertex normals = analytic `(p - center) / radius`
- **Quads**: 2 triangles, face normal = normalize(u × v), all 4 vertices share same normal
- **RotateY**: Vertices + normals both rotated (Y-axis)
- **Translate**: Vertices offset, normals unchanged (direction vectors)
- **HittableList inside transforms**: Handled by recursion in `tessellate_object`

### MIS notes

- CPU `lights` list has: light quad + glass sphere (empty material for direction sampling)
- GPU matches this: light quad via `find_light_quad` + sphere via `find_glass_sphere`
- In raygen strategy 2: 50% light rectangle / 50% sphere solid-angle sampling (matching `random_to_sphere`)
- Sphere PDF value = `1.0 / solid_angle` (matching CPU `sphere.pdf_value()`)
- Sphere solid-angle sampling uses ONB toward sphere center with z in `[cos_theta_max, 1]`

## Testing

88 unit tests across all modules (88 with `--features cuda`). Key tests:
- `cuda::scene::tests::test_box_with_transform_not_empty` — Regressed: HittableList silently dropped in tessellation
- `cuda::scene::tests::test_sphere_vertex_normals_*` — Normals are unit length, correct direction
- `cuda::scene::tests::test_gpu_material_size` — 36 bytes (matches GPU GpuMaterialData)
- `camera::tests::test_gpu_png_gamma_matches_cpu_encoding` — Gamma correction consistency
