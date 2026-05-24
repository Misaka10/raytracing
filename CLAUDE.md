# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

All compilation requires `stb_image.h` and `stb_image_write.h` in `external/` (already present).

```sh
# MSVC (one-shot, fast for development)
.\build_msvc.bat

# CMake
mkdir build && cd build && cmake .. && cmake --build .
```

Outputs go to `build/`. The main renderer writes PPM to stdout (progress to stderr) and PNG to the path in `cam.png_filename`.

There are no tests in this project — correctness is verified by inspecting the rendered image output.

## Architecture

This is Peter Shirley's *Ray Tracing: The Next Week* path tracer — a C++17 header-only codebase where everything lives in `include/` and `src/` is a set of standalone `main()` programs.

### Include graph (compile order matters)

`rtweekend.h` is the root — it provides `pi`, `infinity`, RNG helpers, and pulls in `vec3.h`, `ray.h`, `color.h`, `interval.h`. Every other header relies on it being included first.

```
rtweekend.h
├── aabb.h        (needs interval, vec3, ray)
├── onb.h         (needs vec3)
├── texture.h     (needs perlin.h, rtw_stb_image.h; indirectly needs rtweekend)
├── hittable.h    (needs aabb.h; forward-declares material)
├── hittable_list.h (needs hittable.h)
├── bvh.h         (needs hittable_list.h)
├── pdf.h         (needs hittable_list.h, onb.h)
├── sphere.h      (needs hittable.h)
├── quad.h        (needs hittable.h)
├── material.h    (needs hittable.h, pdf.h, texture.h) — defines scatter_record before use
├── constant_medium.h (needs hittable.h, material.h)
├── camera.h      (needs material.h, pdf.h) — the render loop + stb PNG output
```

The include order in `main.cc` is the canonical order: `rtweekend.h` → `camera.h` → `hittable_list.h` → `material.h` → `quad.h` → `sphere.h`.

### The render loop (`camera.h`)

`camera::render()` drives everything:

1. Stratified sampling: `sqrt_spp × sqrt_spp` subpixel samples per pixel
2. `get_ray()` generates camera rays with defocus blur and time sampling
3. `ray_color()` recursively traces each ray, calling `world.hit()` to find the nearest intersection, then `material::scatter()` to bounce
4. PPM output written line-by-line; optional PNG via `stbi_write_png`

### `ray_color()` and MIS

`ray_color()` is in `camera.h` and implements multiple importance sampling:

- Hit nothing → return background
- Hit a surface → get emission via `material::emitted()`, then scatter via `material::scatter()`
- If `scatter_record::skip_pdf` is true (metal, dielectric), recurse directly — no PDF required
- Otherwise, build a `mixture_pdf` (50/50 blend of light sampling + BRDF sampling), generate a scattered direction, and apply the MIS weight: `attenuation × scattering_pdf × sample_color / pdf_value`

### The hittable tree

```
hittable (abstract)
├── sphere          — stationary or moving (time-interpolated center)
├── quad            — axis-aligned quadrilateral; also the building block for box()
├── constant_medium — volumetric fog via random distance sampling inside a boundary
├── hittable_list   — flat collection, used as the scene root; also provides pdf_value/random averaging
├── bvh_node        — recursive spatial partition (sorts objects along longest axis, builds tree)
├── translate       — decorator: offsets ray, then offsets hit result back
└── rotate_y        — decorator: rotates ray into object space, rotates hit back to world space
```

### Materials and their scatter strategies

| Material | `skip_pdf` | Strategy |
|----------|-----------|----------|
| `lambertian` | false | Cosine-weighted hemisphere PDF; `scattering_pdf` = cos/π |
| `metal` | true | Perfect/slightly-fuzzed reflection; no PDF needed |
| `dielectric` | true | Refraction or Schlick reflectance; no PDF needed |
| `diffuse_light` | N/A | Only emits — `scatter()` returns false, so only emission is added |
| `isotropic` | false | Uniform sphere PDF; `scattering_pdf` = 1/(4π) |

### PDF system (`pdf.h`)

- `sphere_pdf` — uniform sampling over the unit sphere (1/4π)
- `cosine_pdf` — cosine-weighted sampling around a normal (uses ONB transform)
- `hittable_pdf` — samples a direction toward a random point on a light source
- `mixture_pdf` — 50/50 blend of two PDFs; `generate()` picks randomly, `value()` averages

### Standalone demos (`src/`)

Each `.cc` file is a self-contained Monte Carlo program (no dependency on the renderer headers, only `rtweekend.h`). They demonstrate specific concepts from the book: estimating π, importance sampling, hemisphere integration, etc. Each compiles to its own executable via CMake.

### Conventions

- `std::rand()` for RNG (not `<random>`) — single global seed
- `shared_ptr` for all heap objects (materials, hittables, textures, PDFs) — `make_shared` via using declarations in `rtweekend.h`
- `using point3 = vec3` and `using color = vec3` — geometric clarity aliases
- All code is CC0 (public domain)
