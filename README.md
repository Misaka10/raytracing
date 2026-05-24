# Ray Tracing: The Next Week

A physically based Monte Carlo path tracer in C++ by **Peter Shirley**, companion code to the book *Ray Tracing: The Next Week* (the sequel to *Ray Tracing in One Weekend*).

This program renders a classic Cornell box scene with diffuse walls, a glass sphere, and a light source — simulating light transport with multiple importance sampling and recursive bounces.

All code is dedicated to the **public domain** under the [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0/) license.

## Features

- Monte Carlo path tracing with recursive bounces (up to `max_depth`)
- Multiple importance sampling (MIS) via mixture PDFs
- Materials: Lambertian (diffuse), metal (with fuzz), dielectric/glass (Schlick approximation)
- Area lights (emissive quads) and volumetric scattering (constant-density fog)
- BVH tree for ray-scene intersection acceleration
- Perlin noise and image-based textures
- Defocus blur (thin-lens camera model)
- Motion blur (moving spheres with ray time sampling)
- Object instancing via translate and rotate-y decorators
- Stratified sampling for anti-aliasing and gamma correction
- PPM image output to stdout (progress to stderr)

## File Overview

### Main Renderer

| File | Purpose |
|---|---|
| `main.cc` | Builds the Cornell box scene and calls `camera::render()` |
| `camera.h` | Camera model with FOV, defocus blur, stratified sampling, and the PPM render loop |
| `hittable.h` | Abstract `hittable` interface plus `translate` and `rotate_y` decorators |
| `hittable_list.h` | Collection of hittable objects with aggregate bounding box and PDF averaging |
| `bvh.h` | Bounding Volume Hierarchy node for spatial partitioning acceleration |
| `sphere.h` | Sphere primitive (stationary and moving) with ray intersection |
| `quad.h` | Axis-aligned quadrilateral primitive and the `box()` helper |
| `material.h` | Material system: lambertian, metal, dielectric, diffuse_light, isotropic |
| `texture.h` | Textures: solid color, checker, image (via stb_image), Perlin noise |
| `perlin.h` | Perlin noise generator with turbulence |
| `pdf.h` | PDF classes for importance sampling: sphere, cosine, hittable, mixture |
| `onb.h` | Orthonormal basis construction for surface sampling |
| `constant_medium.h` | Volumetric fog (constant-density participating medium) |

### Core Utilities

| File | Purpose |
|---|---|
| `rtweekend.h` | Common includes, constants (pi, infinity), utility math functions |
| `vec3.h` | 3D vector class with full arithmetic, dot/cross product, reflection, refraction |
| `color.h` | RGB color (alias for vec3) with gamma correction and PPM output |
| `ray.h` | Ray class (origin, direction, time) with parametric evaluation |
| `interval.h` | 1D interval with contains/surrounds/clamp operations |
| `aabb.h` | Axis-aligned bounding box and ray-AABB intersection |
| `rtw_stb_image.h` | Wrapper around the stb_image header library for texture loading |

### Standalone Monte Carlo Demos

| File | Purpose |
|---|---|
| `pi.cc` | Estimating pi via rejection sampling (uniform vs. stratified) |
| `integrate_x_sq.cc` | Integrating x^2 with importance sampling |
| `cos_cubed.cc` | Estimating integral of cos^3(theta) over a hemisphere (uniform sampling) |
| `cos_density.cc` | Same integral as above, but cosine-weighted (importance sampling) |
| `sphere_importance.cc` | Cosine-squared over sphere surface via uniform spherical sampling |
| `sphere_plot.cc` | Generates 200 random points uniformly on a unit sphere |
| `estimate_halfway.cc` | Median estimation of sin^2(x)*exp(-x/(2*pi)) via sorted MC samples |

## Building

Requires a **C++17** compiler (g++, clang++, or MSVC) and the `stb_image.h` single-header library.

1. Download `stb_image.h` from [nothings/stb](https://github.com/nothings/stb) and place it in an `external/` directory at the project root:

   ```
   external/
     stb_image.h
   ```

2. Compile and run the main renderer:

   ```sh
   # g++ or clang++
   g++ -std=c++17 -O2 main.cc -o rt.exe && ./rt.exe > output.ppm

   # MSVC
   cl /EHsc /std:c++17 /O2 main.cc /Fe:rt.exe && rt.exe > output.ppm
   ```

   Each standalone demo can be compiled the same way (e.g., `g++ -std=c++17 pi.cc -o pi.exe`).

## Dependencies

- C++17 standard library (cmath, cstdlib, iostream, limits, memory, vector)
- [stb_image.h](https://github.com/nothings/stb) — single-header image loader (for `image_texture`)
