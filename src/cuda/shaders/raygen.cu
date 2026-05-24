#include "common.h"
#include "random.h"
#include "materials.h"
#include "pdf.h"
#include <optix_device.h>

extern "C" {
__constant__ LaunchParams launch_params;
}

// Reconstruct face normal from geometry normal + ray direction
__device__ inline GpuFloat3 face_normal(GpuFloat3 ray_dir, GpuFloat3 geo_normal, bool* front_face) {
    if (vec_dot(ray_dir, geo_normal) < 0.0f) {
        *front_face = true;
        return geo_normal;
    } else {
        *front_face = false;
        return {-geo_normal.x, -geo_normal.y, -geo_normal.z};
    }
}

// Sample a random point uniformly on the area light rectangle
__device__ inline GpuFloat3 sample_light_point(RngState* rng) {
    float u = rng_uniform(rng);
    float v = rng_uniform(rng);
    return vec_add(
        launch_params.light_corner,
        vec_add(scl_mul(u, launch_params.light_u), scl_mul(v, launch_params.light_v))
    );
}

// Light surface normal (computed from light_u × light_v, normalized)
__device__ inline GpuFloat3 light_normal() {
    GpuFloat3 n = vec_cross(launch_params.light_u, launch_params.light_v);
    return vec_normalize(n);
}

// Check if a ray (origin + t*dir) hits the light rectangle.
// Returns the PDF value (distance^2 / (cos_light * area)), or 0 if missed.
__device__ inline float light_pdf_value(GpuFloat3 origin, GpuFloat3 dir, GpuFloat3 l_normal) {
    // Intersect ray with light plane: n·(p - corner) = 0
    // t = n·(corner - origin) / n·dir
    GpuFloat3 to_corner = vec_sub(launch_params.light_corner, origin);
    float denom = vec_dot(l_normal, dir);
    if (fabsf(denom) < 1e-7f) return 0.0f;
    float t = vec_dot(l_normal, to_corner) / denom;
    if (t <= 1e-4f) return 0.0f;

    // Intersection point
    GpuFloat3 ip = vec_add(origin, scl_mul(t, dir));

    // Check if within light rectangle (project onto u, v axes)
    GpuFloat3 local = vec_sub(ip, launch_params.light_corner);
    float u_len_sq = vec_dot(launch_params.light_u, launch_params.light_u);
    float v_len_sq = vec_dot(launch_params.light_v, launch_params.light_v);
    float alpha = vec_dot(local, launch_params.light_u) / u_len_sq;
    float beta  = vec_dot(local, launch_params.light_v) / v_len_sq;
    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return 0.0f;

    float dist_sq = t * t * vec_dot(dir, dir);
    float cos_light = fabsf(vec_dot(l_normal, dir));
    if (cos_light < 1e-7f) return 0.0f;
    return dist_sq / (cos_light * (1.0f / launch_params.light_area_inv));
}

extern "C" __global__ __launch_bounds__(256, 2) void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel_idx = idx.y * launch_params.width + idx.x;
    const unsigned int sqrt_spp = launch_params.sqrt_spp;
    const CameraParams& cam = launch_params.camera;

    // Precompute light normal once per launch
    GpuFloat3 l_normal = light_normal();

    // Per-pixel RNG seeded from global seed + pixel index (deterministic)
    RngState rng = rng_init(launch_params.seed
        ^ (pixel_idx * 0x9e3779b97f4a7c15ULL)
        ^ ((unsigned long long)idx.y << 32 | idx.x));

    GpuFloat3 accumulated = {0, 0, 0};

    for (unsigned int sj = 0; sj < sqrt_spp; sj++) {
        for (unsigned int si = 0; si < sqrt_spp; si++) {
            // Stratified sample (matches CPU Camera::get_ray)
            float px = ((float)si + rng_uniform(&rng)) / (float)sqrt_spp - 0.5f;
            float py = ((float)sj + rng_uniform(&rng)) / (float)sqrt_spp - 0.5f;

            GpuFloat3 pixel_sample = vec_add(
                cam.pixel00_loc,
                vec_add(
                    scl_mul((float)idx.x + px, cam.pixel_delta_u),
                    scl_mul((float)idx.y + py, cam.pixel_delta_v)
                )
            );

            GpuFloat3 ray_origin = cam.lookfrom;
            // Defocus blur
            if (cam.defocus_angle > 0.0f) {
                float r1 = rng_uniform(&rng);
                float r2 = rng_uniform(&rng);
                float disk_r = sqrtf(r1);
                float disk_theta = 2.0f * 3.141592653589793f * r2;
                ray_origin = vec_add(ray_origin,
                    vec_add(
                        scl_mul(disk_r * cosf(disk_theta), cam.defocus_disk_u),
                        scl_mul(disk_r * sinf(disk_theta), cam.defocus_disk_v)
                    )
                );
            }
            GpuFloat3 ray_dir = vec_normalize(vec_sub(pixel_sample, ray_origin));

            GpuFloat3 throughput = {1, 1, 1};
            GpuFloat3 color = {0, 0, 0};

            for (unsigned int depth = 0; depth < launch_params.max_depth; depth++) {
                // OptiX 9.x: all payload registers must be unsigned int
                unsigned int p0 = 1; // miss flag
                unsigned int p1 = 0, p2 = 0, p3 = 0; // hit position
                unsigned int p4 = 0, p5 = 0, p6 = 0; // normal
                unsigned int p7 = 0; // material ID

                optixTrace(
                    launch_params.traversable,
                    *reinterpret_cast<float3*>(&ray_origin),
                    *reinterpret_cast<float3*>(&ray_dir),
                    0.001f, 1e20f, 0.0f,
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_NONE,
                    0, 0, 0,
                    p0, p1, p2, p3, p4, p5, p6, p7
                );

                if (p0 == 1) {
                    // Miss — add background contribution
                    color = vec_add(color, vec_mul(throughput, launch_params.background));
                    break;
                }

                // Read hit data from payload
                GpuFloat3 hit_point = {
                    __uint_as_float(p1),
                    __uint_as_float(p2),
                    __uint_as_float(p3)
                };
                GpuFloat3 normal = {
                    __uint_as_float(p4),
                    __uint_as_float(p5),
                    __uint_as_float(p6)
                };
                unsigned int mat_id = p7;

                // Validate material ID
                if (mat_id >= launch_params.material_count) {
                    color = vec_add(color, vec_mul(throughput, {1.0f, 0.0f, 1.0f})); // magenta error
                    break;
                }

                GpuMaterialData mat = launch_params.materials[mat_id];

                // Emission
                if (mat.mat_type == MAT_DIFFUSE_LIGHT) {
                    color = vec_add(color, vec_mul(throughput, mat.emission));
                    break;
                }

                // Determine front face and correct normal
                bool front_face;
                normal = face_normal(ray_dir, normal, &front_face);

                // Scatter
                ScatterResult sr;
                sr.absorbed = false;
                sr.skip_pdf = false;

                switch (mat.mat_type) {
                    case MAT_LAMBERTIAN:
                        sr = scatter_lambertian(normal, mat.albedo, &rng);
                        break;
                    case MAT_METAL:
                        sr = scatter_metal(ray_dir, normal, mat.albedo, mat.fuzz, &rng);
                        break;
                    case MAT_DIELECTRIC:
                        sr = scatter_dielectric(ray_dir, normal, mat.ir, front_face, &rng);
                        break;
                    case MAT_ISOTROPIC:
                        sr = scatter_isotropic(mat.albedo, &rng);
                        break;
                    default:
                        sr.absorbed = true;
                        break;
                }

                if (sr.absorbed) {
                    break;
                }

                if (sr.skip_pdf) {
                    // Metal/dielectric: direct recursion
                    throughput = vec_mul(throughput, sr.attenuation);
                    if (vec_is_zero(throughput)) break;

                    ray_origin = hit_point;
                    ray_dir = sr.skip_ray_dir;
                    continue;
                }

                // === MIS with light sampling (50/50 mixture) ===

                GpuFloat3 scattered_dir;
                float pdf_val;
                float scattering_pdf;

                if (rng_uniform(&rng) < 0.5f) {
                    // Strategy 1: BRDF (cosine-weighted hemisphere) sampling
                    scattered_dir = sr.scattered_dir;
                    scattering_pdf = sr.pdf_value;  // cos(theta) / pi

                    // Mixture PDF: 0.5 * BSDF + 0.5 * hittable_pdf
                    // hittable_pdf = 0.5 * light_pdf + 0.5 * sphere_pdf (sphere_pdf=0 since CPU sphere has no pdf_value)
                    float light_pdf = light_pdf_value(hit_point, scattered_dir, l_normal);
                    float hittable_pdf = 0.5f * light_pdf; // 0.5 * light + 0.5 * 0
                    pdf_val = 0.5f * scattering_pdf + 0.5f * hittable_pdf;
                } else {
                    // Strategy 2: hittable sampling — 50% light rect / 50% sphere
                    if (rng_uniform(&rng) < 0.5f) {
                        // Light rectangle sampling
                        GpuFloat3 light_pt = sample_light_point(&rng);
                        scattered_dir = vec_normalize(vec_sub(light_pt, hit_point));
                        scattering_pdf = cosine_pdf_value(normal, scattered_dir);

                        float light_pdf = light_pdf_value(hit_point, scattered_dir, l_normal);
                        float hittable_pdf = 0.5f * light_pdf; // 0.5 * light + 0.5 * 0
                        pdf_val = 0.5f * scattering_pdf + 0.5f * hittable_pdf;

                        // Trace shadow ray to check light visibility
                        unsigned int sp0 = 0;
                        unsigned int sp1 = 0, sp2 = 0, sp3 = 0, sp4 = 0, sp5 = 0, sp6 = 0, sp7 = 0;
                        optixTrace(
                            launch_params.traversable,
                            *reinterpret_cast<float3*>(&hit_point),
                            *reinterpret_cast<float3*>(&scattered_dir),
                            0.001f, 0.999f * sqrtf(vec_dot(vec_sub(light_pt, hit_point), vec_sub(light_pt, hit_point))),
                            0.0f,
                            OptixVisibilityMask(255),
                            OPTIX_RAY_FLAG_NONE,
                            0, 1, 0,
                            sp0, sp1, sp2, sp3, sp4, sp5, sp6, sp7
                        );

                        if (sp0 == 0) {
                            unsigned int hit_mat_id = sp7;
                            bool hit_light = false;
                            if (hit_mat_id < launch_params.material_count) {
                                GpuMaterialData hit_mat = launch_params.materials[hit_mat_id];
                                hit_light = (hit_mat.mat_type == MAT_DIFFUSE_LIGHT);
                            }
                            if (!hit_light) {
                                pdf_val = 0.0f; // occluded by non-light geometry
                            }
                        }
                    } else {
                        // Sphere direction sampling (matching CPU hittable_pdf for glass sphere)
                        // Sample random point on sphere surface, scatter toward it
                        GpuFloat3 sphere_pt;
                        {
                            float u1 = rng_uniform(&rng);
                            float u2 = rng_uniform(&rng);
                            float sz = 1.0f - 2.0f * u2;
                            float sr = sqrtf(fmaxf(0.0f, 1.0f - sz * sz));
                            float phi = 2.0f * 3.141592653589793f * u1;
                            sphere_pt = {
                                launch_params.sphere_center.x + launch_params.sphere_radius * sr * cosf(phi),
                                launch_params.sphere_center.y + launch_params.sphere_radius * sz,
                                launch_params.sphere_center.z + launch_params.sphere_radius * sr * sinf(phi)
                            };
                        }
                        scattered_dir = vec_normalize(vec_sub(sphere_pt, hit_point));
                        scattering_pdf = cosine_pdf_value(normal, scattered_dir);

                        // hittable_pdf = 0.5 * light_pdf_value + 0.5 * sphere_pdf_value
                        // sphere_pdf_value = 0 (CPU sphere doesn't implement pdf_value)
                        // For direction toward sphere, light_pdf = 0 (doesn't hit light plane)
                        float hittable_pdf = 0.0f; // both light and sphere pdf = 0
                        pdf_val = 0.5f * scattering_pdf + 0.5f * hittable_pdf;

                        // Sphere direction — trace full path (recurse), not shadow ray
                        // The ray will scatter/refract when it hits the sphere
                        // skip_pdf is NOT used here — we pass through MIS weighting
                    }
                }

                if (pdf_val < 1e-10f) {
                    // Retry with BRDF-only as fallback
                    scattered_dir = sr.scattered_dir;
                    scattering_pdf = sr.pdf_value;
                    pdf_val = scattering_pdf;
                }

                if (pdf_val < 1e-10f) break;

                // MIS weighting: throughput *= attenuation * scattering_pdf / pdf_val
                throughput = vec_mul(throughput,
                    scl_mul(scattering_pdf / pdf_val, sr.attenuation));

                if (vec_is_zero(throughput)) break;

                // Russian roulette
                if (depth > 3) {
                    float q = fmaxf(vec_max_component(throughput), 0.05f);
                    if (rng_uniform(&rng) > q) break;
                    throughput = scl_div(throughput, q);
                }

                ray_origin = hit_point;
                ray_dir = scattered_dir;
            }

            accumulated = vec_add(accumulated, color);
        }
    }

    accumulated = scl_mul(launch_params.pixel_samples_scale, accumulated);

    // Clamp and write to framebuffer
    accumulated.x = fminf(fmaxf(accumulated.x, 0.0f), 100.0f);
    accumulated.y = fminf(fmaxf(accumulated.y, 0.0f), 100.0f);
    accumulated.z = fminf(fmaxf(accumulated.z, 0.0f), 100.0f);

    launch_params.framebuffer[pixel_idx] = accumulated;
}
