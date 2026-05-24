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

// Check if a ray from origin in direction hits the sphere
__device__ inline bool dir_hits_sphere(GpuFloat3 origin, GpuFloat3 dir) {
    GpuFloat3 oc = vec_sub(origin, launch_params.sphere_center);
    float a = vec_dot(dir, dir);
    float h = vec_dot(oc, dir);
    float c = vec_dot(oc, oc) - launch_params.sphere_radius * launch_params.sphere_radius;
    float discriminant = h * h - a * c;
    if (discriminant < 0.0f) return false;
    float sqrtd = sqrtf(discriminant);
    return (-h - sqrtd) / a > 0.001f || (-h + sqrtd) / a > 0.001f;
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

                // Determine front face and correct normal (before emission check)
                bool front_face;
                normal = face_normal(ray_dir, normal, &front_face);

                // Emission (only from front face, matching CPU)
                if (mat.mat_type == MAT_DIFFUSE_LIGHT) {
                    if (front_face) {
                        color = vec_add(color, vec_mul(throughput, mat.emission));
                    }
                    break;
                }

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

                // Sphere PDF helper: returns 1/solid_angle if direction hits sphere, else 0
                // Matching CPU sphere.pdf_value() at sphere.rs:58-67
                float sphere_pdf_val = 0.0f;
                {
                    GpuFloat3 sc = launch_params.sphere_center;
                    float sr = launch_params.sphere_radius;
                    GpuFloat3 oc = vec_sub(hit_point, sc);
                    // If origin is inside the sphere, skip (shouldn't happen for solid-angle sampling)
                    float dist_sq = vec_dot(oc, oc);
                    if (dist_sq > sr * sr + 1e-4f) {
                        float cos_theta_max = sqrtf(1.0f - fminf(1.0f, sr * sr / dist_sq));
                        float solid_angle = 2.0f * 3.141592653589793f * (1.0f - cos_theta_max);
                        if (solid_angle > 1e-10f) {
                            sphere_pdf_val = 1.0f / solid_angle;
                        }
                    }
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
                    // hittable_pdf = 0.5 * light_pdf + 0.5 * sphere_pdf
                    float light_pdf = light_pdf_value(hit_point, scattered_dir, l_normal);
                    float dir_sphere_pdf = dir_hits_sphere(hit_point, scattered_dir) ? sphere_pdf_val : 0.0f;
                    float hittable_pdf = 0.5f * light_pdf + 0.5f * dir_sphere_pdf;
                    pdf_val = 0.5f * scattering_pdf + 0.5f * hittable_pdf;
                } else {
                    // Strategy 2: hittable sampling — 50% light rect / 50% sphere
                    if (rng_uniform(&rng) < 0.5f) {
                        // Light rectangle sampling
                        GpuFloat3 light_pt = sample_light_point(&rng);
                        scattered_dir = vec_normalize(vec_sub(light_pt, hit_point));
                        scattering_pdf = cosine_pdf_value(normal, scattered_dir);

                        float light_pdf = light_pdf_value(hit_point, scattered_dir, l_normal);
                        float rect_sphere_pdf = dir_hits_sphere(hit_point, scattered_dir) ? sphere_pdf_val : 0.0f;
                        float hittable_pdf = 0.5f * light_pdf + 0.5f * rect_sphere_pdf;
                        pdf_val = 0.5f * scattering_pdf + 0.5f * hittable_pdf;
                    } else {
                        // Sphere solid-angle sampling (matching CPU sphere.random() / random_to_sphere)
                        GpuFloat3 to_sphere = vec_sub(launch_params.sphere_center, hit_point);
                        float dist_sq = vec_dot(to_sphere, to_sphere);
                        GpuFloat3 dir_to_sphere = scl_div(to_sphere, sqrtf(dist_sq));
                        Onb onb_s = onb_from_normal(dir_to_sphere);
                        {
                            float r1 = rng_uniform(&rng);
                            float r2 = rng_uniform(&rng);
                            float cos_theta_max = sqrtf(1.0f - launch_params.sphere_radius * launch_params.sphere_radius / dist_sq);
                            float z = 1.0f + r2 * (cos_theta_max - 1.0f);
                            float sin_theta = sqrtf(1.0f - z * z);
                            float phi = 2.0f * 3.141592653589793f * r1;
                            scattered_dir = onb_transform(&onb_s, {cosf(phi) * sin_theta, sinf(phi) * sin_theta, z});
                        }
                        scattering_pdf = cosine_pdf_value(normal, scattered_dir);
                        float light_pdf = light_pdf_value(hit_point, scattered_dir, l_normal);
                        float hittable_pdf = 0.5f * light_pdf + 0.5f * sphere_pdf_val;
                        pdf_val = 0.5f * scattering_pdf + 0.5f * hittable_pdf;
                    }
                }

                if (pdf_val < 1e-10f) break;

                // MIS weighting: throughput *= attenuation * scattering_pdf / pdf_val
                throughput = vec_mul(throughput,
                    scl_mul(scattering_pdf / pdf_val, sr.attenuation));

                if (vec_is_zero(throughput)) break;

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
