#include "common.h"
#include "random.h"
#include "materials.h"
#include "pdf.h"
#include <optix_device.h>

extern "C" {
__constant__ LaunchParams launch_params;
}

// Reconstruct face normal from geometry normal + ray direction
// On GPU we don't have the CPU HitRecord::set_face_normal, so we compute it inline
__device__ inline float3 face_normal(float3 ray_dir, float3 geo_normal, bool* front_face) {
    if (vec_dot(ray_dir, geo_normal) < 0.0f) {
        *front_face = true;
        return geo_normal;
    } else {
        *front_face = false;
        return make_float3(-geo_normal.x, -geo_normal.y, -geo_normal.z);
    }
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel_idx = idx.y * launch_params.width + idx.x;
    const unsigned int sqrt_spp = launch_params.sqrt_spp;
    const CameraParams& cam = launch_params.camera;

    // Per-pixel RNG seeded from global seed + pixel index (deterministic)
    RngState rng = rng_init(launch_params.seed
        ^ (pixel_idx * 0x9e3779b97f4a7c15ULL)
        ^ ((unsigned long long)idx.y << 32 | idx.x));

    float3 accumulated = make_float3(0, 0, 0);

    for (unsigned int sj = 0; sj < sqrt_spp; sj++) {
        for (unsigned int si = 0; si < sqrt_spp; si++) {
            // Stratified sample (matches CPU Camera::get_ray)
            float px = ((float)si + rng_uniform(&rng)) / (float)sqrt_spp - 0.5f;
            float py = ((float)sj + rng_uniform(&rng)) / (float)sqrt_spp - 0.5f;

            float3 pixel_sample = vec_add(
                cam.pixel00_loc,
                vec_add(
                    scl_mul((float)idx.x + px, cam.pixel_delta_u),
                    scl_mul((float)idx.y + py, cam.pixel_delta_v)
                )
            );

            float3 ray_origin = cam.lookfrom;
            // Defocus blur
            if (cam.defocus_angle > 0.0f) {
                float r1 = rng_uniform(&rng);
                float r2 = rng_uniform(&rng);
                // Random in unit disk
                float disk_r = sqrtf(r1);
                float disk_theta = 2.0f * 3.141592653589793f * r2;
                ray_origin = vec_add(ray_origin,
                    vec_add(
                        scl_mul(disk_r * cosf(disk_theta), cam.defocus_disk_u),
                        scl_mul(disk_r * sinf(disk_theta), cam.defocus_disk_v)
                    )
                );
            }
            float3 ray_dir = vec_normalize(vec_sub(pixel_sample, ray_origin));

            float3 throughput = make_float3(1, 1, 1);
            float3 color = make_float3(0, 0, 0);

            for (unsigned int depth = 0; depth < launch_params.max_depth; depth++) {
                // OptiX 9.x: all payload registers must be unsigned int
                unsigned int p0 = 1; // miss flag
                unsigned int p1 = 0, p2 = 0, p3 = 0; // hit position
                unsigned int p4 = 0, p5 = 0, p6 = 0; // normal
                unsigned int p7 = 0; // material ID

                optixTrace(
                    launch_params.traversable,
                    ray_origin, ray_dir,
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
                float3 hit_point = make_float3(
                    __uint_as_float(p1),
                    __uint_as_float(p2),
                    __uint_as_float(p3)
                );
                float3 normal = make_float3(
                    __uint_as_float(p4),
                    __uint_as_float(p5),
                    __uint_as_float(p6)
                );
                unsigned int mat_id = p7;

                // Validate material ID
                if (mat_id >= launch_params.material_count) {
                    color = vec_add(color, vec_mul(throughput,
                        make_float3(1.0f, 0.0f, 1.0f))); // magenta error
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

                // MIS: sample scattered direction
                // For now, use the scatter result's direction directly
                // Full MIS with light sampling deferred to future phase
                float3 scattered_dir = sr.scattered_dir;
                float pdf_val = sr.pdf_value;

                if (pdf_val < 1e-10f) break;

                // Scattering PDF
                float scattering_pdf = cosine_pdf_value(normal, scattered_dir);

                // Trace shadow ray toward light (simple light at center of Cornell box ceiling)
                // Phase 3: simplified — just bounce without explicit light sampling
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
