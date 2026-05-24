#ifndef GPU_PDF_H
#define GPU_PDF_H

#include "common.h"
#include "random.h"

// Light sampling: pick a random point on a light source.
// For simplicity (Phase 3), we sample toward the light position from the hit point.
// The light position is passed from the host as part of the launch params.
// Full hittable_pdf requires traversing the light geometry in shader — deferred to Phase 4.

// Cosine PDF value
__device__ inline float cosine_pdf_value(float3 normal, float3 dir) {
    float cos = vec_dot(normal, dir);
    return cos > 0.0f ? cos / 3.141592653589793f : 0.0f;
}

// Mixture PDF: 50/50 blend of light sampling and BRDF (cosine) sampling
// p = 0.5 * p_light + 0.5 * p_cosine
__device__ inline float mixture_pdf_value(
    float3 dir, float3 normal, float3 light_point, float3 hit_point)
{
    float cosine_pdf = cosine_pdf_value(normal, dir);

    // Light PDF: uniform over the solid angle subtended by a sphere at light_point
    // Simplified: direction toward light
    float3 to_light = vec_sub(light_point, hit_point);
    float dist_sq = vec_dot(to_light, to_light);
    float light_pdf = 0.0f;
    if (dist_sq > 0.0001f) {
        // Approximate: treat light as point source, weight by inverse square
        // This is a simplification — full hittable_pdf needs light geometry
        light_pdf = 1.0f / dist_sq;
    }
    // Cap light PDF to avoid singularities
    light_pdf = fminf(light_pdf, 1000.0f);

    return 0.5f * cosine_pdf + 0.5f * light_pdf;
}

// Sample a direction for MIS: 50/50 chance of cosine or light sampling
__device__ inline float3 mixture_pdf_generate(
    float3 normal, float3 light_point, float3 hit_point, RngState* rng)
{
    if (rng_uniform(rng) < 0.5f) {
        // Cosine-weighted hemisphere sampling
        Onb onb = onb_from_normal(normal);
        float r1 = rng_uniform(rng);
        float r2 = rng_uniform(rng);
        float phi = 2.0f * 3.141592653589793f * r1;
        float r = sqrtf(r2);
        float x = cosf(phi) * r;
        float y = sinf(phi) * r;
        float z = sqrtf(1.0f - r2);
        return onb_transform(&onb, make_float3(x, y, z));
    } else {
        // Sample toward the light point (with some spread)
        float3 to_light = vec_sub(light_point, hit_point);
        float dist = vec_length(to_light);
        float3 dir_to_light = scl_div(to_light, dist);

        // Add small random offset to avoid perfect alignment
        float3 offset = scl_mul(0.1f, random_unit_sphere_direction(rng));
        return vec_normalize(vec_add(dir_to_light, offset));
    }
}

#endif // GPU_PDF_H
