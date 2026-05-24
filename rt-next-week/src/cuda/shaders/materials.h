#ifndef GPU_MATERIALS_H
#define GPU_MATERIALS_H

#include "common.h"
#include "random.h"

// Output of a scatter evaluation
struct ScatterResult {
    float3 attenuation;
    float3 scattered_dir;
    float  pdf_value;
    bool   skip_pdf;      // metal/dielectric return skip_pdf=true
    float3 skip_ray_dir;  // direct reflection/refraction direction
    bool   absorbed;      // dielectric total internal reflection
};

// Cosine-weighted hemisphere sampling around z-up
__device__ inline float3 random_cosine_direction(RngState* rng) {
    float r1 = rng_uniform(rng);
    float r2 = rng_uniform(rng);
    float phi = 2.0f * 3.141592653589793f * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);
    return make_float3(x, y, z);
}

// Uniform sphere sampling
__device__ inline float3 random_unit_sphere_direction(RngState* rng) {
    float r1 = rng_uniform(rng);
    float r2 = rng_uniform(rng);
    float z = 1.0f - 2.0f * r1;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * 3.141592653589793f * r2;
    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

// Build ONB from a normal (z-axis = normal)
struct Onb {
    float3 u, v, w;
};

__device__ inline Onb onb_from_normal(float3 n) {
    Onb onb;
    onb.w = n;
    float3 a = fabsf(n.x) > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    onb.v = vec_normalize(vec_cross(n, a));
    onb.u = vec_cross(n, onb.v);
    return onb;
}

__device__ inline float3 onb_transform(const Onb* onb, float3 v) {
    return vec_add(
        vec_add(scl_mul(v.x, onb->u), scl_mul(v.y, onb->v)),
        scl_mul(v.z, onb->w)
    );
}

// Schlick reflectance approximation
__device__ inline float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// Lambertian scatter: cosine-weighted hemisphere
__device__ inline ScatterResult scatter_lambertian(
    float3 normal, float3 albedo, RngState* rng)
{
    ScatterResult sr;
    sr.attenuation = albedo;
    Onb onb = onb_from_normal(normal);
    sr.scattered_dir = onb_transform(&onb, random_cosine_direction(rng));
    sr.pdf_value = vec_dot(normal, sr.scattered_dir) / 3.141592653589793f;
    sr.skip_pdf = false;
    sr.skip_ray_dir = make_float3(0,0,0);
    sr.absorbed = false;
    return sr;
}

// Metal scatter: perfect/slightly-fuzzed reflection
__device__ inline ScatterResult scatter_metal(
    float3 ray_dir, float3 normal, float3 albedo, float fuzz, RngState* rng)
{
    ScatterResult sr;
    sr.attenuation = albedo;
    float3 reflected = vec_reflect(vec_normalize(ray_dir), normal);
    float3 fuzz_dir = scl_mul(fuzz, random_unit_sphere_direction(rng));
    sr.skip_ray_dir = vec_normalize(vec_add(reflected, fuzz_dir));
    sr.skip_pdf = true;
    sr.scattered_dir = make_float3(0,0,0);
    sr.pdf_value = 0.0f;
    sr.absorbed = vec_dot(sr.skip_ray_dir, normal) <= 0.0f;
    return sr;
}

// Dielectric scatter: refraction + Schlick reflection
__device__ inline ScatterResult scatter_dielectric(
    float3 ray_dir, float3 normal, float ir, bool front_face, RngState* rng)
{
    ScatterResult sr;
    sr.attenuation = make_float3(1.0f, 1.0f, 1.0f);
    float refraction_ratio = front_face ? (1.0f / ir) : ir;

    float3 unit_dir = vec_normalize(ray_dir);
    float cos_theta = fminf(vec_dot(make_float3(-unit_dir.x, -unit_dir.y, -unit_dir.z), normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rng_uniform(rng)) {
        // Reflect
        sr.skip_ray_dir = vec_reflect(unit_dir, normal);
    } else {
        // Refract
        sr.skip_ray_dir = vec_refract(unit_dir, normal, refraction_ratio);
    }
    sr.skip_pdf = true;
    sr.scattered_dir = make_float3(0,0,0);
    sr.pdf_value = 0.0f;
    sr.absorbed = false;
    return sr;
}

// Isotropic scatter: uniform sphere
__device__ inline ScatterResult scatter_isotropic(
    float3 albedo, RngState* rng)
{
    ScatterResult sr;
    sr.attenuation = albedo;
    sr.scattered_dir = random_unit_sphere_direction(rng);
    sr.pdf_value = 1.0f / (4.0f * 3.141592653589793f);
    sr.skip_pdf = false;
    sr.skip_ray_dir = make_float3(0,0,0);
    sr.absorbed = false;
    return sr;
}

#endif // GPU_MATERIALS_H
