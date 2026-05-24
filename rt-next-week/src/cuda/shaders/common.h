#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <optix.h>

// Material type constants (must match scene.rs GpuMaterial::mat_type)
#define MAT_LAMBERTIAN    0
#define MAT_METAL         1
#define MAT_DIELECTRIC    2
#define MAT_DIFFUSE_LIGHT 3
#define MAT_ISOTROPIC     4

struct GpuMaterialData {
    unsigned int mat_type;
    float3 albedo;
    float  fuzz;
    float  ir;
    float3 emission;
};

struct CameraParams {
    float3 lookfrom;
    float3 lookat;
    float3 vup;
    float  vfov;
    float  aspect_ratio;
    float  defocus_angle;
    float  focus_dist;
    float3 u, v, w;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    float3 defocus_disk_u;
    float3 defocus_disk_v;
};

struct LaunchParams {
    unsigned int            width;
    unsigned int            height;
    unsigned int            seed;
    unsigned int            sqrt_spp;
    unsigned int            max_depth;
    float                   pixel_samples_scale;
    float3                  background;
    CameraParams            camera;
    float3*                 framebuffer;
    GpuMaterialData*        materials;
    unsigned int            material_count;
    float3*                 vertex_buffer;     // triangle vertices (3*N floats)
    unsigned int*           index_buffer;      // triangle indices (3*M uints)
    unsigned int*           tri_material;      // per-triangle material index
    OptixTraversableHandle  traversable;
};

// CUDA float3 helpers
__device__ inline float3 scl_mul(float s, float3 v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ inline float3 scl_div(float3 v, float s) {
    float inv = 1.0f / s;
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ inline float3 vec_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 vec_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 vec_mul(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float vec_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 vec_cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float vec_length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float3 vec_normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    float inv = 1.0f / len;
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ inline float3 vec_reflect(float3 v, float3 n) {
    float d = 2.0f * vec_dot(v, n);
    return make_float3(v.x - d * n.x, v.y - d * n.y, v.z - d * n.z);
}

__device__ inline float3 vec_refract(float3 uv, float3 n, float etai_over_etat) {
    float cos_theta = fminf(vec_dot(make_float3(-uv.x, -uv.y, -uv.z), n), 1.0f);
    float3 r_out_perp = scl_mul(etai_over_etat, vec_add(uv, scl_mul(cos_theta, n)));
    float3 r_out_parallel = scl_mul(-sqrtf(fabsf(1.0f - vec_dot(r_out_perp, r_out_perp))), n);
    return vec_add(r_out_perp, r_out_parallel);
}

__device__ inline float vec_max_component(float3 v) {
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

__device__ inline bool vec_is_zero(float3 v) {
    return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
}

// Payload: all registers must be unsigned int in OptiX 9.x
// Registers: [miss, hit_x, hit_y, hit_z, normal/bary_u, normal/bary_v, normal/bary_w, mat_id]
#define PAYLOAD_REGS 8

#endif // GPU_COMMON_H
