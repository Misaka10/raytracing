#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <optix.h>

// 4-byte aligned float3 — matches host-side GpuFloat3 layout exactly.
// CUDA built-in float3 has __align__(8) on device (sm >= 2.0), which
// would cause struct layout mismatches with host-side data.
typedef struct { float x, y, z; } GpuFloat3;

// Material type constants (must match scene.rs GpuMaterial::mat_type)
#define MAT_LAMBERTIAN    0
#define MAT_METAL         1
#define MAT_DIELECTRIC    2
#define MAT_DIFFUSE_LIGHT 3
#define MAT_ISOTROPIC     4

struct GpuMaterialData {
    unsigned int mat_type;
    GpuFloat3 albedo;
    float  fuzz;
    float  ir;
    GpuFloat3 emission;
};

struct CameraParams {
    GpuFloat3 lookfrom;
    GpuFloat3 lookat;
    GpuFloat3 vup;
    float  vfov;
    float  aspect_ratio;
    float  defocus_angle;
    float  focus_dist;
    GpuFloat3 u, v, w;
    GpuFloat3 pixel00_loc;
    GpuFloat3 pixel_delta_u;
    GpuFloat3 pixel_delta_v;
    GpuFloat3 defocus_disk_u;
    GpuFloat3 defocus_disk_v;
};

struct LaunchParams {
    unsigned int            width;
    unsigned int            height;
    unsigned int            seed;
    unsigned int            sqrt_spp;
    unsigned int            max_depth;
    float                   pixel_samples_scale;
    GpuFloat3               background;
    CameraParams            camera;
    GpuFloat3*              framebuffer;
    GpuMaterialData*        materials;
    unsigned int            material_count;
    GpuFloat3*              vertex_buffer;     // triangle vertices (3*N floats)
    GpuFloat3*              normal_buffer;     // per-vertex normals (same indexing as vertex_buffer)
    unsigned int*           index_buffer;      // triangle indices (3*M uints)
    unsigned int*           tri_material;      // per-triangle material index
    OptixTraversableHandle  traversable;
    // Light sampling (area light rectangle)
    GpuFloat3               light_corner;
    GpuFloat3               light_u;
    GpuFloat3               light_v;
    float                   light_area_inv;
    // Sphere for MIS direction sampling (matching CPU lights list)
    GpuFloat3               sphere_center;
    float                   sphere_radius;
    GpuFloat3*              albedo_buffer;
    GpuFloat3*              guide_normal_buffer;
};

// GpuFloat3 helpers (same semantics as CUDA float3 but 4-byte aligned)
__device__ inline GpuFloat3 scl_mul(float s, GpuFloat3 v) {
    return {s * v.x, s * v.y, s * v.z};
}

__device__ inline GpuFloat3 scl_div(GpuFloat3 v, float s) {
    float inv = 1.0f / s;
    return {v.x * inv, v.y * inv, v.z * inv};
}

__device__ inline GpuFloat3 vec_add(GpuFloat3 a, GpuFloat3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ inline GpuFloat3 vec_sub(GpuFloat3 a, GpuFloat3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline GpuFloat3 vec_mul(GpuFloat3 a, GpuFloat3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ inline float vec_dot(GpuFloat3 a, GpuFloat3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline GpuFloat3 vec_cross(GpuFloat3 a, GpuFloat3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ inline float vec_length(GpuFloat3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline GpuFloat3 vec_normalize(GpuFloat3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

__device__ inline GpuFloat3 vec_reflect(GpuFloat3 v, GpuFloat3 n) {
    float d = 2.0f * vec_dot(v, n);
    return {v.x - d * n.x, v.y - d * n.y, v.z - d * n.z};
}

__device__ inline GpuFloat3 vec_refract(GpuFloat3 uv, GpuFloat3 n, float etai_over_etat) {
    float cos_theta = fminf(vec_dot({-uv.x, -uv.y, -uv.z}, n), 1.0f);
    GpuFloat3 r_out_perp = scl_mul(etai_over_etat, vec_add(uv, scl_mul(cos_theta, n)));
    GpuFloat3 r_out_parallel = scl_mul(-sqrtf(fabsf(1.0f - vec_dot(r_out_perp, r_out_perp))), n);
    return vec_add(r_out_perp, r_out_parallel);
}

__device__ inline float vec_max_component(GpuFloat3 v) {
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

__device__ inline bool vec_is_zero(GpuFloat3 v) {
    return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
}

// Payload: all registers must be unsigned int in OptiX 9.x
// Registers: [miss, hit_x, hit_y, hit_z, normal/bary_u, normal/bary_v, normal/bary_w, mat_id]
#define PAYLOAD_REGS 8

// Compile-time size checks (must match host-side structs in optix_bridge.cu and scene.rs)
// GpuMaterialData: u32 + GpuFloat3 + f32 + f32 + GpuFloat3 = 4 + 12 + 4 + 4 + 12 = 36
static_assert(sizeof(GpuMaterialData) == 36, "GpuMaterialData size must be 36 bytes");
// CameraParams: 11 GpuFloat3 + 4 float = 11*12 + 4*4 = 148
static_assert(sizeof(CameraParams) == 148, "CameraParams size must be 148 bytes");

#endif // GPU_COMMON_H
