#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <optix.h>

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
    float3                  background;
    CameraParams            camera;
    float3*                 framebuffer;
    OptixTraversableHandle  traversable;
};

// CUDA float3 helpers (float3 lacks operator overloading for scalar ops)
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

__device__ inline float vec_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 vec_normalize(float3 v) {
    float len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    float inv = rsqrtf(len_sq);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

// Payload: all registers must be unsigned int in OptiX 9.x
// Registers: [miss, hit_x, hit_y, hit_z, normal_x, normal_y, normal_z, mat_id]
#define PAYLOAD_REGS 8

#endif // GPU_COMMON_H
