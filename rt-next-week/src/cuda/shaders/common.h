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

#endif // GPU_COMMON_H
