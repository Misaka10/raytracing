#include "common.h"
#include <optix_device.h>

extern "C" {
__constant__ LaunchParams launch_params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel_idx = idx.y * launch_params.width + idx.x;

    // Generate camera ray matching CPU get_ray()
    const float u = (float(idx.x) + 0.5f) / float(launch_params.width);
    const float v = (float(idx.y) + 0.5f) / float(launch_params.height);

    const CameraParams& cam = launch_params.camera;
    const float3 lookfrom = cam.lookfrom;
    float3 ray_dir = cam.pixel00_loc
        + u * cam.pixel_delta_u
        + v * cam.pixel_delta_v
        - cam.lookfrom;
    ray_dir = normalize(ray_dir);

    // Initialize payload: 8 x 4-byte registers
    // Layout: [miss_flag, hit_x, hit_y, hit_z, normal/bary_u, normal/bary_v, normal/bary_w, material_id]
    unsigned int p0 = 1; // miss = true
    float p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    float p4 = 0.0f, p5 = 0.0f, p6 = 0.0f;
    unsigned int p7 = 0;

    optixTrace(
        launch_params.traversable,
        lookfrom, ray_dir,
        0.001f, 1e20f, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 0, 0,
        p0, p1, p2, p3, p4, p5, p6, p7
    );

    float3 result;
    if (p0 == 1) {
        // Miss → black background
        result = launch_params.background;
    } else {
        // Hit → barycentrics as RGB (proves RT Core BVH + intersection works)
        result = make_float3(p4, p5, p6);
    }

    launch_params.framebuffer[pixel_idx] = result;
}
