#include "common.h"
#include <optix_device.h>

extern "C" {
__constant__ LaunchParams launch_params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel_idx = idx.y * launch_params.width + idx.x;

    const CameraParams& cam = launch_params.camera;

    // Generate camera ray matching CPU get_ray()
    const float u = (float(idx.x) + 0.5f) / float(launch_params.width);
    const float v = (float(idx.y) + 0.5f) / float(launch_params.height);

    float3 ray_origin = cam.lookfrom;
    float3 ray_dir = vec_sub(
        vec_add(cam.pixel00_loc,
            vec_add(scl_mul(u, cam.pixel_delta_u), scl_mul(v, cam.pixel_delta_v))),
        cam.lookfrom);
    ray_dir = vec_normalize(ray_dir);

    // Payload: ALL registers must be unsigned int (OptiX 9.x hard requirement)
    // Layout: [miss_flag, hit_x, hit_y, hit_z, normal/bary_u, normal/bary_v, normal/bary_w, mat_id]
    unsigned int p0 = 1; // miss = true
    unsigned int p1 = 0, p2 = 0, p3 = 0;
    unsigned int p4 = 0, p5 = 0, p6 = 0;
    unsigned int p7 = 0;

    optixTrace(
        launch_params.traversable,
        ray_origin, ray_dir,
        0.001f, 1e20f, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 0, 0,
        p0, p1, p2, p3, p4, p5, p6, p7
    );

    float3 result;
    if (p0 == 1) {
        // Miss → background
        result = launch_params.background;
    } else {
        // Hit → barycentrics as RGB (visual debug for Phase 1)
        result = make_float3(
            __uint_as_float(p4),
            __uint_as_float(p5),
            __uint_as_float(p6)
        );
    }

    launch_params.framebuffer[pixel_idx] = result;
}
