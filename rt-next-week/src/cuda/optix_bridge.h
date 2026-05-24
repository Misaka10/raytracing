#ifndef OPTIX_BRIDGE_H
#define OPTIX_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct OptiXBridge OptiXBridge;

/* Camera parameters matching GPU CameraParams */
typedef struct {
    float lookfrom[3];
    float lookat[3];
    float vup[3];
    float vfov;
    float aspect_ratio;
    float defocus_angle;
    float focus_dist;
    float u[3], v[3], w[3];
    float pixel00_loc[3];
    float pixel_delta_u[3];
    float pixel_delta_v[3];
    float defocus_disk_u[3];
    float defocus_disk_v[3];
} BridgeCameraParams;

/* Initialize OptiX + CUDA context.
   PTX strings are null-terminated source for each shader program.
   Returns NULL on failure; call optix_bridge_get_error() for details. */
OptiXBridge* optix_bridge_init(
    const char* ptx_raygen,
    const char* ptx_closesthit,
    const char* ptx_miss
);

/* Destroy bridge and free all GPU resources. */
void optix_bridge_destroy(OptiXBridge* bridge);

/* Build triangle acceleration structure (RT Core hardware BVH).
   vertices: array of 3*float per vertex, interleaved xyz
   indices: array of 3*uint per triangle
   tri_count: number of triangles
   Returns true on success. */
bool optix_bridge_build_accel(
    OptiXBridge* bridge,
    const float* vertices,
    const unsigned int* indices,
    int tri_count
);

/* Create the OptiX pipeline (raygen + closesthit + miss).
   Must be called after build_accel.
   width, height: output image dimensions. */
bool optix_bridge_create_pipeline(
    OptiXBridge* bridge,
    int width,
    int height
);

/* Launch the render.
   output: pre-allocated float buffer (width * height * 3 floats, RGB interleaved).
   camera: camera parameters (matches CPU Camera).
   seed: RNG seed (0 = random).
   Returns true on success. */
bool optix_bridge_render(
    OptiXBridge* bridge,
    float* output,
    const BridgeCameraParams* camera,
    unsigned int seed
);

/* Get last error message. */
const char* optix_bridge_get_error(const OptiXBridge* bridge);

#ifdef __cplusplus
}
#endif

#endif /* OPTIX_BRIDGE_H */
