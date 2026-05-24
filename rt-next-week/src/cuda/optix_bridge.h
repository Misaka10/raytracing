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

/* Upload material data to GPU. Must be called before render.
   materials: array of material structs (36 bytes each, matches GpuMaterialData)
   count: number of materials
   Returns true on success. */
bool optix_bridge_set_materials(
    OptiXBridge* bridge,
    const void* materials,
    unsigned int count
);

/* Upload per-triangle material index data.
   tri_material: array of uint, one per triangle
   tri_count: number of triangles
   Returns true on success. */
bool optix_bridge_set_tri_material(
    OptiXBridge* bridge,
    const unsigned int* tri_material,
    int tri_count
);

/* Set render parameters (samples per pixel, max depth).
   Must be called before render. */
bool optix_bridge_set_render_params(
    OptiXBridge* bridge,
    unsigned int sqrt_spp,
    unsigned int max_depth,
    float pixel_samples_scale
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

/* Apply OptiX AI denoiser (Tensor Core accelerated, HDR model).
   Denoises the last rendered frame in-place (modifies d_output).
   Must be called after render. Returns true on success. */
bool optix_bridge_denoise(OptiXBridge* bridge);

/* Get last error message. */
const char* optix_bridge_get_error(const OptiXBridge* bridge);

#ifdef __cplusplus
}
#endif

#endif /* OPTIX_BRIDGE_H */
