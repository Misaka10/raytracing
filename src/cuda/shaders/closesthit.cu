#include "common.h"
#include <optix_device.h>

extern "C" __global__ void __closesthit__ch() {
    // Access payload registers by pointer
    unsigned int* p = (unsigned int*)optixGetPayloadPointer();

    p[0] = 0; // miss = false

    // Hit point (world space)
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir    = optixGetWorldRayDirection();
    const float  t          = optixGetRayTmax();
    float3 hit = ray_origin + t * ray_dir;
    ((float*)(p + 1))[0] = hit.x;
    ((float*)(p + 1))[1] = hit.y;
    ((float*)(p + 1))[2] = hit.z;

    // Barycentrics → stored in normal fields (for visual debugging in Phase 1)
    const float2 bary = optixGetTriangleBarycentrics();
    float u = bary.x;
    float v = bary.y;
    float w = 1.0f - u - v;

    ((float*)(p + 4))[0] = u;
    ((float*)(p + 4))[1] = v;
    ((float*)(p + 4))[2] = w;

    // Material ID (placeholder)
    p[7] = 0;
}
