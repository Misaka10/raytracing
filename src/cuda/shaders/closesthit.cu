#include "common.h"
#include <optix_device.h>

extern "C" __global__ void __closesthit__ch() {
    // OptiX 9.x: set individual payload registers (all unsigned int)
    optixSetPayload_0(0); // miss = false

    // Hit point (world space)
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir    = optixGetWorldRayDirection();
    const float  t          = optixGetRayTmax();
    float3 hit = vec_add(ray_origin, scl_mul(t, ray_dir));
    optixSetPayload_1(__float_as_uint(hit.x));
    optixSetPayload_2(__float_as_uint(hit.y));
    optixSetPayload_3(__float_as_uint(hit.z));

    // Barycentrics stored in normal fields (Phase 1 visual debug)
    const float2 bary = optixGetTriangleBarycentrics();
    float u = bary.x;
    float v = bary.y;
    float w = 1.0f - u - v;
    optixSetPayload_4(__float_as_uint(u));
    optixSetPayload_5(__float_as_uint(v));
    optixSetPayload_6(__float_as_uint(w));

    // Material ID (placeholder — Phase 2 fills from SBT data)
    optixSetPayload_7(0);
}
