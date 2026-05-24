#include "common.h"
#include <optix_device.h>

extern "C" {
__constant__ LaunchParams launch_params;
}

extern "C" __global__ void __closesthit__ch() {
    // OptiX 9.x: set individual payload registers (all unsigned int)
    optixSetPayload_0(0); // miss = false

    // Hit point (world space) — OptiX returns built-in float3, convert to GpuFloat3
    float3 _ro = optixGetWorldRayOrigin();
    float3 _rd = optixGetWorldRayDirection();
    GpuFloat3 ray_origin = {_ro.x, _ro.y, _ro.z};
    GpuFloat3 ray_dir    = {_rd.x, _rd.y, _rd.z};
    const float  t       = optixGetRayTmax();
    GpuFloat3 hit = vec_add(ray_origin, scl_mul(t, ray_dir));
    optixSetPayload_1(__float_as_uint(hit.x));
    optixSetPayload_2(__float_as_uint(hit.y));
    optixSetPayload_3(__float_as_uint(hit.z));

    // Smooth normal via barycentric interpolation of vertex normals
    // (for quads all 3 vertex normals are identical → interpolated result = face normal)
    unsigned int prim_idx = optixGetPrimitiveIndex();
    unsigned int i0 = launch_params.index_buffer[prim_idx * 3 + 0];
    unsigned int i1 = launch_params.index_buffer[prim_idx * 3 + 1];
    unsigned int i2 = launch_params.index_buffer[prim_idx * 3 + 2];

    GpuFloat3 n0 = launch_params.normal_buffer[i0];
    GpuFloat3 n1 = launch_params.normal_buffer[i1];
    GpuFloat3 n2 = launch_params.normal_buffer[i2];

    float2 bary = optixGetTriangleBarycentrics();
    float beta  = bary.x;
    float gamma = bary.y;
    float alpha = 1.0f - beta - gamma;

    GpuFloat3 normal = vec_add(
        vec_add(scl_mul(alpha, n0), scl_mul(beta, n1)),
        scl_mul(gamma, n2)
    );
    normal = vec_normalize(normal);

    optixSetPayload_4(__float_as_uint(normal.x));
    optixSetPayload_5(__float_as_uint(normal.y));
    optixSetPayload_6(__float_as_uint(normal.z));

    // Material ID from per-triangle buffer
    unsigned int mat_id = 0;
    if (launch_params.tri_material) {
        mat_id = launch_params.tri_material[prim_idx];
    }
    optixSetPayload_7(mat_id);
}
