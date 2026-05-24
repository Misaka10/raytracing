#include "common.h"
#include <optix_device.h>

extern "C" __global__ void __miss__ms() {
    // OptiX 9.x: set payload register 0 to indicate miss
    optixSetPayload_0(1); // miss = true
}
