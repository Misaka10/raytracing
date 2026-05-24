#include "common.h"
#include <optix_device.h>

extern "C" __global__ void __miss__ms() {
    // Set miss flag via payload pointer
    unsigned int* p = (unsigned int*)optixGetPayloadPointer();
    p[0] = 1; // miss = true
}
