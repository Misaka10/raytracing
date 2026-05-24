#include "optix_bridge.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// GPU-side launch params (must match shaders/common.h)
// ============================================================================

struct GpuFloat3 { float x, y, z; };

struct GpuCameraParams {
    GpuFloat3 lookfrom, lookat, vup;
    float vfov, aspect_ratio, defocus_angle, focus_dist;
    GpuFloat3 u, v, w;
    GpuFloat3 pixel00_loc;
    GpuFloat3 pixel_delta_u, pixel_delta_v;
    GpuFloat3 defocus_disk_u, defocus_disk_v;
};

struct GpuLaunchParams {
    unsigned int            width;
    unsigned int            height;
    unsigned int            seed;
    GpuFloat3               background;
    GpuCameraParams         camera;
    GpuFloat3*              framebuffer;
    OptixTraversableHandle  traversable;
};

// ============================================================================
// SBT record structures
// ============================================================================

struct SbtRecordHeader {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) unsigned char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct RaygenSbtRecord {
    SbtRecordHeader header;
};

struct MissSbtRecord {
    SbtRecordHeader header;
};

struct HitgroupSbtRecord {
    SbtRecordHeader header;
};

// ============================================================================
// OptiXBridge state
// ============================================================================

struct OptiXBridge {
    // CUDA
    CUcontext            cuCtx;
    CUstream             stream;

    // OptiX
    OptixDeviceContext   optixCtx;

    // Pipeline components
    OptixModule                  module;
    OptixProgramGroup            raygenPG;
    OptixProgramGroup            missPG;
    OptixProgramGroup            hitgroupPG;
    OptixPipeline                pipeline;

    // SBT
    RaygenSbtRecord              sbtRaygen;
    MissSbtRecord                sbtMiss;
    HitgroupSbtRecord            sbtHitgroup;
    CUdeviceptr                  d_sbtRaygen;
    CUdeviceptr                  d_sbtMiss;
    CUdeviceptr                  d_sbtHitgroup;

    // Acceleration structure
    OptixTraversableHandle       gasHandle;
    CUdeviceptr                  d_gasBuffer;
    CUdeviceptr                  d_vertexBuffer;
    CUdeviceptr                  d_indexBuffer;
    bool                         hasAccel;

    // Output
    CUdeviceptr                  d_output;
    CUdeviceptr                  d_launchParams;

    int                          width;
    int                          height;

    char                         errorMsg[512];
};

// ============================================================================
// Helpers
// ============================================================================

static void setError(OptiXBridge* b, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b->errorMsg, sizeof(b->errorMsg), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[OptiXBridge] ERROR: %s\n", b->errorMsg);
}

// Convert host camera params to GPU layout
static void fillGpuCamera(const BridgeCameraParams* src, GpuCameraParams* dst) {
    memcpy(&dst->lookfrom, src->lookfrom, sizeof(float) * 3);
    memcpy(&dst->lookat, src->lookat, sizeof(float) * 3);
    memcpy(&dst->vup, src->vup, sizeof(float) * 3);
    dst->vfov = src->vfov;
    dst->aspect_ratio = src->aspect_ratio;
    dst->defocus_angle = src->defocus_angle;
    dst->focus_dist = src->focus_dist;
    memcpy(&dst->u, src->u, sizeof(float) * 3);
    memcpy(&dst->v, src->v, sizeof(float) * 3);
    memcpy(&dst->w, src->w, sizeof(float) * 3);
    memcpy(&dst->pixel00_loc, src->pixel00_loc, sizeof(float) * 3);
    memcpy(&dst->pixel_delta_u, src->pixel_delta_u, sizeof(float) * 3);
    memcpy(&dst->pixel_delta_v, src->pixel_delta_v, sizeof(float) * 3);
    memcpy(&dst->defocus_disk_u, src->defocus_disk_u, sizeof(float) * 3);
    memcpy(&dst->defocus_disk_v, src->defocus_disk_v, sizeof(float) * 3);
}

// ============================================================================
// CUDA check macro
// ============================================================================

#define CUDA_CHECK(call) do { \
    CUresult _err = (call); \
    if (_err != CUDA_SUCCESS) { \
        const char* _name; \
        cuGetErrorName(_err, &_name); \
        setError(bridge, "CUDA error %s at %s:%d", _name, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define CUDA_CHECK_FREE(call) do { \
    CUresult _err = (call); \
    if (_err != CUDA_SUCCESS) { \
        const char* _name; \
        cuGetErrorName(_err, &_name); \
        setError(bridge, "CUDA error %s at %s:%d", _name, __FILE__, __LINE__); \
    } \
} while(0)

#define OPTIX_CHECK(call) do { \
    OptixResult _res = (call); \
    if (_res != OPTIX_SUCCESS) { \
        setError(bridge, "OptiX error %d at %s:%d", (int)_res, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define OPTIX_CHECK_LOG(call) do { \
    OptixResult _res = (call); \
    if (_res != OPTIX_SUCCESS) { \
        setError(bridge, "OptiX error %d at %s:%d: %s", (int)_res, __FILE__, __LINE__, log); \
        return false; \
    } \
} while(0)

// ============================================================================
// Public API Implementation
// ============================================================================

OptiXBridge* optix_bridge_init(
    const char* ptx_raygen,
    const char* ptx_closesthit,
    const char* ptx_miss)
{
    OptiXBridge* bridge = (OptiXBridge*)calloc(1, sizeof(OptiXBridge));
    if (!bridge) return NULL;

    bridge->hasAccel = false;
    bridge->width = 0;
    bridge->height = 0;
    bridge->d_output = 0;
    bridge->d_launchParams = 0;
    bridge->d_vertexBuffer = 0;
    bridge->d_indexBuffer = 0;
    bridge->d_gasBuffer = 0;
    bridge->d_sbtRaygen = 0;
    bridge->d_sbtMiss = 0;
    bridge->d_sbtHitgroup = 0;

    // --- CUDA init ---
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) {
        setError(bridge, "cuInit failed: %d. Is CUDA driver installed?", (int)cuErr);
        free(bridge);
        return NULL;
    }

    int deviceCount = 0;
    CUDA_CHECK(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        setError(bridge, "No CUDA-capable device found.");
        free(bridge);
        return NULL;
    }

    CUdevice cuDevice;
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));

    // Print device name for diagnostics
    char deviceName[256];
    CUDA_CHECK_FREE(cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice));
    fprintf(stderr, "[OptiXBridge] Using CUDA device: %s\n", deviceName);

    CUDA_CHECK(cuCtxCreate(&bridge->cuCtx, CU_CTX_SCHED_SPIN, cuDevice));
    CUDA_CHECK(cuStreamCreate(&bridge->stream, CU_STREAM_DEFAULT));

    // --- OptiX init ---
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions optixOpts = {};
    optixOpts.logCallbackFunction = NULL;
    optixOpts.logCallbackLevel = 3; // warnings + errors

    OPTIX_CHECK(optixDeviceContextCreate(bridge->cuCtx, &optixOpts, &bridge->optixCtx));

    // --- Build pipeline ---

    // Pipeline compile options
    OptixPipelineCompileOptions pipelineCompileOpts = {};
    pipelineCompileOpts.usesMotionBlur = false;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOpts.numPayloadValues = 8; // match our Payload struct register count
    pipelineCompileOpts.numAttributeValues = 2; // barycentrics
    pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "launch_params";

    // Create module from embedded PTX
    OptixModuleCompileOptions moduleCompileOpts = {};
    moduleCompileOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    char log[2048];
    size_t logSize = sizeof(log);

    // Compile single module containing all programs
    // We combine PTX by including all in one module
    size_t ptxRaygenLen = strlen(ptx_raygen);
    size_t ptxChLen = strlen(ptx_closesthit);
    size_t ptxMsLen = strlen(ptx_miss);

    // Allocate buffer for combined PTX with separators
    size_t combinedLen = ptxRaygenLen + ptxChLen + ptxMsLen + 3;
    char* combinedPtx = (char*)malloc(combinedLen + 1);
    if (!combinedPtx) {
        setError(bridge, "Failed to allocate PTX buffer");
        optix_bridge_destroy(bridge);
        return NULL;
    }

    size_t off = 0;
    memcpy(combinedPtx + off, ptx_raygen, ptxRaygenLen); off += ptxRaygenLen;
    combinedPtx[off++] = '\n';
    memcpy(combinedPtx + off, ptx_closesthit, ptxChLen); off += ptxChLen;
    combinedPtx[off++] = '\n';
    memcpy(combinedPtx + off, ptx_miss, ptxMsLen); off += ptxMsLen;
    combinedPtx[off] = '\0';

    OptixResult modResult = optixModuleCreate(
        bridge->optixCtx,
        &moduleCompileOpts,
        &pipelineCompileOpts,
        combinedPtx,
        off,
        log, &logSize,
        &bridge->module
    );
    free(combinedPtx);

    if (modResult != OPTIX_SUCCESS) {
        setError(bridge, "optixModuleCreate failed: %s", log);
        optix_bridge_destroy(bridge);
        return NULL;
    }

    // Create program groups
    {
        OptixProgramGroupOptions pgOpts = {};

        OptixProgramGroupDesc raygenDesc = {};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = bridge->module;
        raygenDesc.raygen.entryFunctionName = "__raygen__rg";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            bridge->optixCtx, &raygenDesc, 1, &pgOpts,
            log, &logSize, &bridge->raygenPG
        ));

        OptixProgramGroupDesc missDesc = {};
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.miss.module = bridge->module;
        missDesc.miss.entryFunctionName = "__miss__ms";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            bridge->optixCtx, &missDesc, 1, &pgOpts,
            log, &logSize, &bridge->missPG
        ));

        OptixProgramGroupDesc hitDesc = {};
        hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc.hitgroup.moduleCH = bridge->module;
        hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hitDesc.hitgroup.moduleAH = NULL;
        hitDesc.hitgroup.entryFunctionNameAH = NULL;

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            bridge->optixCtx, &hitDesc, 1, &pgOpts,
            log, &logSize, &bridge->hitgroupPG
        ));
    }

    // Link pipeline
    OptixProgramGroup pgList[] = {
        bridge->raygenPG,
        bridge->missPG,
        bridge->hitgroupPG
    };

    OptixPipelineLinkOptions pipelineLinkOpts = {};
    pipelineLinkOpts.maxTraceDepth = 31;
    pipelineLinkOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        bridge->optixCtx,
        &pipelineCompileOpts,
        &pipelineLinkOpts,
        pgList,
        sizeof(pgList) / sizeof(pgList[0]),
        log, &logSize,
        &bridge->pipeline
    ));

    // Compute stack sizes
    OptixStackSizes stackSizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(bridge->raygenPG, &stackSizes, bridge->pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(bridge->missPG, &stackSizes, bridge->pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(bridge->hitgroupPG, &stackSizes, bridge->pipeline));

    unsigned int directStackSize = 0, continuationStackSize = 0, maxTraversableDepth = 1;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        maxTraversableDepth,
        0, // maxCCDepth
        0, // maxDCDepth
        &directStackSize,
        &continuationStackSize
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        bridge->pipeline,
        directStackSize,
        continuationStackSize,
        maxTraversableDepth
    ));

    // Build SBT records
    OPTIX_CHECK(optixSbtRecordPackHeader(bridge->raygenPG, &bridge->sbtRaygen));
    OPTIX_CHECK(optixSbtRecordPackHeader(bridge->missPG, &bridge->sbtMiss));
    OPTIX_CHECK(optixSbtRecordPackHeader(bridge->hitgroupPG, &bridge->sbtHitgroup));

    // Upload SBT records to device
    CUDA_CHECK(cuMemAlloc(&bridge->d_sbtRaygen, sizeof(RaygenSbtRecord)));
    CUDA_CHECK(cuMemAlloc(&bridge->d_sbtMiss, sizeof(MissSbtRecord)));
    CUDA_CHECK(cuMemAlloc(&bridge->d_sbtHitgroup, sizeof(HitgroupSbtRecord)));

    CUDA_CHECK(cuMemcpyHtoD(bridge->d_sbtRaygen, &bridge->sbtRaygen, sizeof(RaygenSbtRecord)));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_sbtMiss, &bridge->sbtMiss, sizeof(MissSbtRecord)));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_sbtHitgroup, &bridge->sbtHitgroup, sizeof(HitgroupSbtRecord)));

    fprintf(stderr, "[OptiXBridge] Initialized successfully\n");
    return bridge;
}

void optix_bridge_destroy(OptiXBridge* bridge) {
    if (!bridge) return;

    if (bridge->d_output)      cuMemFree(bridge->d_output);
    if (bridge->d_launchParams) cuMemFree(bridge->d_launchParams);
    if (bridge->d_vertexBuffer) cuMemFree(bridge->d_vertexBuffer);
    if (bridge->d_indexBuffer)  cuMemFree(bridge->d_indexBuffer);
    if (bridge->d_gasBuffer)    cuMemFree(bridge->d_gasBuffer);
    if (bridge->d_sbtRaygen)    cuMemFree(bridge->d_sbtRaygen);
    if (bridge->d_sbtMiss)      cuMemFree(bridge->d_sbtMiss);
    if (bridge->d_sbtHitgroup)  cuMemFree(bridge->d_sbtHitgroup);

    if (bridge->pipeline)       optixPipelineDestroy(bridge->pipeline);
    if (bridge->raygenPG)       optixProgramGroupDestroy(bridge->raygenPG);
    if (bridge->missPG)         optixProgramGroupDestroy(bridge->missPG);
    if (bridge->hitgroupPG)     optixProgramGroupDestroy(bridge->hitgroupPG);
    if (bridge->module)         optixModuleDestroy(bridge->module);
    if (bridge->optixCtx)       optixDeviceContextDestroy(bridge->optixCtx);

    if (bridge->stream)         cuStreamDestroy(bridge->stream);
    if (bridge->cuCtx)          cuCtxDestroy(bridge->cuCtx);

    free(bridge);
}

bool optix_bridge_build_accel(
    OptiXBridge* bridge,
    const float* vertices,
    const int* indices,
    int tri_count)
{
    if (!bridge) return false;

    const size_t vertexSize = tri_count * 3 * 3 * sizeof(float);
    const size_t indexSize  = tri_count * 3 * sizeof(int);

    // Upload vertex and index data
    CUDA_CHECK(cuMemAlloc(&bridge->d_vertexBuffer, vertexSize));
    CUDA_CHECK(cuMemAlloc(&bridge->d_indexBuffer, indexSize));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_vertexBuffer, vertices, vertexSize));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_indexBuffer, indices, indexSize));

    // Build acceleration structure (RT Core hardware BVH)
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.numVertices = (unsigned int)(tri_count * 3);
    buildInput.triangleArray.vertexBuffers = &bridge->d_vertexBuffer;
    buildInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.numIndexTriplets = (unsigned int)tri_count;
    buildInput.triangleArray.indexBuffer = bridge->d_indexBuffer;
    buildInput.triangleArray.flags = NULL; // one flag per triangle, NULL = all 0
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOpts = {};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        bridge->optixCtx,
        &accelOpts,
        &buildInput,
        1, // num inputs
        &bufferSizes
    ));

    // Compacted size buffer
    CUdeviceptr d_temp;
    CUDA_CHECK(cuMemAlloc(&d_temp, bufferSizes.tempSizeInBytes));

    // Output buffer
    CUdeviceptr d_outputAccel;
    CUDA_CHECK(cuMemAlloc(&d_outputAccel, bufferSizes.outputSizeInBytes));

    // Compacted size (uint64)
    CUdeviceptr d_compactedSize;
    CUDA_CHECK(cuMemAlloc(&d_compactedSize, sizeof(unsigned long long)));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = d_compactedSize;

    OPTIX_CHECK(optixAccelBuild(
        bridge->optixCtx,
        bridge->stream,
        &accelOpts,
        &buildInput,
        1,
        d_temp,
        bufferSizes.tempSizeInBytes,
        d_outputAccel,
        bufferSizes.outputSizeInBytes,
        &bridge->gasHandle,
        &emitDesc,
        1
    ));

    CUDA_CHECK(cuStreamSynchronize(bridge->stream));

    // Compact the acceleration structure
    unsigned long long compactedSize = 0;
    CUDA_CHECK(cuMemcpyDtoH(&compactedSize, d_compactedSize, sizeof(unsigned long long)));

    CUDA_CHECK(cuMemAlloc(&bridge->d_gasBuffer, (size_t)compactedSize));
    OPTIX_CHECK(optixAccelCompact(
        bridge->optixCtx,
        bridge->stream,
        bridge->gasHandle,
        bridge->d_gasBuffer,
        (size_t)compactedSize,
        &bridge->gasHandle
    ));

    CUDA_CHECK(cuStreamSynchronize(bridge->stream));

    // Free temp buffers
    CUDA_CHECK_FREE(cuMemFree(d_temp));
    CUDA_CHECK_FREE(cuMemFree(d_outputAccel));
    CUDA_CHECK_FREE(cuMemFree(d_compactedSize));

    bridge->hasAccel = true;
    fprintf(stderr, "[OptiXBridge] BVH built: %d triangles, %llu bytes (RT Core hardware)\n",
            tri_count, compactedSize);
    return true;
}

bool optix_bridge_create_pipeline(
    OptiXBridge* bridge,
    int width,
    int height)
{
    if (!bridge || !bridge->hasAccel) return false;

    bridge->width = width;
    bridge->height = height;

    const size_t outputSize = width * height * 3 * sizeof(float);

    // Allocate output buffer
    if (bridge->d_output) cuMemFree(bridge->d_output);
    CUDA_CHECK(cuMemAlloc(&bridge->d_output, outputSize));

    // Allocate launch params buffer
    if (bridge->d_launchParams) cuMemFree(bridge->d_launchParams);
    CUDA_CHECK(cuMemAlloc(&bridge->d_launchParams, sizeof(GpuLaunchParams)));

    fprintf(stderr, "[OptiXBridge] Pipeline created: %dx%d, output buffer %zu MB\n",
            width, height, outputSize / (1024 * 1024));
    return true;
}

bool optix_bridge_render(
    OptiXBridge* bridge,
    float* output,
    const BridgeCameraParams* camera,
    unsigned int seed)
{
    if (!bridge || !bridge->hasAccel || bridge->width == 0) return false;

    // Prepare launch params
    GpuLaunchParams params = {};
    params.width  = (unsigned int)bridge->width;
    params.height = (unsigned int)bridge->height;
    params.seed   = seed;
    params.background = { 0.0f, 0.0f, 0.0f };
    params.framebuffer = (GpuFloat3*)bridge->d_output;
    params.traversable = bridge->gasHandle;
    fillGpuCamera(camera, &params.camera);

    CUDA_CHECK(cuMemcpyHtoD(bridge->d_launchParams, &params, sizeof(GpuLaunchParams)));

    // Setup SBT
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord                = bridge->d_sbtRaygen;
    sbt.missRecordBase              = bridge->d_sbtMiss;
    sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = bridge->d_sbtHitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupSbtRecord);
    sbt.hitgroupRecordCount         = 1;

    // Launch
    OPTIX_CHECK(optixLaunch(
        bridge->pipeline,
        bridge->stream,
        bridge->d_launchParams,
        sizeof(GpuLaunchParams),
        &sbt,
        (unsigned int)bridge->width,
        (unsigned int)bridge->height,
        1 // depth = 1 (only raygen launches rays)
    ));

    CUDA_CHECK(cuStreamSynchronize(bridge->stream));

    // Download result
    const size_t outputSize = bridge->width * bridge->height * 3 * sizeof(float);
    CUDA_CHECK(cuMemcpyDtoH(output, bridge->d_output, outputSize));

    return true;
}

const char* optix_bridge_get_error(const OptiXBridge* bridge) {
    if (!bridge) return "Null bridge pointer";
    return bridge->errorMsg;
}
