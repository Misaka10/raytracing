#include "optix_bridge.h"

#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// GPU 端启动参数（必须与 shaders/common.h 保持一致）
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

// 必须与 shaders/common.h GpuMaterialData 一致（36 字节，4 字节对齐）
struct GpuMaterial {
    unsigned int mat_type;
    GpuFloat3 albedo;
    float  fuzz;
    float  ir;
    GpuFloat3 emission;
};

struct GpuLaunchParams {
    unsigned int            width;
    unsigned int            height;
    unsigned int            seed;
    unsigned int            sqrt_spp;
    unsigned int            max_depth;
    float                   pixel_samples_scale;
    GpuFloat3               background;
    GpuCameraParams         camera;
    GpuFloat3*              framebuffer;
    GpuMaterial*            materials;
    unsigned int            material_count;
    GpuFloat3*              vertex_buffer;
    GpuFloat3*              normal_buffer;
    unsigned int*           index_buffer;
    unsigned int*           tri_material;
    OptixTraversableHandle  traversable;
    // Light sampling
    GpuFloat3               light_corner;
    GpuFloat3               light_u;
    GpuFloat3               light_v;
    float                   light_area_inv;
    // Sphere for MIS
    GpuFloat3               sphere_center;
    float                   sphere_radius;
    GpuFloat3*              albedo_buffer;
    GpuFloat3*              guide_normal_buffer;
};

// 验证主机端结构体大小与 GPU 端 (common.h) 预期一致
static_assert(sizeof(GpuMaterial) == 36, "GpuMaterial must be 36 bytes");
static_assert(sizeof(GpuCameraParams) == 148, "GpuCameraParams must be 148 bytes");

// ============================================================================
// SBT 记录结构体
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
// OptiXBridge 状态
// ============================================================================

struct OptiXBridge {
    // CUDA
    CUcontext            cuCtx;
    CUstream             stream;

    // OptiX
    OptixDeviceContext   optixCtx;

    // Pipeline components
    OptixModule                  moduleRaygen;
    OptixModule                  moduleClosesthit;
    OptixModule                  moduleMiss;
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
    CUdeviceptr                  d_normalBuffer;
    CUdeviceptr                  d_indexBuffer;
    bool                         hasAccel;

    // Output
    CUdeviceptr                  d_output;
    CUdeviceptr                  d_launchParams;

    // Material data
    CUdeviceptr                  d_materials;
    unsigned int                 materialCount;
    CUdeviceptr                  d_triMaterial;  // per-triangle material indices

    int                          width;
    int                          height;

    // Render params (set before render)
    unsigned int                 sqrtSpp;
    unsigned int                 maxDepth;
    float                        pixelScale;

    // Light params
    GpuFloat3                    lightCorner;
    GpuFloat3                    lightU;
    GpuFloat3                    lightV;
    float                        lightAreaInv;
    // Sphere for MIS
    GpuFloat3                    sphereCenter;
    float                        sphereRadius;

    // Denoiser
    OptixDenoiser                denoiser;
    CUdeviceptr                  d_denoiserState;
    size_t                       denoiserStateSize;
    CUdeviceptr                  d_denoiserScratch;
    size_t                       denoiserScratchSize;
    CUdeviceptr                  d_denoisedOutput;
    CUdeviceptr                  d_albedoBuffer;
    CUdeviceptr                  d_guideNormalBuffer;
    int                          denoiserWidth;
    int                          denoiserHeight;
    bool                         denoiserSetup;

    char                         deviceName[256];
    char                         errorMsg[512];
};

// ============================================================================
// 辅助函数
// ============================================================================

static void setError(OptiXBridge* b, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b->errorMsg, sizeof(b->errorMsg), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[OptiXBridge] ERROR: %s\n", b->errorMsg);
}

// 将主机端相机参数复制到 GPU 布局
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
// CUDA 检查宏
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
// 公共 API 实现
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
    bridge->sqrtSpp = 1;
    bridge->maxDepth = 50;
    bridge->pixelScale = 1.0f;
    bridge->denoiser = 0;
    bridge->d_denoiserState = 0;
    bridge->denoiserStateSize = 0;
    bridge->d_denoiserScratch = 0;
    bridge->denoiserScratchSize = 0;
    bridge->d_denoisedOutput = 0;
    bridge->d_albedoBuffer = 0;
    bridge->d_guideNormalBuffer = 0;
    bridge->denoiserWidth = 0;
    bridge->denoiserHeight = 0;
    bridge->denoiserSetup = false;
    bridge->d_output = 0;
    bridge->d_launchParams = 0;
    bridge->d_materials = 0;
    bridge->materialCount = 0;
    bridge->d_triMaterial = 0;
    bridge->d_vertexBuffer = 0;
    bridge->d_normalBuffer = 0;
    bridge->d_indexBuffer = 0;
    bridge->d_gasBuffer = 0;
    bridge->d_sbtRaygen = 0;
    bridge->d_sbtMiss = 0;
    bridge->d_sbtHitgroup = 0;

    // --- CUDA 初始化 ---
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

    // 存储并打印设备名称用于诊断
    CUDA_CHECK_FREE(cuDeviceGetName(bridge->deviceName, sizeof(bridge->deviceName), cuDevice));
    fprintf(stderr, "[OptiXBridge] Using CUDA device: %s\n", bridge->deviceName);

    CUDA_CHECK(cuCtxCreate(&bridge->cuCtx, NULL, 0, cuDevice));
    CUDA_CHECK(cuStreamCreate(&bridge->stream, CU_STREAM_DEFAULT));

    // --- OptiX 初始化 ---
    OPTIX_CHECK(optixInit());

    // OptiX 日志回调，用于详细诊断
    static auto logCallback = [](unsigned int level, const char* tag, const char* msg, void*) {
        fprintf(stderr, "[OptiX][%u][%s] %s\n", level, tag, msg);
    };

    OptixDeviceContextOptions optixOpts = {};
    optixOpts.logCallbackFunction = logCallback;
    optixOpts.logCallbackLevel = 4; // 输出所有消息（包括 info）

    OPTIX_CHECK(optixDeviceContextCreate(bridge->cuCtx, &optixOpts, &bridge->optixCtx));

    // --- 构建管线 ---

    // 管线编译选项
    OptixPipelineCompileOptions pipelineCompileOpts = {};
    pipelineCompileOpts.usesMotionBlur = false;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                                               | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOpts.numPayloadValues = 8; // 匹配 Payload 结构体的寄存器数量
    pipelineCompileOpts.numAttributeValues = 2; // 重心坐标
    pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "launch_params";

    // 从嵌入的 PTX 创建模块
    OptixModuleCompileOptions moduleCompileOpts = {};
    moduleCompileOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    char log[2048];
    size_t logSize = sizeof(log);

    // 创建 3 个独立的 OptiX 模块（每个 PTX 一个），避免
    // 重复符号定义（每个 PTX 是独立的编译单元）。
    auto createModule = [&](const char* ptx, size_t ptxLen, const char* name,
                             OptixModule* outModule) -> bool {
        logSize = sizeof(log);
        OptixResult res = optixModuleCreate(
            bridge->optixCtx, &moduleCompileOpts, &pipelineCompileOpts,
            ptx, ptxLen, log, &logSize, outModule);
        if (res != OPTIX_SUCCESS) {
            setError(bridge, "optixModuleCreate(%s) failed: %s", name, log);
            return false;
        }
        return true;
    };

    if (!createModule(ptx_raygen, strlen(ptx_raygen), "raygen", &bridge->moduleRaygen)) {
        optix_bridge_destroy(bridge);
        return NULL;
    }
    if (!createModule(ptx_closesthit, strlen(ptx_closesthit), "closesthit", &bridge->moduleClosesthit)) {
        optix_bridge_destroy(bridge);
        return NULL;
    }
    if (!createModule(ptx_miss, strlen(ptx_miss), "miss", &bridge->moduleMiss)) {
        optix_bridge_destroy(bridge);
        return NULL;
    }

    // 创建着色器程序组
    {
        OptixProgramGroupOptions pgOpts = {};

        OptixProgramGroupDesc raygenDesc = {};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = bridge->moduleRaygen;
        raygenDesc.raygen.entryFunctionName = "__raygen__rg";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            bridge->optixCtx, &raygenDesc, 1, &pgOpts,
            log, &logSize, &bridge->raygenPG
        ));

        OptixProgramGroupDesc missDesc = {};
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.miss.module = bridge->moduleMiss;
        missDesc.miss.entryFunctionName = "__miss__ms";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            bridge->optixCtx, &missDesc, 1, &pgOpts,
            log, &logSize, &bridge->missPG
        ));

        OptixProgramGroupDesc hitDesc = {};
        hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc.hitgroup.moduleCH = bridge->moduleClosesthit;
        hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hitDesc.hitgroup.moduleAH = NULL;
        hitDesc.hitgroup.entryFunctionNameAH = NULL;

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            bridge->optixCtx, &hitDesc, 1, &pgOpts,
            log, &logSize, &bridge->hitgroupPG
        ));
    }

    // 链接管线
    OptixProgramGroup pgList[] = {
        bridge->raygenPG,
        bridge->missPG,
        bridge->hitgroupPG
    };

    OptixPipelineLinkOptions pipelineLinkOpts = {};
    pipelineLinkOpts.maxTraceDepth = 31;
    pipelineLinkOpts.maxTraversableGraphDepth = 1;

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

    // 栈大小：使用 OptiX 内部默认值（更简单且始终正确）
    // optixProgramGroupGetStackSize 已从 OptiX 9.x 函数表中移除

    // 构建 SBT 记录
    OPTIX_CHECK(optixSbtRecordPackHeader(bridge->raygenPG, &bridge->sbtRaygen));
    OPTIX_CHECK(optixSbtRecordPackHeader(bridge->missPG, &bridge->sbtMiss));
    OPTIX_CHECK(optixSbtRecordPackHeader(bridge->hitgroupPG, &bridge->sbtHitgroup));

    // 将 SBT 记录上传到设备
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
    if (bridge->d_denoisedOutput) cuMemFree(bridge->d_denoisedOutput);
    if (bridge->d_albedoBuffer)      cuMemFree(bridge->d_albedoBuffer);
    if (bridge->d_guideNormalBuffer) cuMemFree(bridge->d_guideNormalBuffer);
    if (bridge->d_denoiserScratch) cuMemFree(bridge->d_denoiserScratch);
    if (bridge->d_denoiserState) cuMemFree(bridge->d_denoiserState);
    if (bridge->denoiser)       optixDenoiserDestroy(bridge->denoiser);
    if (bridge->d_triMaterial) cuMemFree(bridge->d_triMaterial);
    if (bridge->d_materials)   cuMemFree(bridge->d_materials);
    if (bridge->d_vertexBuffer) cuMemFree(bridge->d_vertexBuffer);
    if (bridge->d_normalBuffer) cuMemFree(bridge->d_normalBuffer);
    if (bridge->d_indexBuffer)  cuMemFree(bridge->d_indexBuffer);
    if (bridge->d_gasBuffer)    cuMemFree(bridge->d_gasBuffer);
    if (bridge->d_sbtRaygen)    cuMemFree(bridge->d_sbtRaygen);
    if (bridge->d_sbtMiss)      cuMemFree(bridge->d_sbtMiss);
    if (bridge->d_sbtHitgroup)  cuMemFree(bridge->d_sbtHitgroup);

    if (bridge->pipeline)       optixPipelineDestroy(bridge->pipeline);
    if (bridge->raygenPG)       optixProgramGroupDestroy(bridge->raygenPG);
    if (bridge->missPG)         optixProgramGroupDestroy(bridge->missPG);
    if (bridge->hitgroupPG)     optixProgramGroupDestroy(bridge->hitgroupPG);
    if (bridge->moduleRaygen)    optixModuleDestroy(bridge->moduleRaygen);
    if (bridge->moduleClosesthit) optixModuleDestroy(bridge->moduleClosesthit);
    if (bridge->moduleMiss)      optixModuleDestroy(bridge->moduleMiss);
    if (bridge->optixCtx)       optixDeviceContextDestroy(bridge->optixCtx);

    if (bridge->stream)         cuStreamDestroy(bridge->stream);
    if (bridge->cuCtx)          cuCtxDestroy(bridge->cuCtx);

    free(bridge);
}

bool optix_bridge_build_accel(
    OptiXBridge* bridge,
    const float* vertices,
    const unsigned int* indices,
    const float* normals,
    int tri_count,
    int vertex_count)
{
    if (!bridge) return false;

    const size_t vertexSize = (size_t)vertex_count * 3 * sizeof(float);
    const size_t indexSize  = (size_t)tri_count * 3 * sizeof(unsigned int);
    const size_t normalSize = (size_t)vertex_count * 3 * sizeof(float);

    // 重建时释放旧缓冲区
    if (bridge->d_vertexBuffer) { CUDA_CHECK_FREE(cuMemFree(bridge->d_vertexBuffer)); bridge->d_vertexBuffer = 0; }
    if (bridge->d_normalBuffer) { CUDA_CHECK_FREE(cuMemFree(bridge->d_normalBuffer)); bridge->d_normalBuffer = 0; }
    if (bridge->d_indexBuffer)  { CUDA_CHECK_FREE(cuMemFree(bridge->d_indexBuffer));  bridge->d_indexBuffer = 0; }
    if (bridge->d_gasBuffer)    { CUDA_CHECK_FREE(cuMemFree(bridge->d_gasBuffer));    bridge->d_gasBuffer = 0; }

    // 上传顶点、法线和索引数据
    CUDA_CHECK(cuMemAlloc(&bridge->d_vertexBuffer, vertexSize));
    CUDA_CHECK(cuMemAlloc(&bridge->d_normalBuffer, normalSize));
    CUDA_CHECK(cuMemAlloc(&bridge->d_indexBuffer, indexSize));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_vertexBuffer, vertices, vertexSize));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_normalBuffer, normals, normalSize));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_indexBuffer, indices, indexSize));

    // 构建加速结构（RT Core 硬件 BVH）
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.numVertices = (unsigned int)vertex_count;
    buildInput.triangleArray.vertexBuffers = &bridge->d_vertexBuffer;
    buildInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.numIndexTriplets = (unsigned int)tri_count;
    buildInput.triangleArray.indexBuffer = bridge->d_indexBuffer;
    unsigned int triangleFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags = triangleFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOpts = {};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                         | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                         | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        bridge->optixCtx,
        &accelOpts,
        &buildInput,
        1, // 输入数量
        &bufferSizes
    ));

    // 临时缓冲区
    CUdeviceptr d_temp;
    CUDA_CHECK(cuMemAlloc(&d_temp, bufferSizes.tempSizeInBytes));

    // 输出缓冲区
    CUdeviceptr d_outputAccel;
    CUDA_CHECK(cuMemAlloc(&d_outputAccel, bufferSizes.outputSizeInBytes));

    // 压缩后的结构大小 (uint64)
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

    // 压缩加速结构
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

    // 释放临时缓冲区
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

    const size_t outputSize = (size_t)width * (size_t)height * 3 * sizeof(float);

    // 分配输出缓冲区
    if (bridge->d_output) cuMemFree(bridge->d_output);
    CUDA_CHECK(cuMemAlloc(&bridge->d_output, outputSize));

    // 分配启动参数缓冲区
    if (bridge->d_launchParams) cuMemFree(bridge->d_launchParams);
    CUDA_CHECK(cuMemAlloc(&bridge->d_launchParams, sizeof(GpuLaunchParams)));

    // 分配降噪器引导缓冲区（albedo + 世界空间法线，每像素 float3）
    if (bridge->d_albedoBuffer) cuMemFree(bridge->d_albedoBuffer);
    if (bridge->d_guideNormalBuffer) cuMemFree(bridge->d_guideNormalBuffer);
    CUDA_CHECK(cuMemAlloc(&bridge->d_albedoBuffer, outputSize));
    // 回滚：如果法线缓冲区分配失败，释放 albedo 缓冲区
    {
        CUresult r = cuMemAlloc(&bridge->d_guideNormalBuffer, outputSize);
        if (r != CUDA_SUCCESS) {
            cuMemFree(bridge->d_albedoBuffer);
            bridge->d_albedoBuffer = 0;
            const char* nm;
            cuGetErrorName(r, &nm);
            setError(bridge, "CUDA error %s at %s:%d (guideNormalBuffer)", nm, __FILE__, __LINE__);
            return false;
        }
    }
    CUDA_CHECK(cuMemsetD8(bridge->d_albedoBuffer, 0, outputSize));
    CUDA_CHECK(cuMemsetD8(bridge->d_guideNormalBuffer, 0, outputSize));

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

    // 准备启动参数
    GpuLaunchParams params = {};
    params.width  = (unsigned int)bridge->width;
    params.height = (unsigned int)bridge->height;
    params.seed   = seed;
    params.sqrt_spp = bridge->sqrtSpp;
    params.max_depth = bridge->maxDepth;
    params.pixel_samples_scale = bridge->pixelScale;
    params.background = { 0.0f, 0.0f, 0.0f };
    params.framebuffer = (GpuFloat3*)bridge->d_output;
    params.materials = (GpuMaterial*)bridge->d_materials;
    params.material_count = bridge->materialCount;
    params.vertex_buffer = (GpuFloat3*)bridge->d_vertexBuffer;
    params.normal_buffer = (GpuFloat3*)bridge->d_normalBuffer;
    params.index_buffer = (unsigned int*)bridge->d_indexBuffer;
    params.tri_material = (unsigned int*)bridge->d_triMaterial;
    params.traversable = bridge->gasHandle;
    params.light_corner = bridge->lightCorner;
    params.light_u = bridge->lightU;
    params.light_v = bridge->lightV;
    params.light_area_inv = bridge->lightAreaInv;
    params.sphere_center = bridge->sphereCenter;
    params.sphere_radius = bridge->sphereRadius;
    params.albedo_buffer = (GpuFloat3*)bridge->d_albedoBuffer;
    params.guide_normal_buffer = (GpuFloat3*)bridge->d_guideNormalBuffer;
    fillGpuCamera(camera, &params.camera);

    CUDA_CHECK(cuMemcpyHtoD(bridge->d_launchParams, &params, sizeof(GpuLaunchParams)));

    // 设置 SBT
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord                = bridge->d_sbtRaygen;
    sbt.missRecordBase              = bridge->d_sbtMiss;
    sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = bridge->d_sbtHitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupSbtRecord);
    sbt.hitgroupRecordCount         = 1;

    // 发射光线
    OPTIX_CHECK(optixLaunch(
        bridge->pipeline,
        bridge->stream,
        bridge->d_launchParams,
        sizeof(GpuLaunchParams),
        &sbt,
        (unsigned int)bridge->width,
        (unsigned int)bridge->height,
        1 // depth = 1（仅 raygen 发射光线）
    ));

    CUDA_CHECK(cuStreamSynchronize(bridge->stream));

    // 下载渲染结果到主机
    const size_t outputSize = (size_t)bridge->width * (size_t)bridge->height * 3 * sizeof(float);
    CUDA_CHECK(cuMemcpyDtoH(output, bridge->d_output, outputSize));

    return true;
}

bool optix_bridge_set_tri_material(OptiXBridge* bridge, const unsigned int* tri_material, int tri_count) {
    if (!bridge || !tri_material || tri_count <= 0) return false;

    if (bridge->d_triMaterial) {
        CUDA_CHECK_FREE(cuMemFree(bridge->d_triMaterial));
        bridge->d_triMaterial = 0;
    }

    const size_t size = tri_count * sizeof(unsigned int);
    CUDA_CHECK(cuMemAlloc(&bridge->d_triMaterial, size));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_triMaterial, tri_material, size));

    fprintf(stderr, "[OptiXBridge] Uploaded tri_material: %d entries (%zu bytes)\n", tri_count, size);
    return true;
}

bool optix_bridge_set_materials(OptiXBridge* bridge, const void* materials, unsigned int count) {
    if (!bridge || !materials || count == 0) return false;

    // 如果存在旧缓冲区则先释放
    if (bridge->d_materials) {
        CUDA_CHECK_FREE(cuMemFree(bridge->d_materials));
        bridge->d_materials = 0;
    }

    const size_t size = count * sizeof(GpuMaterial);
    CUDA_CHECK(cuMemAlloc(&bridge->d_materials, size));
    CUDA_CHECK(cuMemcpyHtoD(bridge->d_materials, materials, size));
    bridge->materialCount = count;

    fprintf(stderr, "[OptiXBridge] Uploaded %u materials (%zu bytes)\n", count, size);
    return true;
}

bool optix_bridge_set_render_params(OptiXBridge* bridge, unsigned int sqrt_spp, unsigned int max_depth, float pixel_samples_scale) {
    if (!bridge) return false;
    bridge->sqrtSpp = sqrt_spp;
    bridge->maxDepth = max_depth;
    bridge->pixelScale = pixel_samples_scale;
    return true;
}

bool optix_bridge_set_light(
    OptiXBridge* bridge,
    const float* corner,
    const float* u,
    const float* v,
    float area_inv)
{
    if (!bridge) return false;
    bridge->lightCorner = { corner[0], corner[1], corner[2] };
    bridge->lightU      = { u[0], u[1], u[2] };
    bridge->lightV      = { v[0], v[1], v[2] };
    bridge->lightAreaInv = area_inv;
    fprintf(stderr, "[OptiXBridge] Light set: corner=(%.1f,%.1f,%.1f) u=(%.1f,%.1f,%.1f) v=(%.1f,%.1f,%.1f) area_inv=%.6f\n",
            corner[0], corner[1], corner[2], u[0], u[1], u[2], v[0], v[1], v[2], area_inv);
    return true;
}

bool optix_bridge_set_sphere(
    OptiXBridge* bridge,
    const float* center,
    float radius)
{
    if (!bridge) return false;
    bridge->sphereCenter = { center[0], center[1], center[2] };
    bridge->sphereRadius = radius;
    fprintf(stderr, "[OptiXBridge] Sphere set: center=(%.1f,%.1f,%.1f) radius=%.1f\n",
            center[0], center[1], center[2], radius);
    return true;
}

bool optix_bridge_denoise(OptiXBridge* bridge, float* output) {
    if (!bridge || !bridge->d_output || bridge->width == 0) return false;

    int width = bridge->width;
    int height = bridge->height;

    // 分辨率变更：销毁旧降噪器并重新创建为正确尺寸
    if (bridge->denoiser && (bridge->denoiserWidth != width || bridge->denoiserHeight != height)) {
        fprintf(stderr, "[OptiXBridge] Resolution changed %dx%d -> %dx%d, recreating denoiser\n",
                bridge->denoiserWidth, bridge->denoiserHeight, width, height);
        if (bridge->d_denoisedOutput)   { cuMemFree(bridge->d_denoisedOutput);   bridge->d_denoisedOutput = 0; }
        if (bridge->d_denoiserScratch)  { cuMemFree(bridge->d_denoiserScratch);  bridge->d_denoiserScratch = 0; }
        if (bridge->d_denoiserState)    { cuMemFree(bridge->d_denoiserState);    bridge->d_denoiserState = 0; }
        if (bridge->denoiser)           { optixDenoiserDestroy(bridge->denoiser); bridge->denoiser = 0; }
        bridge->denoiserSetup = false;
        bridge->denoiserStateSize = 0;
        bridge->denoiserScratchSize = 0;
    }

    // 一次性创建（HDR 模型 + albedo/normal 引导缓冲区）
    if (!bridge->denoiser) {
        OptixDenoiserOptions opts = {};
        opts.guideAlbedo = 1;
        opts.guideNormal = 1;

        OPTIX_CHECK(optixDenoiserCreate(
            bridge->optixCtx,
            OPTIX_DENOISER_MODEL_KIND_HDR,
            &opts,
            &bridge->denoiser
        ));

        OptixDenoiserSizes sizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            bridge->denoiser,
            (unsigned int)width,
            (unsigned int)height,
            &sizes
        ));

        bridge->denoiserStateSize = sizes.stateSizeInBytes;
        bridge->denoiserScratchSize = sizes.withoutOverlapScratchSizeInBytes;

        CUDA_CHECK(cuMemAlloc(&bridge->d_denoiserState, bridge->denoiserStateSize));
        CUDA_CHECK(cuMemAlloc(&bridge->d_denoiserScratch, bridge->denoiserScratchSize));

        OPTIX_CHECK(optixDenoiserSetup(
            bridge->denoiser,
            bridge->stream,
            (unsigned int)width,
            (unsigned int)height,
            bridge->d_denoiserState,
            bridge->denoiserStateSize,
            bridge->d_denoiserScratch,
            bridge->denoiserScratchSize
        ));

        size_t outputSize = (size_t)width * (size_t)height * 3 * sizeof(float);
        CUDA_CHECK(cuMemAlloc(&bridge->d_denoisedOutput, outputSize));

        bridge->denoiserWidth = width;
        bridge->denoiserHeight = height;
        bridge->denoiserSetup = true;
        fprintf(stderr, "[OptiXBridge] Denoiser created (HDR model + albedo/normal guides, Tensor Core)\n");
    }

    unsigned int rowStride = (unsigned int)((size_t)width * 3 * sizeof(float));

    // 输入颜色层
    OptixImage2D inputImage = {};
    inputImage.data = bridge->d_output;
    inputImage.width = (unsigned int)width;
    inputImage.height = (unsigned int)height;
    inputImage.rowStrideInBytes = rowStride;
    inputImage.pixelStrideInBytes = 3 * sizeof(float);
    inputImage.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // 输出图像
    OptixImage2D outputImage = {};
    outputImage.data = bridge->d_denoisedOutput;
    outputImage.width = (unsigned int)width;
    outputImage.height = (unsigned int)height;
    outputImage.rowStrideInBytes = rowStride;
    outputImage.pixelStrideInBytes = 3 * sizeof(float);
    outputImage.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // 引导层 albedo（首次命中表面反射率）
    OptixImage2D albedoImage = {};
    albedoImage.data = bridge->d_albedoBuffer;
    albedoImage.width = (unsigned int)width;
    albedoImage.height = (unsigned int)height;
    albedoImage.rowStrideInBytes = rowStride;
    albedoImage.pixelStrideInBytes = 3 * sizeof(float);
    albedoImage.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // 引导层 normal（首次命中世界空间法线）
    OptixImage2D normalImage = {};
    normalImage.data = bridge->d_guideNormalBuffer;
    normalImage.width = (unsigned int)width;
    normalImage.height = (unsigned int)height;
    normalImage.rowStrideInBytes = rowStride;
    normalImage.pixelStrideInBytes = 3 * sizeof(float);
    normalImage.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixDenoiserGuideLayer guideLayer = {};
    guideLayer.albedo = albedoImage;
    guideLayer.normal = normalImage;

    OptixDenoiserLayer inputLayer = {};
    inputLayer.input = inputImage;
    inputLayer.output = outputImage;

    OptixDenoiserParams params = {};
    // hdrIntensity 默认为 0.0（自动计算）—— 对 HDR 场景至关重要

    OPTIX_CHECK(optixDenoiserInvoke(
        bridge->denoiser,
        bridge->stream,
        &params,
        bridge->d_denoiserState,
        bridge->denoiserStateSize,
        &guideLayer,
        &inputLayer,
        1,
        0, 0,
        bridge->d_denoiserScratch,
        bridge->denoiserScratchSize
    ));

    CUDA_CHECK(cuStreamSynchronize(bridge->stream));

    // 将降噪结果复制回 d_output（设备端到设备端）
    size_t outputSize = (size_t)width * (size_t)height * 3 * sizeof(float);
    CUDA_CHECK(cuMemcpyDtoD(
        bridge->d_output,
        bridge->d_denoisedOutput,
        outputSize
    ));

    // 下载到主机端，调用者可获取降噪后的数据
    CUDA_CHECK(cuMemcpyDtoH(output, bridge->d_output, outputSize));

    fprintf(stderr, "[OptiXBridge] Denoised %dx%d image (Tensor Core HDR with guides)\n", width, height);
    return true;
}

const char* optix_bridge_get_device_name(const OptiXBridge* bridge) {
    if (!bridge) return "";
    return bridge->deviceName;
}

const char* optix_bridge_get_error(const OptiXBridge* bridge) {
    if (!bridge) return "Null bridge pointer";
    return bridge->errorMsg;
}
