#![allow(dead_code)]

//! FFI bindings for the OptiX C bridge library.
//!
//! All functions are unsafe wrappers around the C API defined in `optix_bridge.h`.
//! The bridge manages CUDA contexts, OptiX pipelines, acceleration structures,
//! and kernel launches internally.

use std::ffi::c_char;

/// Load embedded PTX shader code (compiled by build.rs from src/cuda/shaders/*.cu).
/// Each .cu file is compiled to .ptx in OUT_DIR during the cargo build.
fn ptx_str(name: &str) -> &'static str {
    match name {
        "raygen.cu" => include_str!(concat!(env!("OUT_DIR"), "/raygen.cu.ptx")),
        "closesthit.cu" => include_str!(concat!(env!("OUT_DIR"), "/closesthit.cu.ptx")),
        "miss.cu" => include_str!(concat!(env!("OUT_DIR"), "/miss.cu.ptx")),
        _ => panic!("Unknown shader: {}", name),
    }
}

/// Convenience: load all three shader PTX strings as a tuple.
pub fn load_ptx_shaders() -> (&'static str, &'static str, &'static str) {
    (ptx_str("raygen.cu"), ptx_str("closesthit.cu"), ptx_str("miss.cu"))
}

/// C-compatible camera parameters (matches `BridgeCameraParams` in optix_bridge.h).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BridgeCameraParams {
    pub lookfrom: [f32; 3],
    pub lookat: [f32; 3],
    pub vup: [f32; 3],
    pub vfov: f32,
    pub aspect_ratio: f32,
    pub defocus_angle: f32,
    pub focus_dist: f32,
    pub u: [f32; 3],
    pub v: [f32; 3],
    pub w: [f32; 3],
    pub pixel00_loc: [f32; 3],
    pub pixel_delta_u: [f32; 3],
    pub pixel_delta_v: [f32; 3],
    pub defocus_disk_u: [f32; 3],
    pub defocus_disk_v: [f32; 3],
}

/// Opaque handle to the OptiX bridge state.
pub struct OptiXBridge {
    _private: *mut std::ffi::c_void,
}

// FFI declarations
extern "C" {
    fn optix_bridge_init(
        ptx_raygen: *const c_char,
        ptx_closesthit: *const c_char,
        ptx_miss: *const c_char,
    ) -> *mut std::ffi::c_void;

    fn optix_bridge_destroy(bridge: *mut std::ffi::c_void);

    fn optix_bridge_build_accel(
        bridge: *mut std::ffi::c_void,
        vertices: *const f32,
        indices: *const u32,
        normals: *const f32,
        tri_count: i32,
        vertex_count: i32,
    ) -> bool;

    fn optix_bridge_create_pipeline(bridge: *mut std::ffi::c_void, width: i32, height: i32)
        -> bool;

    fn optix_bridge_set_tri_material(
        bridge: *mut std::ffi::c_void,
        tri_material: *const u32,
        tri_count: i32,
    ) -> bool;

    fn optix_bridge_set_materials(
        bridge: *mut std::ffi::c_void,
        materials: *const std::ffi::c_void,
        count: u32,
    ) -> bool;

    fn optix_bridge_set_render_params(
        bridge: *mut std::ffi::c_void,
        sqrt_spp: u32,
        max_depth: u32,
        pixel_samples_scale: f32,
    ) -> bool;

    fn optix_bridge_render(
        bridge: *mut std::ffi::c_void,
        output: *mut f32,
        camera: *const BridgeCameraParams,
        seed: u32,
    ) -> bool;

    fn optix_bridge_get_error(bridge: *const std::ffi::c_void) -> *const c_char;

    fn optix_bridge_get_device_name(bridge: *const std::ffi::c_void) -> *const c_char;

    fn optix_bridge_denoise(bridge: *mut std::ffi::c_void) -> bool;

    fn optix_bridge_set_sphere(
        bridge: *mut std::ffi::c_void,
        center: *const f32,
        radius: f32,
    ) -> bool;

    fn optix_bridge_set_light(
        bridge: *mut std::ffi::c_void,
        corner: *const f32,
        u: *const f32,
        v: *const f32,
        area_inv: f32,
    ) -> bool;
}

impl OptiXBridge {
    /// Initialize OptiX + CUDA. PTX strings are the compiled shader code.
    pub fn new(ptx_raygen: &str, ptx_closesthit: &str, ptx_miss: &str) -> Option<Self> {
        let ptx_r = std::ffi::CString::new(ptx_raygen).ok()?;
        let ptx_c = std::ffi::CString::new(ptx_closesthit).ok()?;
        let ptx_m = std::ffi::CString::new(ptx_miss).ok()?;

        let ptr = unsafe { optix_bridge_init(ptx_r.as_ptr(), ptx_c.as_ptr(), ptx_m.as_ptr()) };

        if ptr.is_null() {
            None
        } else {
            Some(OptiXBridge { _private: ptr })
        }
    }

    /// Build triangle acceleration structure (RT Core hardware BVH).
    pub fn build_accel(
        &mut self,
        vertices: &[f32],
        indices: &[u32],
        normals: &[f32],
        tri_count: i32,
        vertex_count: i32,
    ) -> bool {
        unsafe {
            optix_bridge_build_accel(
                self._private,
                vertices.as_ptr(),
                indices.as_ptr(),
                normals.as_ptr(),
                tri_count,
                vertex_count,
            )
        }
    }

    /// Create the OptiX pipeline (raygen + closesthit + miss).
    pub fn create_pipeline(&mut self, width: i32, height: i32) -> bool {
        unsafe { optix_bridge_create_pipeline(self._private, width, height) }
    }

    /// Upload per-triangle material index data.
    pub fn set_tri_material(&mut self, tri_material: &[u32]) -> bool {
        unsafe {
            optix_bridge_set_tri_material(
                self._private,
                tri_material.as_ptr(),
                tri_material.len() as i32,
            )
        }
    }

    /// Upload material data to GPU. `materials` must be a slice of
    /// `#[repr(C)]` structs matching the C-side GpuMaterial layout (36 bytes each).
    pub fn set_materials<T>(&mut self, materials: &[T]) -> bool {
        let byte_len = materials.len() * std::mem::size_of::<T>();
        if byte_len == 0 {
            return true;
        }
        unsafe {
            optix_bridge_set_materials(
                self._private,
                materials.as_ptr() as *const std::ffi::c_void,
                materials.len() as u32,
            )
        }
    }

    /// Set render parameters (spp, max depth, etc.).
    pub fn set_render_params(&mut self, sqrt_spp: u32, max_depth: u32, pixel_scale: f32) -> bool {
        unsafe { optix_bridge_set_render_params(self._private, sqrt_spp, max_depth, pixel_scale) }
    }

    /// Launch the render. Output buffer must be pre-allocated to width * height * 3 floats.
    pub fn render(&mut self, output: &mut [f32], camera: &BridgeCameraParams, seed: u32) -> bool {
        unsafe {
            optix_bridge_render(
                self._private,
                output.as_mut_ptr(),
                camera as *const BridgeCameraParams,
                seed,
            )
        }
    }

    /// Get last error message from the bridge.
    pub fn get_error(&self) -> String {
        unsafe {
            let ptr = optix_bridge_get_error(self._private);
            if ptr.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    /// Get the CUDA device name. Returns empty string if not initialized.
    pub fn get_device_name(&self) -> &str {
        unsafe {
            let ptr = optix_bridge_get_device_name(self._private);
            if ptr.is_null() {
                return "";
            }
            std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
        }
    }

    /// Set area light geometry for importance sampling.
    pub fn set_light(
        &mut self,
        corner: &[f32; 3],
        u: &[f32; 3],
        v: &[f32; 3],
        area_inv: f32,
    ) -> bool {
        unsafe {
            optix_bridge_set_light(self._private, corner.as_ptr(), u.as_ptr(), v.as_ptr(), area_inv)
        }
    }

    /// Set sphere geometry for MIS direction sampling (matching CPU lights list).
    pub fn set_sphere(&mut self, center: &[f32; 3], radius: f32) -> bool {
        unsafe { optix_bridge_set_sphere(self._private, center.as_ptr(), radius) }
    }

    /// Apply AI denoiser (Tensor Core accelerated) to the last rendered frame.
    pub fn denoise(&mut self) -> bool {
        unsafe { optix_bridge_denoise(self._private) }
    }
}

impl Drop for OptiXBridge {
    fn drop(&mut self) {
        unsafe { optix_bridge_destroy(self._private) }
    }
}

// Safety: OptiXBridge is not Send/Sync by default since it holds CUDA context.
// However, our usage pattern is single-threaded for GPU operations.
unsafe impl Send for OptiXBridge {}

// ============================================================================
// GPU Diagnostics (for --check-gpu)
// ============================================================================

// CUDA device attribute enum values (from cuda.h)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Probe CUDA driver directly (fast, no OptiX dependency).
/// Uses raw CUDA Driver API — symbols provided by cuda.lib (linked by build.rs).
fn cuda_driver_probe() -> serde_json::Value {
    extern "C" {
        fn cuInit(flags: u32) -> i32;
        fn cuDriverGetVersion(version: *mut i32) -> i32;
        fn cuDeviceGetCount(count: *mut i32) -> i32;
        fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
        fn cuDeviceGetName(name: *mut std::ffi::c_char, len: i32, dev: i32) -> i32;
        fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> i32;
        fn cuDeviceTotalMem_v2(bytes: *mut u64, dev: i32) -> i32;
    }

    unsafe {
        if cuInit(0) != 0 {
            return serde_json::json!({
                "available": false,
                "device_name": null,
                "driver_version": null,
                "compute_capability": null,
                "vram_mb": null,
                "error": "cuInit failed: CUDA driver not installed or too old"
            });
        }

        // Driver version (e.g. 12000 = R560.x)
        let mut driver_ver: i32 = 0;
        cuDriverGetVersion(&mut driver_ver);
        let driver_major = driver_ver / 1000;
        let driver_minor = (driver_ver % 1000) / 10;

        let mut count: i32 = 0;
        if cuDeviceGetCount(&mut count) != 0 {
            return serde_json::json!({
                "available": false,
                "device_name": null,
                "driver_version": format!("{}.{}", driver_major, driver_minor),
                "compute_capability": null,
                "vram_mb": null,
                "error": "cuDeviceGetCount failed"
            });
        }
        if count == 0 {
            return serde_json::json!({
                "available": false,
                "device_name": null,
                "driver_version": format!("{}.{}", driver_major, driver_minor),
                "compute_capability": null,
                "vram_mb": null,
                "error": "No CUDA-capable devices found (count=0)"
            });
        }

        let mut device: i32 = 0;
        if cuDeviceGet(&mut device, 0) != 0 {
            return serde_json::json!({
                "available": false,
                "device_name": null,
                "driver_version": format!("{}.{}", driver_major, driver_minor),
                "compute_capability": null,
                "vram_mb": null,
                "error": "cuDeviceGet failed"
            });
        }

        let mut name_buf = [0i8; 256];
        if cuDeviceGetName(name_buf.as_mut_ptr() as *mut std::ffi::c_char, 256, device) != 0 {
            return serde_json::json!({
                "available": true,
                "device_name": null,
                "driver_version": format!("{}.{}", driver_major, driver_minor),
                "compute_capability": null,
                "vram_mb": null,
                "error": "cuDeviceGetName failed"
            });
        }

        let name = std::ffi::CStr::from_ptr(name_buf.as_ptr()).to_string_lossy().into_owned();

        // Compute capability
        let mut cc_major: i32 = 0;
        let mut cc_minor: i32 = 0;
        let cc = if cuDeviceGetAttribute(
            &mut cc_major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        ) == 0
            && cuDeviceGetAttribute(
                &mut cc_minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            ) == 0
        {
            Some(format!("{}.{}", cc_major, cc_minor))
        } else {
            None
        };

        // VRAM (via cuDeviceTotalMem_v2 — returns bytes as u64)
        let mut total_mem: u64 = 0;
        let vram_mb = if cuDeviceTotalMem_v2(&mut total_mem, device) == 0 {
            Some(total_mem / (1024 * 1024))
        } else {
            None
        };

        // Check minimum requirements
        let mut warnings: Vec<&str> = Vec::new();
        if driver_ver < 12000 {
            warnings.push(
                "Driver too old: NVIDIA R560+ required for OptiX 9.x support. Please update your \
                 driver.",
            );
        }
        if let Some(ref cc_str) = cc {
            let parts: Vec<&str> = cc_str.split('.').collect();
            if let (Some(major_str), Some(minor_str)) = (parts.first(), parts.get(1)) {
                if let (Ok(major), Ok(minor)) = (major_str.parse::<i32>(), minor_str.parse::<i32>())
                {
                    let cc_num = major * 10 + minor;
                    if cc_num < 75 {
                        warnings.push(
                            "GPU compute capability below 7.5 (Turing). This build targets sm_75 \
                             — older GPUs may not run all shaders correctly.",
                        );
                    }
                }
            }
        }

        serde_json::json!({
            "available": true,
            "device_name": name,
            "driver_version": format!("{}.{}", driver_major, driver_minor),
            "compute_capability": cc,
            "vram_mb": vram_mb,
            "device_count": count,
            "warnings": if warnings.is_empty() { None } else { Some(warnings.iter().map(|s| s.to_string()).collect::<Vec<_>>()) },
            "error": null
        })
    }
}

/// Probe full OptiX bridge initialization with real shader PTX.
fn optix_bridge_probe(device_name: Option<&str>) -> serde_json::Value {
    let (ptx_r, ptx_c, ptx_m) = load_ptx_shaders();
    match OptiXBridge::new(ptx_r, ptx_c, ptx_m) {
        Some(bridge) => {
            let name = bridge.get_device_name();
            let name_str = if name.is_empty() {
                device_name.map(|s| s.to_string())
            } else {
                Some(name.to_string())
            };
            serde_json::json!({
                "available": true,
                "device_name": name_str,
                "error": null
            })
        }
        None => {
            serde_json::json!({
                "available": false,
                "device_name": device_name,
                "error": "optixModuleCreate failed. This usually means PTX was compiled for a GPU architecture not supported by your driver, or OptiX SDK/driver version mismatch. Try rebuilding with --features cuda after updating CUDA toolkit and OptiX SDK."
            })
        }
    }
}

/// Run GPU diagnostics: probe CUDA driver, get device info, test OptiX init.
/// Prints JSON to stdout. Works even when OptiX cannot initialize.
pub fn check_gpu_diagnostics() -> anyhow::Result<()> {
    let cuda = cuda_driver_probe();
    let cuda_available = cuda["available"].as_bool().unwrap_or(false);
    let cuda_device = cuda["device_name"].as_str();

    let optix = if cuda_available {
        optix_bridge_probe(cuda_device)
    } else {
        serde_json::json!({
            "available": false,
            "device_name": null,
            "error": "Skipped: CUDA driver not available"
        })
    };

    let optix_available = optix["available"].as_bool().unwrap_or(false);

    let status = if optix_available {
        "ok"
    } else if cuda_available {
        "no_optix"
    } else {
        "no_cuda_driver"
    };

    let output = serde_json::json!({
        "status": status,
        "cuda": cuda,
        "optix": optix,
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that BridgeCameraParams has the correct size/memory layout.
    #[test]
    fn test_camera_params_layout() {
        // 11 float3 arrays + 4 scalar floats = 11*12 + 4*4 = 148 bytes
        assert_eq!(std::mem::size_of::<BridgeCameraParams>(), 148);
    }
}
