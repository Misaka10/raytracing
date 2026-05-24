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
        tri_count: i32,
    ) -> bool;

    fn optix_bridge_create_pipeline(
        bridge: *mut std::ffi::c_void,
        width: i32,
        height: i32,
    ) -> bool;

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
}

impl OptiXBridge {
    /// Initialize OptiX + CUDA. PTX strings are the compiled shader code.
    pub fn new(ptx_raygen: &str, ptx_closesthit: &str, ptx_miss: &str) -> Option<Self> {
        let ptx_r = std::ffi::CString::new(ptx_raygen).ok()?;
        let ptx_c = std::ffi::CString::new(ptx_closesthit).ok()?;
        let ptx_m = std::ffi::CString::new(ptx_miss).ok()?;

        let ptr = unsafe {
            optix_bridge_init(
                ptx_r.as_ptr(),
                ptx_c.as_ptr(),
                ptx_m.as_ptr(),
            )
        };

        if ptr.is_null() {
            None
        } else {
            Some(OptiXBridge { _private: ptr })
        }
    }

    /// Build triangle acceleration structure (RT Core hardware BVH).
    pub fn build_accel(&mut self, vertices: &[f32], indices: &[u32], tri_count: i32) -> bool {
        unsafe {
            optix_bridge_build_accel(
                self._private,
                vertices.as_ptr(),
                indices.as_ptr(),
                tri_count,
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
            optix_bridge_set_tri_material(self._private, tri_material.as_ptr(), tri_material.len() as i32)
        }
    }

    /// Upload material data to GPU. `materials` must be a slice of
    /// `#[repr(C)]` structs matching the C-side GpuMaterial layout (36 bytes each).
    pub fn set_materials<T>(&mut self, materials: &[T]) -> bool {
        let byte_len = materials.len() * std::mem::size_of::<T>();
        if byte_len == 0 { return true; }
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
        unsafe {
            optix_bridge_set_render_params(self._private, sqrt_spp, max_depth, pixel_scale)
        }
    }

    /// Launch the render. Output buffer must be pre-allocated to width * height * 3 floats.
    pub fn render(
        &mut self,
        output: &mut [f32],
        camera: &BridgeCameraParams,
        seed: u32,
    ) -> bool {
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
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
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
