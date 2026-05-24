//! CUDA/OptiX GPU rendering module.
//!
//! Provides hardware-accelerated path tracing using NVIDIA RT Cores (BVH traversal),
//! Tensor Cores (AI denoising), and CUDA cores (material shading / MIS).

#[cfg(feature = "cuda")]
pub mod optix;

// Future phases:
// #[cfg(feature = "cuda")]
// pub mod scene;
// #[cfg(feature = "cuda")]
// pub mod render;
// #[cfg(feature = "cuda")]
// pub mod denoise;
