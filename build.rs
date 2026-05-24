//! Build script for CUDA/OptiX GPU rendering support.
//!
//! When the `cuda` feature is enabled:
//! 1. Locates CUDA Toolkit and OptiX SDK
//! 2. Compiles .cu shader files to .ptx using NVCC
//! 3. Compiles optix_bridge.cu to a static library
//!
//! Without `cuda` feature: no-op.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_cuda_toolkit() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_PATH") {
        let p = PathBuf::from(&path);
        if p.join("bin").join("nvcc.exe").exists() {
            return Some(p);
        }
    }

    let cuda_base = Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
    if cuda_base.exists() {
        let mut versions: Vec<_> = std::fs::read_dir(cuda_base)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        versions.sort_by_key(|e| e.file_name());
        versions.reverse();

        for entry in versions {
            let nvcc = entry.path().join("bin").join("nvcc.exe");
            if nvcc.exists() {
                return Some(entry.path());
            }
        }
    }
    None
}

fn find_optix_sdk() -> Option<PathBuf> {
    if let Ok(path) = env::var("OPTIX_PATH") {
        let p = PathBuf::from(&path);
        if p.join("include").join("optix.h").exists() {
            return Some(p);
        }
    }

    let candidates = [
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0",
        "C:/Program Files/NVIDIA Corporation/OptiX SDK 8.1",
        "C:/Program Files/NVIDIA Corporation/OptiX SDK 8.0",
    ];

    for c in &candidates {
        let p = Path::new(c);
        if p.join("include").join("optix.h").exists() {
            return Some(p.to_path_buf());
        }
    }

    let base = Path::new("C:/ProgramData/NVIDIA Corporation");
    if base.exists() {
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.filter_map(|e| e.ok()) {
                let name_str = entry.file_name().to_string_lossy().to_lowercase();
                if name_str.contains("optix") {
                    let p = entry.path();
                    if p.join("include").join("optix.h").exists() {
                        return Some(p);
                    }
                }
            }
        }
    }

    None
}

fn find_msvc_lib() -> Option<String> {
    let candidates = [
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC",
    ];

    for base in &candidates {
        let p = Path::new(base);
        if p.exists() {
            if let Ok(entries) = std::fs::read_dir(p) {
                let mut versions: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .collect();
                versions.sort_by_key(|e| e.file_name());
                versions.reverse();

                for v in versions {
                    let lib = v.path()
                        .join("bin")
                        .join("Hostx64")
                        .join("x64")
                        .join("lib.exe");
                    if lib.exists() {
                        return Some(lib.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    // Try vswhere
    if let Ok(output) = Command::new("vswhere")
        .args(["-latest", "-products", "*",
               "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
               "-property", "installationPath"])
        .output()
    {
        let install_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !install_path.is_empty() {
            let vc_path = Path::new(&install_path).join("VC").join("Tools").join("MSVC");
            if vc_path.exists() {
                if let Ok(entries) = std::fs::read_dir(&vc_path) {
                    let mut versions: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                        .collect();
                    versions.sort_by_key(|e| e.file_name());
                    versions.reverse();

                    for v in versions {
                        let lib = v.path()
                            .join("bin")
                            .join("Hostx64")
                            .join("x64")
                            .join("lib.exe");
                        if lib.exists() {
                            return Some(lib.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

fn main() {
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    if !cuda_enabled {
        eprintln!("[build.rs] CUDA feature not enabled, skipping GPU build");
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/optix_bridge.cu");
    println!("cargo:rerun-if-changed=src/cuda/optix_bridge.h");
    println!("cargo:rerun-if-changed=src/cuda/shaders/");

    let cuda_path = find_cuda_toolkit().unwrap_or_else(|| {
        eprintln!("ERROR: CUDA Toolkit not found.");
        eprintln!("  Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads");
        eprintln!("  Or set CUDA_PATH environment variable to your CUDA installation.");
        std::process::exit(1);
    });

    let optix_path = find_optix_sdk().unwrap_or_else(|| {
        eprintln!("ERROR: OptiX SDK not found.");
        eprintln!("  Install OptiX SDK 8.x from https://developer.nvidia.com/optix/download");
        eprintln!("  Or set OPTIX_PATH environment variable to your OptiX SDK installation.");
        eprintln!("  Expected directory containing include/optix.h");
        std::process::exit(1);
    });

    let nvcc = cuda_path.join("bin").join("nvcc.exe");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("src").join("cuda").join("shaders");
    let optix_include = optix_path.join("include");
    let cuda_include = cuda_path.join("include");

    eprintln!("[build.rs] NVCC:  {}", nvcc.display());
    eprintln!("[build.rs] CUDA:  {}", cuda_path.display());
    eprintln!("[build.rs] OptiX: {}", optix_path.display());

    // --- Compile shader .cu files to .ptx ---
    let shaders = ["raygen.cu", "closesthit.cu", "miss.cu"];

    for shader in &shaders {
        let input = shader_dir.join(shader);
        let output = out_dir.join(format!("{}.ptx", shader));

        eprintln!("[build.rs] Compiling {} -> {}", shader, output.display());

        let status = Command::new(&nvcc)
            .arg("-ptx")
            .arg("--use-fast-math")
            .arg("-lineinfo")
            .arg(format!("-I{}", optix_include.display()))
            .arg(format!("-I{}", shader_dir.display()))
            .arg("-o").arg(&output)
            .arg(&input)
            .status()
            .unwrap_or_else(|e| {
                panic!("NVCC not found at {}: {}\nInstall CUDA Toolkit 12.x.", nvcc.display(), e);
            });

        if !status.success() {
            panic!("NVCC failed compiling {}. Fix shader errors and retry.", shader);
        }
    }

    // --- Compile optix_bridge.cu to static library ---
    let bridge_cu = manifest_dir.join("src").join("cuda").join("optix_bridge.cu");
    let bridge_obj = out_dir.join("optix_bridge.obj");
    let bridge_lib = out_dir.join("optix_bridge.lib");

    eprintln!("[build.rs] Compiling optix_bridge.cu -> obj");

    let status = Command::new(&nvcc)
        .arg("-c")
        .arg("--use-fast-math")
        .arg("-lineinfo")
        .arg(format!("-I{}", optix_include.display()))
        .arg(format!("-I{}", cuda_include.display()))
        .arg("-o").arg(&bridge_obj)
        .arg(&bridge_cu)
        .status()
        .expect("NVCC failed for optix_bridge.cu");

    if !status.success() {
        panic!("NVCC failed compiling optix_bridge.cu. Fix errors and retry.");
    }

    // Create static library from .obj
    let lib_exe = find_msvc_lib().unwrap_or_else(|| "lib.exe".to_string());
    eprintln!("[build.rs] Creating static library with {}", lib_exe);

    let status = Command::new(&lib_exe)
        .arg(format!("/OUT:{}", bridge_lib.display()))
        .arg(&bridge_obj)
        .status()
        .unwrap_or_else(|_| {
            panic!("Failed to run {}. Install Visual Studio 2022 Build Tools.", lib_exe);
        });

    if !status.success() {
        panic!("lib.exe failed. Check Visual Studio installation.");
    }

    // --- Emit cargo link directives ---
    let lib_dir = out_dir.to_string_lossy().to_string();
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=static=optix_bridge");

    let optix_lib_dir = optix_path.join("lib");
    if optix_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", optix_lib_dir.display());
    }
    println!("cargo:rustc-link-lib=optix");

    let cuda_lib_dir = cuda_path.join("lib").join("x64");
    if cuda_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    }
    println!("cargo:rustc-link-lib=cudart");

    eprintln!("[build.rs] GPU build complete.");
}
