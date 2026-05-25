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
        "C:/optix-sdk-9.1.0",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0",
        "C:/Program Files/NVIDIA Corporation/OptiX SDK 9.1",
        "C:/Program Files/NVIDIA Corporation/OptiX SDK 8.1",
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

/// Scan a MSVC directory for the latest version, return the bin dir path.
fn find_latest_msvc_bin(msvc_dir: &Path, tool: &str) -> Option<PathBuf> {
    if !msvc_dir.exists() {
        return None;
    }
    let entries = std::fs::read_dir(msvc_dir).ok()?;
    let mut versions: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    versions.sort_by_key(|e| e.file_name());
    versions.reverse();
    for v in versions {
        let exe = v.path().join("bin").join("Hostx64").join("x64").join(tool);
        if exe.exists() {
            return Some(v.path().join("bin").join("Hostx64").join("x64"));
        }
    }
    None
}

/// Find the latest MSVC tool binary path.
fn find_msvc_tool(tool: &str) -> Option<PathBuf> {
    // Check VSINSTALLDIR env var first (set by VS Developer Command Prompt)
    if let Ok(vs_dir) = std::env::var("VSINSTALLDIR") {
        let msvc_dir = Path::new(&vs_dir).join("VC").join("Tools").join("MSVC");
        if let Some(bin) = find_latest_msvc_bin(&msvc_dir, tool) {
            return Some(bin);
        }
    }

    // Search VS 2022 Community/Professional/Enterprise/BuildTools
    let vs_base = Path::new("C:/Program Files/Microsoft Visual Studio/2022");
    let vs_base_x86 = Path::new("C:/Program Files (x86)/Microsoft Visual Studio/2022");

    for base in [vs_base, vs_base_x86] {
        if !base.exists() {
            continue;
        }
        for edition in &["Community", "Professional", "Enterprise", "BuildTools"] {
            let msvc_dir = base.join(edition).join("VC").join("Tools").join("MSVC");
            if let Some(bin) = find_latest_msvc_bin(&msvc_dir, tool) {
                return Some(bin);
            }
        }
    }

    // Try vswhere
    if let Ok(output) = Command::new("vswhere")
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
    {
        let install_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !install_path.is_empty() {
            let msvc_dir = Path::new(&install_path).join("VC").join("Tools").join("MSVC");
            if let Some(bin) = find_latest_msvc_bin(&msvc_dir, tool) {
                return Some(bin);
            }
        }
    }

    None
}

fn find_msvc_bin_dir() -> Option<PathBuf> {
    find_msvc_tool("cl.exe")
}

fn find_msvc_lib() -> Option<String> {
    find_msvc_tool("lib.exe").map(|p| p.join("lib.exe").to_string_lossy().to_string())
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

    // Find MSVC host compiler for NVCC
    let msvc_bin = find_msvc_bin_dir().unwrap_or_else(|| {
        eprintln!("ERROR: MSVC host compiler (cl.exe) not found.");
        eprintln!("  Install Visual Studio 2022 with 'Desktop development with C++' workload.");
        eprintln!("  Or set VSINSTALLDIR environment variable.");
        std::process::exit(1);
    });

    eprintln!("[build.rs] NVCC:  {}", nvcc.display());
    eprintln!("[build.rs] CUDA:  {}", cuda_path.display());
    eprintln!("[build.rs] OptiX: {}", optix_path.display());
    eprintln!("[build.rs] MSVC:  {}", msvc_bin.display());

    // --- Compile shader .cu files to .ptx (parallel) ---
    let shaders = ["raygen.cu", "closesthit.cu", "miss.cu"];

    std::thread::scope(|s| {
        let handles: Vec<_> = shaders
            .iter()
            .map(|shader| {
                let input = shader_dir.join(shader);
                let output = out_dir.join(format!("{}.ptx", shader));
                let nvcc = &nvcc;
                let msvc_bin = &msvc_bin;
                let optix_include = &optix_include;
                let shader_dir = &shader_dir;
                s.spawn(move || {
                    eprintln!("[build.rs] Compiling {} -> {}", shader, output.display());

                    let status = Command::new(&nvcc)
                        .arg("-ptx")
                        .arg("--use_fast_math")
                        .arg("-O3")
                        .arg("-lineinfo")
                        .arg("--extra-device-vectorization")
                        .arg("--gpu-architecture=compute_75")
                        .arg("-Xcompiler")
                        .arg("/MT")
                        .arg(format!("-ccbin={}", msvc_bin.display()))
                        .arg(format!("-I{}", optix_include.display()))
                        .arg(format!("-I{}", shader_dir.display()))
                        .arg("-o")
                        .arg(&output)
                        .arg(&input)
                        .status()
                        .unwrap_or_else(|e| {
                            panic!(
                                "NVCC not found at {}: {}\nInstall CUDA Toolkit 12.x.",
                                nvcc.display(),
                                e
                            );
                        });

                    if !status.success() {
                        panic!("NVCC failed compiling {}. Fix shader errors and retry.", shader);
                    }

                    // CUDA 13.x generates PTX ISA 9.1 which OptiX 9.1 SDK cannot parse.
                    // Patch the .version directive down to 8.5 — the actual instructions
                    // are compute_75-compatible and valid in PTX ISA 8.x.
                    let ptx_content = std::fs::read_to_string(&output).unwrap_or_else(|e| {
                        panic!("Failed to read PTX {}: {}", output.display(), e)
                    });
                    let patched = ptx_content.replace(".version 9.1", ".version 8.5");
                    std::fs::write(&output, patched).unwrap_or_else(|e| {
                        panic!("Failed to write patched PTX {}: {}", output.display(), e)
                    });
                    eprintln!(
                        "[build.rs]   Patched PTX version 9.1 -> 8.5 for OptiX 9.1 compatibility"
                    );
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    // --- Compile optix_bridge.cu to static library ---
    let bridge_cu = manifest_dir.join("src").join("cuda").join("optix_bridge.cu");
    let bridge_obj = out_dir.join("optix_bridge.obj");
    let bridge_lib = out_dir.join("optix_bridge.lib");

    eprintln!("[build.rs] Compiling optix_bridge.cu -> obj");

    let status = Command::new(&nvcc)
        .arg("-c")
        .arg("--use_fast_math")
        .arg("-O3")
        .arg("-lineinfo")
        .arg("--gpu-architecture=compute_120")
        .arg("-Xcompiler")
        .arg("/MT")
        .arg(format!("-ccbin={}", msvc_bin.display()))
        .arg(format!("-I{}", optix_include.display()))
        .arg(format!("-I{}", cuda_include.display()))
        .arg("-o")
        .arg(&bridge_obj)
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
    // Note: OptiX 9.x runtime is part of the NVIDIA driver — no separate optix.lib needed.
    // The function table is populated by optixInit() at runtime via optix_stubs.h.
    let lib_dir = out_dir.to_string_lossy().to_string();
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=static=optix_bridge");

    let cuda_lib_dir = cuda_path.join("lib").join("x64");
    if cuda_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=cuda"); // CUDA driver API (cuInit, cuCtxCreate, etc.) — from nvcuda.dll in driver

    eprintln!("[build.rs] GPU build complete. Bridge lib + shader PTX ready.");
}
