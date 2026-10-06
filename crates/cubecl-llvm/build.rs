fn main() -> Result<(), Box<dyn std::error::Error>> {
    // C++ shims for the LLVM APIs that the C API does not bind, and the features that need them.
    let shims: Vec<&str> = [
        (
            cfg!(any(feature = "amdgpu", feature = "nvptx")),
            &[
                "src/shared/cpp_shims/bitcode.cpp",
                "src/shared/cpp_shims/options.cpp",
            ][..],
        ),
        (
            cfg!(feature = "amdgpu"),
            &[
                "src/amdgpu/cpp_shims/lld.cpp",
                "src/amdgpu/cpp_shims/printf.cpp",
            ],
        ),
        (
            cfg!(feature = "jitdump"),
            &["src/cpu/cpp_shims/perf_support.cpp"],
        ),
    ]
    .into_iter()
    .filter(|(enabled, _)| *enabled)
    .flat_map(|(_, files)| files.iter().copied())
    .collect();

    if !shims.is_empty() {
        let prefix = tracel_llvm_bundler::config::llvm_path()?.into_os_string();
        let mut shim = cc::Build::new();
        shim.cpp(true);
        for file in &shims {
            println!("cargo::rerun-if-changed={file}");
            shim.file(file);
        }

        shim.flags(tracel_llvm_bundler::config::get_cxxflags_args(Some(
            &prefix,
        ))?);

        // The LLVM headers have multiple warning under `cc`'s default `-Wall -Wextra`
        shim.warnings(false);
        shim.opt_level(3);
        shim.compile("cubecl_llvm_shim");
    }

    #[cfg(feature = "amdgpu")]
    {
        println!("cargo:rustc-link-lib=static=lldELF");
        println!("cargo:rustc-link-lib=static=lldCommon");
    }

    tracel_llvm_bundler::llvm_sys::link()?;

    // JIT code keeps frame pointers when the host code does, so `perf --call-graph fp` can walk
    // through kernel frames.
    println!("cargo::rustc-check-cfg=cfg(cubecl_frame_pointers)");
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    if forces_frame_pointers(&std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default()) {
        println!("cargo:rustc-cfg=cubecl_frame_pointers");
    }

    Ok(())
}

include!("build/frame_pointers.rs");
