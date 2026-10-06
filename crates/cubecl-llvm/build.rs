fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "amdgpu", feature = "nvptx"))]
    {
        println!("cargo::rerun-if-changed=src/shared/cpp_shims/bitcode.cpp");
        println!("cargo::rerun-if-changed=src/shared/cpp_shims/options.cpp");
        let prefix = tracel_llvm_bundler::config::llvm_path()?.into_os_string();
        let mut shim = cc::Build::new();
        shim.cpp(true)
            .file("src/shared/cpp_shims/bitcode.cpp")
            .file("src/shared/cpp_shims/options.cpp");
        #[cfg(feature = "amdgpu")]
        {
            println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/lld.cpp");
            println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/printf.cpp");
            shim.file("src/amdgpu/cpp_shims/lld.cpp")
                .file("src/amdgpu/cpp_shims/printf.cpp");
        }

        shim.flags(tracel_llvm_bundler::config::get_cxxflags_args(Some(
            &prefix,
        ))?);

        // The LLVM headers have multiple warning under `cc`'s default `-Wall -Wextra`
        shim.warnings(false);
        shim.opt_level(3);
        shim.compile("cubecl_llvm_shim");

        #[cfg(feature = "amdgpu")]
        {
            println!("cargo:rustc-link-lib=static=lldELF");
            println!("cargo:rustc-link-lib=static=lldCommon");
        }
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
