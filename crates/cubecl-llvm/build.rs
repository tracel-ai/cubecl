use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=CUBECL_DEBUG_PLIRON");
    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }

    #[cfg(feature = "amdgpu")]
    {
        println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/lld.cpp");
        println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/device_libs.cpp");
        println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/printf.cpp");
        let prefix = tracel_llvm_bundler::config::llvm_path()?.into_os_string();
        let mut shim = cc::Build::new();
        shim.cpp(true)
            .file("src/amdgpu/cpp_shims/lld.cpp")
            .file("src/amdgpu/cpp_shims/device_libs.cpp")
            .file("src/amdgpu/cpp_shims/printf.cpp");

        shim.flags(tracel_llvm_bundler::config::get_cxxflags_args(Some(
            &prefix,
        ))?);

        // The LLVM headers have multiple warning under `cc`'s default `-Wall -Wextra`
        shim.warnings(false);
        shim.opt_level(3);
        shim.compile("cubecl_llvm_shim");

        println!("cargo:rustc-link-lib=static=lldELF");
        println!("cargo:rustc-link-lib=static=lldCommon");
    }

    tracel_llvm_bundler::llvm_sys::link()?;

    Ok(())
}
