use std::env;
use std::path::PathBuf;

include!("build_support.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=CUBECL_DEBUG_PLIRON");
    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }

    println!("cargo::rerun-if-changed=build_support.rs");
    println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/lld.cpp");
    println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/device_libs.cpp");
    println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/printf.cpp");
    let prefix: PathBuf = tracel_llvm_bundler::config::llvm_path()?;
    let mut shim = cc::Build::new();
    shim.cpp(true)
        .file("src/amdgpu/cpp_shims/lld.cpp")
        .file("src/amdgpu/cpp_shims/device_libs.cpp")
        .file("src/amdgpu/cpp_shims/printf.cpp");

    let cxxflags =
        tracel_llvm_bundler::config::get_cxxflags(Some(&prefix.clone().into_os_string()))?;
    for flag in shim_flags(&cxxflags, &prefix) {
        shim.flag(flag);
    }

    shim.opt_level(3);

    shim.compile("cubecl_llvm_shim");

    println!("cargo:rustc-link-lib=static=lldELF");
    println!("cargo:rustc-link-lib=static=lldCommon");

    tracel_llvm_bundler::llvm_sys::link()?;

    Ok(())
}
