use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=CUBECL_DEBUG_PLIRON");
    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }

    println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/lld.cpp");
    println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/device_libs.cpp");
    println!("cargo::rerun-if-changed=src/amdgpu/cpp_shims/printf.cpp");
    let prefix = tracel_llvm_bundler::config::llvm_path()?.into_os_string();
    let mut shim = cc::Build::new();
    shim.cpp(true)
        .file("src/amdgpu/cpp_shims/lld.cpp")
        .file("src/amdgpu/cpp_shims/device_libs.cpp")
        .file("src/amdgpu/cpp_shims/printf.cpp");

    for flag in tracel_llvm_bundler::config::get_cxxflags(Some(&prefix))?.split_whitespace() {
        match flag.strip_prefix("-I") {
            Some(dir) => shim.flag("-isystem").flag(dir),
            None => shim.flag(flag),
        };
    }

    shim.opt_level(3);

    shim.compile("cubecl_llvm_shim");

    println!("cargo:rustc-link-lib=static=lldELF");
    println!("cargo:rustc-link-lib=static=lldCommon");

    tracel_llvm_bundler::llvm_sys::link()?;

    Ok(())
}
