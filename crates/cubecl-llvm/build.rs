use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=CUBECL_DEBUG_PLIRON");
    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }

    // Static archives resolve left to right, so the link line has to run
    // shim -> lld -> LLVM. `cc` emits the shim's own link directive from
    // `compile`, so it must be called before the lld libs are printed, and both
    // before the bundler's LLVM list. lld is not in `llvm-config --libs`.
    println!("cargo::rerun-if-changed=src/amdgpu/lld_shim.cpp");
    let prefix = tracel_llvm_bundler::config::llvm_path()?.into_os_string();
    let mut shim = cc::Build::new();
    shim.cpp(true).file("src/amdgpu/lld_shim.cpp");
    for flag in tracel_llvm_bundler::config::get_cxxflags(Some(&prefix))?.split_whitespace() {
        shim.flag(flag);
    }
    shim.compile("cubecl_lld_shim");

    println!("cargo:rustc-link-lib=static=lldELF");
    println!("cargo:rustc-link-lib=static=lldCommon");

    tracel_llvm_bundler::llvm_sys::link()?;

    Ok(())
}
