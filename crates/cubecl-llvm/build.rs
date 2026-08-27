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
    println!("cargo::rerun-if-changed=src/amdgpu/device_libs_shim.cpp");
    println!("cargo::rerun-if-changed=src/amdgpu/printf_shim.cpp");
    let prefix = tracel_llvm_bundler::config::llvm_path()?.into_os_string();
    let mut shim = cc::Build::new();
    shim.cpp(true)
        .file("src/amdgpu/lld_shim.cpp")
        .file("src/amdgpu/device_libs_shim.cpp")
        .file("src/amdgpu/printf_shim.cpp");

    for flag in tracel_llvm_bundler::config::get_cxxflags(Some(&prefix))?.split_whitespace() {
        // LLVM's headers do not build clean under the `-Wall -Wextra` that `cc` adds, and
        // they are not ours to fix: taken as ordinary includes they bury the shims' own
        // diagnostics under pages of template instantiation. `-isystem` keeps the warnings
        // that are about the code in this crate.
        match flag.strip_prefix("-I") {
            Some(dir) => shim.flag("-isystem").flag(dir),
            None => shim.flag(flag),
        };
    }

    // A hardened toolchain predefines `_FORTIFY_SOURCE`, and glibc warns once per file when
    // that is compiled without optimization, fortification being a thing the optimizer does.
    // Undefining it does not help: the hardening wrapper appends its own `-D` after these
    // flags and wins. So give the shims the optimization instead, which is what the warning
    // asks for and costs nothing on three functions of glue. Release profiles already pass
    // more than this, and are left alone.
    if env::var("OPT_LEVEL").as_deref() == Ok("0") {
        shim.opt_level(1);
    }

    shim.compile("cubecl_llvm_shim");

    println!("cargo:rustc-link-lib=static=lldELF");
    println!("cargo:rustc-link-lib=static=lldCommon");

    tracel_llvm_bundler::llvm_sys::link()?;

    Ok(())
}
