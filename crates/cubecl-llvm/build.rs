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

    Ok(())
}
