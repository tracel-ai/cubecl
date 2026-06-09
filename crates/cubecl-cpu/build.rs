use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // required on macos
    tracel_llvm_bundler::config::set_homebrew_library_path()?;

    if env::var("CUBECL_DEBUG_MLIR").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"mlir-dump\"");
    }

    Ok(())
}
