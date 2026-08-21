use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=CUBECL_DEBUG_PLIRON");
    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }

    tracel_llvm_bundler::llvm_sys::link()?;

    Ok(())
}
