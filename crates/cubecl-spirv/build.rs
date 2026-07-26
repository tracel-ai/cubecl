use std::env;

fn main() {
    // Automatically enable spirv-dump if an output path is set
    println!("cargo:rerun-if-env-changed=CUBECL_DEBUG_SPIRV");

    if env::var("CUBECL_DEBUG_SPIRV").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"spirv-dump\"");
    }
}
