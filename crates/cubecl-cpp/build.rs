use std::env;

fn main() {
    // Automatically enable pliron-dump if an output path is set
    println!("cargo:rerun-if-env-changed=CUBECL_DEBUG_PLIRON");

    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }
}
