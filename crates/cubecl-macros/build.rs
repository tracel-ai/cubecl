use std::env;

// Allow overriding nightly macro features on the end user side without having to propagate the feature
// everywhere
fn main() {
    println!("cargo::rustc-check-cfg=cfg(debug_symbols)");
    println!("cargo:rerun-if-env-changed=CUBECL_DEBUG");

    let debug_feature_enabled = env::var("CARGO_FEATURE_DEBUG_SYMBOLS").is_ok();
    let debug_override_enabled = env::var("CUBECL_DEBUG").is_ok();
    let debug_enabled = debug_feature_enabled || debug_override_enabled;

    if debug_enabled {
        println!("cargo:rustc-cfg=debug_symbols");
    }
}
