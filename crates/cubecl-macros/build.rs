use std::env;

// Allow overriding nightly macro features on the end user side without having to propagate the feature
// everywhere
fn main() {
    println!("cargo::rustc-check-cfg=cfg(nightly)");
    println!("cargo::rustc-check-cfg=cfg(debug_symbols)");
    println!("cargo:rerun-if-env-changed=CUBECL_DEBUG_NIGHTLY");
    println!("cargo:rerun-if-env-changed=CUBECL_DEBUG");

    let nightly_feature_enabled = env::var("CARGO_FEATURE_NIGHTLY").is_ok();
    let nightly_override_enabled = env::var("CUBECL_DEBUG_NIGHTLY").is_ok();

    let debug_feature_enabled = env::var("CARGO_FEATURE_DEBUG_SYMBOLS").is_ok();
    let debug_override_enabled = env::var("CUBECL_DEBUG").is_ok();

    if nightly_feature_enabled || nightly_override_enabled {
        println!("cargo:rustc-cfg=nightly");
    }

    if debug_feature_enabled || debug_override_enabled {
        println!("cargo:rustc-cfg=debug_symbols");
    }
}
