use std::collections::BTreeMap;
use std::env;

fn main() {
    // Automatically enable pliron-dump if an output path is set
    println!("cargo:rerun-if-env-changed=CUBECL_DEBUG_PLIRON");

    if env::var("CUBECL_DEBUG_PLIRON").is_ok() && env::var("CARGO_FEATURE_STD").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pliron-dump\"");
    }

    println!("cargo:rustc-check-cfg=cfg(target_default)");
    println!("cargo:rustc-check-cfg=cfg(target_cuda)");
    println!("cargo:rustc-check-cfg=cfg(target_hip)");
    println!("cargo:rustc-check-cfg=cfg(target_metal)");

    let map = BTreeMap::from([
        ("cuda", cfg!(feature = "cuda")),
        ("hip", cfg!(feature = "hip")),
        ("metal", cfg!(feature = "metal")),
    ]);

    let enabled_features = map
        .iter()
        .filter(|(_, enabled)| **enabled)
        .map(|(k, _)| *k)
        .collect::<Vec<_>>();

    if enabled_features.is_empty() || enabled_features.len() > 1 {
        println!("cargo:rustc-cfg=target_default");
    } else {
        println!("cargo:rustc-cfg=target_{}", enabled_features[0]);
    }
}
