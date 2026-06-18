use std::collections::BTreeMap;

fn main() {
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
