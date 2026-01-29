use std::collections::BTreeMap;

fn main() {
    let enable_runtime = cfg!(feature = "test-runtime");

    println!("cargo:rustc-check-cfg=cfg(test_runtime_default)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_cpu)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_cuda)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_hip)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_metal)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_wgpu)");

    if enable_runtime {
        let map = BTreeMap::from([
            ("cpu", cfg!(feature = "cpu")),
            ("cuda", cfg!(feature = "cuda")),
            ("hip", cfg!(feature = "hip")),
            ("metal", cfg!(feature = "metal")),
            ("wgpu", cfg!(feature = "wgpu")),
        ]);

        let enabled_features = map
            .iter()
            .filter(|(_, enabled)| **enabled)
            .map(|(k, _)| *k)
            .collect::<Vec<_>>();

        if enabled_features.is_empty() || enabled_features.len() > 1 {
            println!("cargo:rustc-cfg=test_runtime_default");
        } else {
            println!("cargo:rustc-cfg=test_runtime_{}", enabled_features[0]);
        }
    }
}
