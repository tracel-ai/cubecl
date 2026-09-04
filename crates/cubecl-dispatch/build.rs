use std::collections::BTreeMap;

fn main() {
    let enable_runtime = cfg!(feature = "test-runtime");

    println!("cargo:rustc-check-cfg=cfg(any_runtime)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_default)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_cpu)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_cuda)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_hip)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_metal)");
    println!("cargo:rustc-check-cfg=cfg(test_runtime_wgpu)");

    let map = BTreeMap::from([
        ("cpu", cfg!(feature = "cpu")),
        ("cuda", cfg!(feature = "cuda")),
        ("hip", cfg!(feature = "hip")),
        ("metal", cfg!(feature = "metal-native")),
        ("wgpu", cfg!(feature = "wgpu")),
    ]);

    let enabled_features = map
        .iter()
        .filter(|(_, enabled)| **enabled)
        .map(|(k, _)| *k)
        .collect::<Vec<_>>();

    // Whether this build has a runtime to reach at all.
    if !enabled_features.is_empty() {
        println!("cargo:rustc-cfg=any_runtime");
    }

    // `test-runtime` turns wgpu on itself, so wgpu is the fallback: the tests
    // run on the one other runtime enabled, or on wgpu when there is none or
    // several.
    if enable_runtime {
        let others = enabled_features
            .iter()
            .filter(|name| **name != "wgpu")
            .collect::<Vec<_>>();
        match others.as_slice() {
            [only] => println!("cargo:rustc-cfg=test_runtime_{only}"),
            [] => println!("cargo:rustc-cfg=test_runtime_wgpu"),
            _ => println!("cargo:rustc-cfg=test_runtime_default"),
        }
    }
}
