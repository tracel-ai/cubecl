fn main() {
    println!("cargo::rustc-check-cfg=cfg(compilation_cache)");
    let has_persistence = std::env::var("CARGO_FEATURE_PERSISTENCE").is_ok();
    let is_wasm = std::env::var("CARGO_CFG_TARGET_FAMILY").is_ok_and(|f| f == "wasm");
    if has_persistence && !is_wasm {
        println!("cargo:rustc-cfg=compilation_cache");
    }
}
