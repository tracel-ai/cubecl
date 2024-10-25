fn main() {
    if let Ok(family) = std::env::var("CARGO_CFG_TARGET_FAMILY") {
        if family == "wasm" {
            println!("cargo:rustc-cfg=portable_atomic_unsafe_assume_single_core");
        }
    }
}
