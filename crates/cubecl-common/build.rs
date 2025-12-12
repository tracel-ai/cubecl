use cfg_aliases::cfg_aliases;

fn main() {
    if let Ok(family) = std::env::var("CARGO_CFG_TARGET_FAMILY")
        && family == "wasm"
    {
        println!("cargo:rustc-cfg=portable_atomic_unsafe_assume_single_core");
    }
    // Setup cfg aliases
    cfg_aliases! {
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
        // TODO: We can't yet activate it for everything because of who error handling is done in matmul.
        backtrace: { all(test, features="std") },
    }
}
