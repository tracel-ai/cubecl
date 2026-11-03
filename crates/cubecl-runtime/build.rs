use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        // Some features like autotune caching, compilation caching, and config loading
        // require std with OS-level filesystem and environment access.
        std_io: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")) },
        persistence: { all(feature = "persistence", any(std_io, target_family = "wasm")) },
        // The results and errors the tune cache stores and the autotune log
        // records derive serde wherever either exists.
        serializable: { any(std_io, persistence) },
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
    }

    // Cargo sets `DEBUG` from the `debug` key of the profile this crate is built with.
    println!("cargo::rustc-check-cfg=cfg(cubecl_debug_info)");
    if std::env::var("DEBUG").is_ok_and(|debug| debug == "true") {
        println!("cargo:rustc-cfg=cubecl_debug_info");
    }
}
