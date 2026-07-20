use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        // Some features like autotune caching, compilation caching, and config loading
        // require std with OS-level filesystem and environment access.
        std_io: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")) },
        // Browser storage persistence (IndexedDB).
        browser_cache: { all(target_family = "wasm", feature = "browser-cache") },
        // Autotune results can persist: on disk (std_io) or in browser storage.
        autotune_persistence: { any(std_io, browser_cache) },
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
    }
}
