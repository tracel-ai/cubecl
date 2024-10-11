use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        autotune_persistent_cache: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos")) },
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
    }
}
