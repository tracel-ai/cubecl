use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        // Some features like autotune caching, compilation caching, and config loading rely on files/environment variables,
        // which are only applicable on "standard desktop platforms".
        std_io: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos")) },
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
    }
}
