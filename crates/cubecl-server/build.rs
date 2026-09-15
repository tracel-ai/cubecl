use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        // Some features like autotune caching, compilation caching, and config loading
        // require std with OS-level filesystem and environment access.
        std_io: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")) },
        // Durable persistence: a file on std_io targets, OPFS in the browser.
        persistence: { all(feature = "persistence", any(std_io, target_family = "wasm")) },
        // Tests only. The library reads `EXCLUSIVE_MEMORY_ONLY` from
        // `cubecl-runtime`, which alone decides it; a test run turns the
        // feature on through this crate's, which forwards there.
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
    }
}
