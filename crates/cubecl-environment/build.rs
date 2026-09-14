use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
        // Targets with a working `thread_local!` (includes single-threaded wasm with std).
        stream_local: { feature = "std" },
        // Filesystem and environment access for config loading and caching.
        std_io: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")) },
        // Native durable persistence and bundle tooling.
        native_cache: { all(feature = "persistence", std_io) },
        // Browser durable persistence.
        browser_cache: { all(target_family = "wasm", feature = "persistence") },
        // Tokio runtime support (never on wasm).
        tokio_rt: { all(feature = "tokio", not(target_family = "wasm")) },
        // TODO: We can't yet activate it for everything because of how error handling is done in matmul.
        backtrace: { all(test, feature="std") },
    }
}
