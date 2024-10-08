use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
    }
}
