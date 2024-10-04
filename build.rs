use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        simple_memory_management: { any(feature = "simple-memory-management", target_family = "wasm") },
    }
}
