use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        autotune_persistent_cache: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos")) },
    }
}
