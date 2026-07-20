//! Named environments.
//!
//! An environment is one named local store: a single database holding every
//! namespace this machine has warmed. Exactly one is active at a time, and
//! everything opened afterwards goes to it.
//!
//! Naming them makes it possible to keep several side by side, which is what
//! you want when the same checkout targets more than one machine or you want a
//! throwaway environment for an experiment:
//!
//! ```ignore
//! cubecl_environment::environment::activate("h100");
//! ```
//!
//! Switching affects stores opened *after* the call. Existing stores keep the
//! storage they were opened with, so activate before initializing devices,
//! exactly where you would import a bundle.

use alloc::string::{String, ToString};
#[cfg(feature = "cache")]
use alloc::vec::Vec;

use crate::sync::{Lazy, Mutex};

/// The environment used when none is chosen.
pub const DEFAULT: &str = "default";

/// The extension of an environment's database file.
#[cfg(feature = "cache")]
pub const EXTENSION: &str = "db";

static ACTIVE: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::from(DEFAULT)));

/// Makes `name` the active environment.
///
/// Only affects stores opened afterwards. Call it before devices are
/// initialized, so the caches they open land in the right place.
pub fn activate<N: AsRef<str>>(name: N) {
    let name = sanitize(name.as_ref());
    log::debug!("Activating environment '{name}'");

    *ACTIVE.lock().expect("Lock recovers from poisoning") = name;
}

/// The active environment.
pub fn active() -> String {
    ACTIVE.lock().expect("Lock recovers from poisoning").clone()
}

/// The file name holding the active environment inside a cache root.
#[cfg(feature = "cache")]
pub fn file_name(name: &str) -> String {
    alloc::format!("{}.{EXTENSION}", sanitize(name))
}

/// An environment name reduced to something safe to use as a file name.
///
/// Anything outside `[A-Za-z0-9._-]` becomes `_`, and an empty or
/// dot-only name falls back to [`DEFAULT`], so a name can never escape the
/// cache root.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        return DEFAULT.to_string();
    }

    cleaned
}

/// Every environment that exists under `root`, sorted by name.
#[cfg(feature = "cache")]
pub fn list(root: &std::path::Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };

    let mut names: Vec<String> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? != EXTENSION {
                return None;
            }
            Some(path.file_stem()?.to_string_lossy().to_string())
        })
        .collect();

    names.sort();
    names
}

/// What the active environment currently holds, one row per namespace.
///
/// This is what you consult before bundling, to see which namespaces are warm
/// and worth shipping.
#[cfg(feature = "cache")]
pub fn namespaces(root: &std::path::Path) -> Vec<crate::persistence::NamespaceSummary> {
    match crate::persistence::Database::open_root(root) {
        Some(database) => database.summary(),
        // No database means nothing was ever written to disk; whatever this
        // process warmed is in memory.
        None => crate::persistence::MemoryStorage::namespaces(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_can_never_escape_the_cache_root() {
        assert_eq!(sanitize("../../etc/passwd"), ".._.._etc_passwd");
        assert_eq!(sanitize("a/b"), "a_b");
        assert_eq!(sanitize(""), DEFAULT);
        assert_eq!(sanitize(".."), DEFAULT);
        assert_eq!(sanitize("h100-linux"), "h100-linux");
    }
}
