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
#[cfg(std_io)]
use alloc::vec::Vec;

use crate::sync::{Lazy, Mutex};

/// The environment used when none is chosen.
pub const DEFAULT: &str = "default";

/// The extension of an environment's database file.
#[cfg(std_io)]
pub const EXTENSION: &str = "db";

/// The active environment: its name, and where environments are kept.
///
/// Both live here rather than being passed per store, because an environment
/// *is* the store: letting one cache be opened under a different root would
/// make "a single environment" untrue. Anything that needs both reads them
/// through a single [`active_state`] snapshot, so a concurrent [`activate`] or
/// [`set_root`] can never be observed half-applied.
#[derive(Debug, Clone)]
struct Active {
    name: String,
    #[cfg(std_io)]
    root: Option<std::path::PathBuf>,
}

static ACTIVE: Lazy<Mutex<Active>> = Lazy::new(|| {
    Mutex::new(Active {
        name: String::from(DEFAULT),
        #[cfg(std_io)]
        root: None,
    })
});

fn active_state() -> Active {
    ACTIVE.lock().clone()
}

/// Makes `name` the active environment.
///
/// Only affects stores opened afterwards. Call it before devices are
/// initialized, so the caches they open land in the right place.
pub fn activate<N: AsRef<str>>(name: N) {
    let name = sanitize(name.as_ref());
    log::debug!("Activating environment '{name}'");

    ACTIVE.lock().name = name;
}

/// The active environment.
pub fn active() -> String {
    active_state().name
}

/// Sets the directory environments are kept in.
///
/// Like [`activate`], this only affects stores opened afterwards.
#[cfg(std_io)]
pub fn set_root<P: Into<std::path::PathBuf>>(root: P) {
    let root = root.into();
    log::debug!("Environments rooted at {root:?}");

    ACTIVE.lock().root = Some(root);
}

/// The directory environments are kept in, defaulting to the standard cache
/// root.
#[cfg(std_io)]
pub fn root() -> std::path::PathBuf {
    active_state().root_or_default()
}

/// The database file of the active environment.
///
/// Name and root come from one snapshot, so this never mixes the name from one
/// configuration with the root from another.
#[cfg(std_io)]
pub fn path() -> std::path::PathBuf {
    let active = active_state();
    let name = file_name(&active.name);

    active.root_or_default().join(name)
}

#[cfg(std_io)]
impl Active {
    /// Where environments are kept, falling back to the standard cache root.
    fn root_or_default(self) -> std::path::PathBuf {
        self.root
            .unwrap_or_else(|| crate::persistence::CacheConfig::default().root())
    }
}

/// The file name holding the active environment inside a cache root.
#[cfg(std_io)]
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

/// Every environment that exists, sorted by name.
#[cfg(std_io)]
pub fn list() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root()) else {
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
#[cfg(std_io)]
pub fn namespaces() -> Vec<crate::persistence::NamespaceSummary> {
    #[cfg(native_cache)]
    match crate::persistence::Database::open_active() {
        Some(database) => database.summary(),
        // No database means nothing was ever written to disk; whatever this
        // process warmed is in memory.
        None => crate::persistence::MemoryStorage::namespaces(),
    }

    // Without a persistence backend there is nothing durable to report on.
    #[cfg(not(native_cache))]
    crate::persistence::MemoryStorage::namespaces()
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
