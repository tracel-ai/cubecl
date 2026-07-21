//! Named environments.
//!
//! An environment is one named local store: a single database holding every
//! namespace this machine has warmed. Exactly one is active at a time, and
//! every [`Store`] bound to it goes to it.
//!
//! Like `std`, the environment is a namespace rather than a value: a set of
//! functions over one global state, not an `Environment` struct to pass
//! around. [`store`] creates stores in it, [`bundle`] captures it for
//! shipping, and [`activate`]/[`set_root`]/[`load`] switch it.
//!
//! Naming them makes it possible to keep several side by side, which is what
//! you want when the same checkout targets more than one machine or you want a
//! throwaway environment for an experiment:
//!
//! ```ignore
//! cubecl_environment::environment::activate("h100");
//! ```
//!
//! Switching is dynamic: every store bound to the environment detects the
//! switch and resets — the in-memory cache is dropped and the storage is
//! reopened against the new environment. Detection is one atomic load on the
//! store's read path, so an environment that never switches costs nothing.

use alloc::string::{String, ToString};
#[cfg(std_io)]
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::persistence::{StoreKey, StoreValue};
use crate::sync::{Arc, Lazy, Mutex};

pub use crate::persistence::{CacheOption, Namespace, Store, StoreOptions};

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
    /// Shared rather than owned per reader: [`active`] is called on every path
    /// that opens a store, and the name is immutable once [`activate`] set it,
    /// so handing out a handle costs a refcount bump instead of an allocation.
    name: Arc<str>,
    #[cfg(std_io)]
    root: Option<std::path::PathBuf>,
    /// An explicit database file mounted by [`load`], overriding
    /// `<root>/<name>.db`. Cleared by [`activate`] and [`set_root`], which
    /// select named environments again.
    #[cfg(std_io)]
    file: Option<std::path::PathBuf>,
}

static ACTIVE: Lazy<Mutex<Active>> = Lazy::new(|| {
    Mutex::new(Active {
        name: Arc::from(DEFAULT),
        #[cfg(std_io)]
        root: None,
        #[cfg(std_io)]
        file: None,
    })
});

/// Bumped on every switch. Stores bound to the environment record the value
/// they were opened under and compare on access, which is what lets a switch
/// reach stores that already exist without any registry of them: a mismatch
/// reads as "reset before serving".
static GENERATION: AtomicU32 = AtomicU32::new(0);

/// An opaque token that changes on every environment switch.
///
/// Record it when deriving state from the environment — an index built over a
/// [`Store`], a map hydrated from one — and compare on access: a different
/// value means the derived state describes an environment that is no longer
/// active and must be rebuilt. This is the same mechanism [`Store`] uses to
/// reset itself, exposed for state the stores can't see. The load is relaxed
/// and costs nothing on hot paths.
pub fn generation() -> u32 {
    GENERATION.load(Ordering::Relaxed)
}

/// Called under the [`ACTIVE`] lock by everything that switches, so a store
/// can never observe the new generation with the old location.
fn switched() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// A consistent snapshot of both fields. Only the paths that need the root
/// take it; [`active`] reads the name directly rather than allocating a
/// `PathBuf` it would discard.
#[cfg(std_io)]
fn active_state() -> Active {
    ACTIVE.lock().clone()
}

/// Makes `name` the active environment.
///
/// Takes effect immediately: stores bound to the previous environment reset
/// on their next access and reopen against this one.
pub fn activate<N: AsRef<str>>(name: N) {
    let name = sanitize(name.as_ref());
    log::debug!("Activating environment '{name}'");

    let mut active = ACTIVE.lock();
    active.name = name.into();
    #[cfg(std_io)]
    {
        active.file = None;
    }
    switched();
}

/// A stable identity for the active environment, distinguishing one from
/// another for backends that key by it rather than by a file path.
///
/// The database backend already isolates environments by their file path; the
/// in-memory fallback ([`MemoryStorage`](crate::persistence::MemoryStorage))
/// has no file, so it scopes its process-wide entries by this instead, and a
/// switch reaches it the same way. On targets with a file system the identity
/// is the database path, which folds in the name, the root and any mounted
/// file; elsewhere the name is the whole identity, since [`load`]/[`set_root`]
/// don't exist there.
#[cfg(std_io)]
pub(crate) fn scope() -> String {
    path().display().to_string()
}

#[cfg(not(std_io))]
pub(crate) fn scope() -> String {
    active().to_string()
}

/// The active environment.
///
/// The returned handle derefs to `str`, and cloning it is a refcount bump, so
/// this is cheap enough to call wherever a store is opened.
pub fn active() -> Arc<str> {
    // Deliberately not through `active_state`: that snapshots the root too,
    // which would allocate a `PathBuf` this caller never looks at.
    ACTIVE.lock().name.clone()
}

/// Sets the directory environments are kept in.
///
/// Like [`activate`], this takes effect immediately for every bound store.
#[cfg(std_io)]
pub fn set_root<P: Into<std::path::PathBuf>>(root: P) {
    let root = root.into();
    log::debug!("Environments rooted at {root:?}");

    let mut active = ACTIVE.lock();
    active.root = Some(root);
    active.file = None;
    switched();
}

/// Mounts the database at `file` as the active environment.
///
/// This is how a shipped [`BundleFormat::Sqlite`](crate::bundle::BundleFormat)
/// bundle is used in place: a bundle file carries the same schema as an
/// environment, so loading it makes its entries the ones every bound store
/// serves, with nothing copied. Stores reset on their next access, exactly as
/// with [`activate`].
///
/// The file stays the environment until [`activate`] or [`set_root`] selects
/// a named one again. Writes (newly tuned keys, freshly compiled kernels) land
/// in it like in any environment; if its location is read-only, they degrade
/// to in-memory persistence as usual.
#[cfg(std_io)]
pub fn load<P: Into<std::path::PathBuf>>(file: P) {
    let file = file.into();
    log::debug!("Loading environment from {file:?}");

    let mut active = ACTIVE.lock();
    active.file = Some(file);
    switched();
}

/// The directory environments are kept in, defaulting to the standard cache
/// root.
#[cfg(std_io)]
pub fn root() -> std::path::PathBuf {
    active_state().root_or_default()
}

/// The database file of the active environment: the file mounted by
/// [`load`], or `<root>/<name>.db`.
///
/// Everything comes from one snapshot, so this never mixes the name from one
/// configuration with the root from another.
#[cfg(std_io)]
pub fn path() -> std::path::PathBuf {
    let active = active_state();

    match active.file.clone() {
        Some(file) => file,
        None => {
            let name = active.name.clone();
            active.root_or_default().join(file_name(&name))
        }
    }
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

/// A [`Store`] created from the options, bound to the active environment
/// whenever the options name a storage.
///
/// ```ignore
/// let store: Store<Key, Value> = cubecl_environment::environment::store(
///     StoreOptions::new()
///         .storage(Namespace::new("cuda/ptx"))
///         .cache(CacheOption::Lazy),
/// );
/// ```
pub fn store<K: StoreKey, V: StoreValue>(options: StoreOptions) -> Store<K, V> {
    Store::new(options)
}

/// The active environment, captured for shipping.
///
/// [`save`](Bundle::save) is the whole API: it exports what the environment
/// holds into a bundle file another machine can [`load`] or
/// [`import`](crate::bundle::import).
#[cfg(native_cache)]
#[derive(Debug, Clone)]
pub struct Bundle {
    /// The database file the environment lived in when captured.
    source: std::path::PathBuf,
    /// The environment's name, which becomes the bundle's default name.
    name: String,
}

/// Captures the active environment; see [`Bundle`].
#[cfg(native_cache)]
pub fn bundle() -> Bundle {
    Bundle {
        source: path(),
        name: active().to_string(),
    }
}

#[cfg(native_cache)]
impl Bundle {
    /// Exports the captured environment to `out` in `format`.
    ///
    /// A thin front for [`bundle::export`](crate::bundle::export) over this
    /// one environment; use `export` directly to merge several roots or
    /// restrict the namespaces.
    pub fn save<P: AsRef<std::path::Path>>(
        &self,
        out: P,
        format: crate::bundle::BundleFormat,
    ) -> Result<crate::bundle::BundleManifest, crate::bundle::BundleError> {
        let options = crate::bundle::ExportOptions {
            name: self.name.clone(),
            format,
            ..Default::default()
        };

        crate::bundle::export(&[&self.source], out, &options)
    }
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
