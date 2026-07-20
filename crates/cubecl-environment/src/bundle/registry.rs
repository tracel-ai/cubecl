use alloc::vec::Vec;

use crate::sync::Arc;

use super::Bundle;

static INSTALLED: spin::Mutex<Vec<Arc<dyn Bundle>>> = spin::Mutex::new(Vec::new());

/// Installs a bundle globally.
///
/// Stores opened afterward consult it automatically. Install bundles before
/// initializing devices so caches opened at server construction see them.
pub fn install(bundle: Arc<dyn Bundle>) {
    log::debug!("Installing bundle: {}", bundle.describe());
    INSTALLED.lock().push(bundle);
}

/// Opens and installs every bundle file in `paths`.
///
/// Any bundle that fails to load is skipped with a warning; this never fails.
#[cfg(feature = "cache")]
pub fn install_from_paths<P: AsRef<std::path::Path>>(paths: &[P]) {
    for path in paths {
        let path = path.as_ref();
        match super::SqliteBundle::open(path) {
            Ok(bundle) => install(Arc::new(bundle)),
            Err(err) => {
                log::warn!("Skipping bundle at {path:?}: {err}");
            }
        }
    }
}

/// The currently installed bundles, in installation order.
pub fn installed() -> Vec<Arc<dyn Bundle>> {
    INSTALLED.lock().clone()
}

/// Removes every installed bundle. Mostly useful in tests.
pub fn clear() {
    INSTALLED.lock().clear();
}
