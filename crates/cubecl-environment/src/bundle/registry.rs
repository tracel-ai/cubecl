use alloc::vec::Vec;

use crate::sync::Arc;

use super::SeedSource;

static SEEDS: spin::Mutex<Vec<Arc<dyn SeedSource>>> = spin::Mutex::new(Vec::new());

/// Installs a seed source globally.
///
/// Stores opened afterward consult it automatically. Install bundles before
/// initializing devices so caches opened at server construction see them.
pub fn install(source: Arc<dyn SeedSource>) {
    log::debug!("Installing bundle seed source: {}", source.describe());
    SEEDS.lock().push(source);
}

/// Opens and installs every bundle directory in `paths`.
///
/// Any bundle that fails to load is skipped with a warning; this never fails.
#[cfg(feature = "cache")]
pub fn install_from_paths<P: AsRef<std::path::Path>>(paths: &[P]) {
    for path in paths {
        let path = path.as_ref();
        match super::Bundle::open(path) {
            Ok(bundle) => install(Arc::new(bundle)),
            Err(err) => {
                log::warn!("Skipping bundle at {path:?}: {err}");
            }
        }
    }
}

/// The currently installed seed sources, in installation order.
pub fn seeds() -> Vec<Arc<dyn SeedSource>> {
    SEEDS.lock().clone()
}

/// Removes every installed seed source. Mostly useful in tests.
pub fn clear() {
    SEEDS.lock().clear();
}
