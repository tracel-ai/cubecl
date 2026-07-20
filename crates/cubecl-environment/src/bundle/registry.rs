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

/// Opens and installs every bundle file in `paths`, in either format.
///
/// Any bundle that fails to load is skipped with a warning; this never fails.
#[cfg(feature = "cache")]
pub fn install_from_paths<P: AsRef<std::path::Path>>(paths: &[P]) {
    for path in paths {
        let path = path.as_ref();
        match open_path(path) {
            Ok(bundle) => install(bundle),
            Err(err) => {
                log::warn!("Skipping bundle at {path:?}: {err}");
            }
        }
    }
}

/// Opens a bundle file, whichever format it is in.
///
/// The `SQLite` format is tried first because recognizing it is a cheap header
/// read; a flat bundle has to be loaded into memory to be used at all.
#[cfg(feature = "cache")]
pub fn open_path(path: &std::path::Path) -> Result<Arc<dyn Bundle>, super::BundleError> {
    match super::SqliteBundle::open(path) {
        Ok(bundle) => return Ok(Arc::new(bundle)),
        Err(err) => log::debug!("{path:?} is not a SQLite bundle ({err}), trying the flat format"),
    }

    let bytes = crate::bytes::Bytes::from_bytes_vec(std::fs::read(path)?);
    let bundle = super::EmbeddedBundle::open(bytes)
        .map_err(|err| super::BundleError::InvalidManifest(alloc::format!("{err}")))?;

    Ok(Arc::new(bundle))
}

/// The currently installed bundles, in installation order.
pub fn installed() -> Vec<Arc<dyn Bundle>> {
    INSTALLED.lock().clone()
}

/// Removes every installed bundle. Mostly useful in tests.
pub fn clear() {
    INSTALLED.lock().clear();
}
