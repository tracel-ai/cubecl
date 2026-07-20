use alloc::string::String;
use alloc::vec::Vec;

use crate::persistence::{Origin, storage};

use super::Bundle;

/// What an [`import`] copied.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportReport {
    /// Namespaces the bundle held entries for.
    pub namespaces: Vec<String>,
    /// Entries written into the local storage.
    pub imported: usize,
    /// Entries skipped because the local storage already had the key. A local
    /// value is never overwritten by a bundle.
    pub skipped: usize,
}

/// Copies every entry of `bundle` into the local storage.
///
/// This is the only thing a bundle is for. Once imported, entries are ordinary
/// storage rows and the bundle can be deleted: nothing consults it at runtime,
/// so a lookup never depends on a file staying installed.
///
/// Importing is insert-only and therefore idempotent: a key the storage
/// already holds keeps its value, whether it was computed here or imported
/// earlier. Entries land with [`Origin::Imported`], which lets a locally
/// computed value replace them later if the bundle turns out to be stale.
///
/// Fills the *active* environment; switch with
/// [`environment::activate`](crate::environment::activate) beforehand to
/// target another one.
pub fn import(bundle: &dyn Bundle) -> ImportReport {
    let mut report = ImportReport::default();

    for namespace in bundle.namespaces() {
        let target = storage::open(&namespace);
        let mut imported = 0;
        let mut skipped = 0;

        bundle.scan(
            &namespace,
            &mut |key, value| match target.insert(key, value, Origin::Imported) {
                Some(_) => skipped += 1,
                None => imported += 1,
            },
        );

        log::debug!("Imported {imported} entries into {namespace} ({skipped} already present)");

        report.namespaces.push(namespace);
        report.imported += imported;
        report.skipped += skipped;
    }

    log::info!(
        "Imported {} entries from {} into {} namespaces ({} already present)",
        report.imported,
        bundle.describe(),
        report.namespaces.len(),
        report.skipped,
    );

    report
}
