use alloc::string::String;
use alloc::vec::Vec;

use crate::bytes::Bytes;
use crate::persistence::{InsertSummary, Origin, storage};

use super::Bundle;

/// How many entries one storage transaction carries.
///
/// A bundle is imported in batches rather than entry by entry, because each
/// batch is one exclusive lock on the cache root and a 10k-entry bundle would
/// otherwise block every other writer 10k times. The batch is buffered in
/// memory, so it is bounded rather than "the whole bundle".
const BATCH: usize = 1024;

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
    /// Entries the storage refused to write: a full disk, an unwritable cache
    /// root. They are simply absent, so the application recomputes them.
    pub failed: usize,
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

        let mut batch: Vec<(Bytes, Bytes)> = Vec::with_capacity(BATCH);
        let mut total = InsertSummary::default();
        let commit = |batch: &mut Vec<(Bytes, Bytes)>, total: &mut InsertSummary| {
            if batch.is_empty() {
                return;
            }

            let summary = target.insert_many(&mut batch.drain(..), Origin::Imported);
            total.stored += summary.stored;
            total.conflict += summary.conflict;
            total.failed += summary.failed;
        };

        bundle.scan(&namespace, &mut |key, value| {
            batch.push((
                Bytes::from_bytes_vec(key.to_vec()),
                Bytes::from_bytes_vec(value.to_vec()),
            ));
            if batch.len() == BATCH {
                commit(&mut batch, &mut total);
            }
        });
        commit(&mut batch, &mut total);

        let (imported, skipped) = (total.stored, total.conflict);
        log::debug!("Imported {imported} entries into {namespace} ({skipped} already present)");
        if total.failed > 0 {
            log::warn!("Failed to import {} entries into {namespace}", total.failed);
        }

        report.namespaces.push(namespace);
        report.imported += imported;
        report.skipped += skipped;
        report.failed += total.failed;
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
