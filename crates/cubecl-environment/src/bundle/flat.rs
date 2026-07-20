//! Writing the flat bundle format.
//!
//! The reader lives in [`super::embedded`] and documents the layout. Writing
//! is native-only on purpose: a bundle for any target is produced on a
//! development machine, and only consumed elsewhere.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

use super::embedded::{FORMAT_VERSION, MAGIC};
use super::{BundleError, BundleManifest};
use crate::bytes::Bytes;

/// The entries to write, deduplicated and ordered.
///
/// A `BTreeMap` keyed by `(namespace, key)` gives exactly the ordering the
/// reader binary-searches: namespace ids are assigned in sorted namespace
/// order, so ordering by namespace string is the same as ordering by id.
pub(crate) type Entries = BTreeMap<(String, Vec<u8>), Bytes>;

/// Serializes `entries` and `manifest` into the flat format.
pub(crate) fn write(
    out: &Path,
    entries: &Entries,
    manifest: &BundleManifest,
) -> Result<(), BundleError> {
    let metadata = serde_json::to_vec(manifest)
        .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;

    let namespaces: BTreeSet<&str> = entries
        .keys()
        .map(|(namespace, _)| namespace.as_str())
        .collect();
    let ids: BTreeMap<&str, u32> = namespaces
        .iter()
        .enumerate()
        .map(|(id, namespace)| (*namespace, id as u32))
        .collect();

    let mut index = Bytes::from_bytes_vec(Vec::with_capacity(entries.len() * 20));
    let mut data = Bytes::from_bytes_vec(Vec::new());

    for ((namespace, key), value) in entries {
        let id = ids[namespace.as_str()];

        let key_offset = u32::try_from(data.len()).map_err(|_| too_large())?;
        data.extend_from_byte_slice(key);
        let value_offset = u32::try_from(data.len()).map_err(|_| too_large())?;
        data.extend_from_byte_slice(value);

        index.extend_from_byte_slice(&id.to_le_bytes());
        index.extend_from_byte_slice(&key_offset.to_le_bytes());
        index.extend_from_byte_slice(
            &u32::try_from(key.len())
                .map_err(|_| too_large())?
                .to_le_bytes(),
        );
        index.extend_from_byte_slice(&value_offset.to_le_bytes());
        index.extend_from_byte_slice(
            &u32::try_from(value.len())
                .map_err(|_| too_large())?
                .to_le_bytes(),
        );
    }

    let mut bytes = Bytes::from_bytes_vec(Vec::with_capacity(
        64 + metadata.len() + index.len() + data.len(),
    ));
    bytes.extend_from_byte_slice(MAGIC);
    bytes.extend_from_byte_slice(&FORMAT_VERSION.to_le_bytes());

    bytes.extend_from_byte_slice(
        &u32::try_from(metadata.len())
            .map_err(|_| too_large())?
            .to_le_bytes(),
    );
    bytes.extend_from_byte_slice(&metadata);

    bytes.extend_from_byte_slice(
        &u32::try_from(namespaces.len())
            .map_err(|_| too_large())?
            .to_le_bytes(),
    );
    for namespace in &namespaces {
        bytes.extend_from_byte_slice(
            &u32::try_from(namespace.len())
                .map_err(|_| too_large())?
                .to_le_bytes(),
        );
        bytes.extend_from_byte_slice(namespace.as_bytes());
    }

    bytes.extend_from_byte_slice(
        &u32::try_from(entries.len())
            .map_err(|_| too_large())?
            .to_le_bytes(),
    );
    bytes.extend_from_byte_slice(&index);
    bytes.extend_from_byte_slice(&data);

    std::fs::write(out, &*bytes)?;

    Ok(())
}

/// Every span in the format is a `u32`, so a bundle is capped at 4 GiB.
fn too_large() -> BundleError {
    BundleError::InvalidManifest(String::from(
        "the flat bundle format addresses at most 4 GiB; export fewer namespaces",
    ))
}
