//! Writing the flat bundle format.
//!
//! The reader lives in [`super::embedded`] and documents the layout. Writing
//! is native-only on purpose: a bundle for any target is produced on a
//! development machine, and only consumed elsewhere.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

use super::embedded::{ENTRY_SIZE, FORMAT_VERSION, MAGIC};
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

    let mut index = Bytes::from_bytes_vec(Vec::with_capacity(entries.len() * ENTRY_SIZE));
    let mut data = Bytes::from_bytes_vec(Vec::new());

    for ((namespace, key), value) in entries {
        let id = ids[namespace.as_str()];

        let key_offset = data.len();
        data.extend_from_byte_slice(key);
        let value_offset = data.len();
        data.extend_from_byte_slice(value);

        index.extend_from_byte_slice(&id.to_le_bytes());
        push_u32(&mut index, key_offset)?;
        push_u32(&mut index, key.len())?;
        push_u32(&mut index, value_offset)?;
        push_u32(&mut index, value.len())?;
    }

    let mut bytes = Bytes::from_bytes_vec(Vec::with_capacity(
        64 + metadata.len() + index.len() + data.len(),
    ));
    bytes.extend_from_byte_slice(MAGIC);
    bytes.extend_from_byte_slice(&FORMAT_VERSION.to_le_bytes());

    push_u32(&mut bytes, metadata.len())?;
    bytes.extend_from_byte_slice(&metadata);

    push_u32(&mut bytes, namespaces.len())?;
    for namespace in &namespaces {
        push_u32(&mut bytes, namespace.len())?;
        bytes.extend_from_byte_slice(namespace.as_bytes());
    }

    push_u32(&mut bytes, entries.len())?;
    bytes.extend_from_byte_slice(&index);
    bytes.extend_from_byte_slice(&data);

    std::fs::write(out, &*bytes)?;

    Ok(())
}

/// Appends `value` as the little-endian `u32` the format is written in.
///
/// Every length and offset of the layout goes through here, so a bundle too
/// large to address is reported rather than truncated.
fn push_u32(out: &mut Bytes, value: usize) -> Result<(), BundleError> {
    let value = u32::try_from(value).map_err(|_| BundleError::TooLarge)?;
    out.extend_from_byte_slice(&value.to_le_bytes());

    Ok(())
}
