//! The flat bundle format, readable anywhere.
//!
//! `SQLite` needs a file system, so it can't serve wasm or no-std targets.
//! This format exists for them: one contiguous blob of bytes, produced on a
//! development machine by [`export`](super::export) and consumed by
//! [`EmbeddedBundle`] either from `include_bytes!` or from bytes fetched at
//! runtime.
//!
//! # Layout
//!
//! All integers are little-endian `u32`. Offsets are relative to the start of
//! the data section.
//!
//! ```text
//! magic          8 bytes, MAGIC
//! format         u32, FORMAT_VERSION
//! metadata_len   u32
//! metadata       metadata_len bytes, opaque JSON, only read by native tools
//! namespace_cnt  u32
//! namespaces     namespace_cnt × (u32 len + UTF-8 bytes), sorted
//! entry_cnt      u32
//! entries        entry_cnt × 20 bytes, sorted by (namespace, key)
//! data           keys and values, concatenated
//! ```
//!
//! Both tables are sorted, so a lookup is a binary search over `entries` with
//! no allocation and no deserialization: the format is read in place rather
//! than parsed into memory. Everything is validated once at
//! [`open`](EmbeddedBundle::open), so lookups need no bounds handling beyond
//! ordinary slicing.

use alloc::string::String;
use alloc::vec::Vec;

use super::Bundle;
use crate::bytes::Bytes;
use crate::persistence::NamespaceSummary;

/// Identifies a flat bundle, and catches a file handed over by mistake.
pub const MAGIC: &[u8; 8] = b"CUBECLB\x01";

/// The flat layout this build reads and writes. Entries of any other version
/// are ignored rather than misread.
pub const FORMAT_VERSION: u32 = 1;

/// Size of one entry in the index: namespace id, key span, value span.
const ENTRY_SIZE: usize = 20;

/// Why a byte blob isn't a readable flat bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddedBundleError {
    /// The blob doesn't start with [`MAGIC`].
    NotABundle,
    /// The blob declares a layout this build doesn't read.
    UnsupportedFormat(u32),
    /// The blob is truncated or its offsets point outside it.
    Corrupted(&'static str),
}

impl core::fmt::Display for EmbeddedBundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotABundle => write!(f, "not a cubecl flat bundle"),
            Self::UnsupportedFormat(found) => write!(
                f,
                "flat bundle format {found} is not {FORMAT_VERSION}, its entries are unreadable"
            ),
            Self::Corrupted(what) => write!(f, "corrupted flat bundle: {what}"),
        }
    }
}

impl core::error::Error for EmbeddedBundleError {}

/// A bundle held as one blob of bytes, embedded in the binary with
/// `include_bytes!` or fetched at runtime.
///
/// This is the format for targets without a file system. The blob is held as
/// shared [`Bytes`], so a lookup hands back a zero-copy window into it rather
/// than a copy: serving a compiled kernel costs a reference count, whatever
/// its size.
///
/// ```ignore
/// static BUNDLE: &[u8] = include_bytes!("../bundles/h100.ccb");
///
/// let bundle = EmbeddedBundle::from_static(BUNDLE).expect("valid bundle");
/// cubecl_environment::bundle::install(Arc::new(bundle));
/// ```
#[derive(Debug)]
pub struct EmbeddedBundle {
    bytes: Bytes,
    /// Byte range of the metadata blob.
    metadata: (usize, usize),
    /// Offset of the namespace table and how many entries it holds.
    namespaces: (usize, usize),
    /// Offset of the entry index and how many entries it holds.
    entries: (usize, usize),
    /// Offset of the data section, which every span is relative to.
    data: usize,
}

impl EmbeddedBundle {
    /// Reads a flat bundle, validating it once so that later lookups can't
    /// fail.
    ///
    /// The blob is shared on the way in so that lookups can return zero-copy
    /// windows into it.
    pub fn open(bytes: Bytes) -> Result<Self, EmbeddedBundleError> {
        Self::parse(bytes.shared())
    }

    /// Reads a flat bundle embedded in the binary with `include_bytes!`.
    ///
    /// With the `shared-bytes` feature the static blob is adopted as is, so
    /// the bundle occupies no heap at all. Without it, the bytes are copied
    /// once at startup.
    pub fn from_static(blob: &'static [u8]) -> Result<Self, EmbeddedBundleError> {
        #[cfg(feature = "shared-bytes")]
        let bytes = Bytes::from_shared(
            bytes::Bytes::from_static(blob),
            crate::bytes::AllocationProperty::Other,
        );
        #[cfg(not(feature = "shared-bytes"))]
        let bytes = Bytes::from_bytes_vec(blob.to_vec());

        Self::open(bytes)
    }

    fn parse(bytes: Bytes) -> Result<Self, EmbeddedBundleError> {
        use EmbeddedBundleError::{Corrupted, NotABundle, UnsupportedFormat};

        if bytes.len() < MAGIC.len() + 8 || &bytes[..MAGIC.len()] != MAGIC {
            return Err(NotABundle);
        }

        let mut cursor = MAGIC.len();
        let take_u32 = |cursor: &mut usize| -> Option<u32> {
            let value = read_u32(&bytes, *cursor)?;
            *cursor += 4;
            Some(value)
        };

        let format = take_u32(&mut cursor).ok_or(Corrupted("truncated header"))?;
        if format != FORMAT_VERSION {
            return Err(UnsupportedFormat(format));
        }

        let metadata_len = take_u32(&mut cursor).ok_or(Corrupted("truncated header"))? as usize;
        let metadata_start = cursor;
        let metadata_end = metadata_start
            .checked_add(metadata_len)
            .ok_or(Corrupted("metadata length overflows"))?;
        if metadata_end > bytes.len() {
            return Err(Corrupted("metadata runs past the end"));
        }
        cursor = metadata_end;

        let namespace_count = take_u32(&mut cursor).ok_or(Corrupted("truncated header"))? as usize;
        let namespaces_start = cursor;
        // Walk the namespace table to find where it ends, validating each
        // length as we go.
        for _ in 0..namespace_count {
            let len =
                read_u32(&bytes, cursor).ok_or(Corrupted("truncated namespace table"))? as usize;
            cursor = cursor
                .checked_add(4)
                .and_then(|cursor| cursor.checked_add(len))
                .ok_or(Corrupted("namespace length overflows"))?;
            if cursor > bytes.len() {
                return Err(Corrupted("namespace table runs past the end"));
            }
        }

        let entry_count = read_u32(&bytes, cursor).ok_or(Corrupted("truncated header"))? as usize;
        cursor += 4;
        let entries_start = cursor;
        let index_len = entry_count
            .checked_mul(ENTRY_SIZE)
            .ok_or(Corrupted("entry count overflows"))?;
        let data_start = entries_start
            .checked_add(index_len)
            .ok_or(Corrupted("entry index overflows"))?;
        if data_start > bytes.len() {
            return Err(Corrupted("entry index runs past the end"));
        }

        let this = Self {
            bytes,
            metadata: (metadata_start, metadata_end),
            namespaces: (namespaces_start, namespace_count),
            entries: (entries_start, entry_count),
            data: data_start,
        };

        this.validate_entries()?;

        Ok(this)
    }

    /// Checks every span up front, so lookups never have to.
    fn validate_entries(&self) -> Result<(), EmbeddedBundleError> {
        use EmbeddedBundleError::Corrupted;

        let available = self.bytes.len() - self.data;
        let namespace_count = self.namespaces.1;

        for index in 0..self.entries.1 {
            let entry = self
                .entry(index)
                .ok_or(Corrupted("truncated entry index"))?;

            if entry.namespace as usize >= namespace_count {
                return Err(Corrupted("entry points at an unknown namespace"));
            }
            for (offset, len) in [
                (entry.key_offset, entry.key_len),
                (entry.value_offset, entry.value_len),
            ] {
                let end = (offset as usize)
                    .checked_add(len as usize)
                    .ok_or(Corrupted("entry span overflows"))?;
                if end > available {
                    return Err(Corrupted("entry span runs past the end"));
                }
            }
        }

        // The namespace table must be readable as UTF-8 to be comparable.
        for index in 0..namespace_count {
            self.namespace(index)
                .ok_or(Corrupted("namespace is not valid UTF-8"))?;
        }

        Ok(())
    }

    /// The opaque metadata blob, a JSON manifest when written by
    /// [`export`](super::export).
    pub fn metadata(&self) -> &[u8] {
        &self.bytes[self.metadata.0..self.metadata.1]
    }

    /// How many entries the bundle holds, across all namespaces.
    pub fn len(&self) -> usize {
        self.entries.1
    }

    /// Whether the bundle holds no entries at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entry count and total size per namespace, for reporting.
    pub fn summary(&self) -> Vec<NamespaceSummary> {
        let mut summary: Vec<NamespaceSummary> = (0..self.namespaces.1)
            .filter_map(|index| {
                Some(NamespaceSummary {
                    namespace: alloc::string::ToString::to_string(self.namespace(index)?),
                    entries: 0,
                    bytes: 0,
                })
            })
            .collect();

        for index in 0..self.entries.1 {
            let Some(entry) = self.entry(index) else {
                break;
            };
            if let Some(namespace) = summary.get_mut(entry.namespace as usize) {
                namespace.entries += 1;
                namespace.bytes += u64::from(entry.key_len) + u64::from(entry.value_len);
            }
        }

        summary
    }

    /// The `index`-th namespace of the sorted namespace table.
    fn namespace(&self, index: usize) -> Option<&str> {
        let mut cursor = self.namespaces.0;
        for _ in 0..index {
            let len = read_u32(&self.bytes, cursor)? as usize;
            cursor += 4 + len;
        }
        let len = read_u32(&self.bytes, cursor)? as usize;
        let start = cursor + 4;
        core::str::from_utf8(self.bytes.get(start..start + len)?).ok()
    }

    /// The id of `namespace`, by binary search over the sorted table.
    fn namespace_id(&self, namespace: &str) -> Option<u32> {
        let (mut low, mut high) = (0usize, self.namespaces.1);

        while low < high {
            let mid = low + (high - low) / 2;
            match self.namespace(mid)?.cmp(namespace) {
                core::cmp::Ordering::Less => low = mid + 1,
                core::cmp::Ordering::Greater => high = mid,
                core::cmp::Ordering::Equal => return Some(mid as u32),
            }
        }

        None
    }

    fn entry(&self, index: usize) -> Option<Entry> {
        let at = self.entries.0 + index * ENTRY_SIZE;

        Some(Entry {
            namespace: read_u32(&self.bytes, at)?,
            key_offset: read_u32(&self.bytes, at + 4)?,
            key_len: read_u32(&self.bytes, at + 8)?,
            value_offset: read_u32(&self.bytes, at + 12)?,
            value_len: read_u32(&self.bytes, at + 16)?,
        })
    }

    fn key_of(&self, entry: &Entry) -> &[u8] {
        let start = self.data + entry.key_offset as usize;
        &self.bytes[start..start + entry.key_len as usize]
    }

    fn value_of(&self, entry: &Entry) -> &[u8] {
        let start = self.data + entry.value_offset as usize;
        &self.bytes[start..start + entry.value_len as usize]
    }

    /// The entry's value as a zero-copy window into the blob.
    fn value_window(&self, entry: &Entry) -> Option<Bytes> {
        let start = self.data + entry.value_offset as usize;
        self.bytes
            .view(start, start + entry.value_len as usize)
            .inspect_err(|err| log::warn!("Embedded bundle: can't view an entry: {err:?}"))
            .ok()
    }

    /// The index of the first entry of `namespace`, if it has any.
    fn first_of(&self, namespace: u32) -> Option<usize> {
        let (mut low, mut high) = (0usize, self.entries.1);

        while low < high {
            let mid = low + (high - low) / 2;
            if self.entry(mid)?.namespace < namespace {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        (low < self.entries.1 && self.entry(low)?.namespace == namespace).then_some(low)
    }
}

impl Bundle for EmbeddedBundle {
    fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes> {
        let namespace = self.namespace_id(namespace)?;

        // The index is sorted by (namespace, key), so one binary search over
        // the pair finds the entry without materializing anything.
        let (mut low, mut high) = (0usize, self.entries.1);
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.entry(mid)?;

            match (entry.namespace, self.key_of(&entry)).cmp(&(namespace, key)) {
                core::cmp::Ordering::Less => low = mid + 1,
                core::cmp::Ordering::Greater => high = mid,
                core::cmp::Ordering::Equal => return self.value_window(&entry),
            }
        }

        None
    }

    fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8])) {
        let Some(id) = self.namespace_id(namespace) else {
            return;
        };
        let Some(first) = self.first_of(id) else {
            return;
        };

        for index in first..self.entries.1 {
            let Some(entry) = self.entry(index) else {
                return;
            };
            if entry.namespace != id {
                return;
            }
            visit(self.key_of(&entry), self.value_of(&entry));
        }
    }

    fn describe(&self) -> String {
        alloc::format!("embedded bundle ({} entries)", self.entries.1)
    }
}

/// One row of the entry index.
struct Entry {
    namespace: u32,
    key_offset: u32,
    key_len: u32,
    value_offset: u32,
    value_len: u32,
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Bytes that pass `open` but describe nothing.
    fn empty_bundle() -> Bytes {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // metadata
        bytes.extend_from_slice(&0u32.to_le_bytes()); // namespaces
        bytes.extend_from_slice(&0u32.to_le_bytes()); // entries
        Bytes::from_bytes_vec(bytes)
    }

    #[test]
    fn an_empty_bundle_reads_as_empty() {
        let bundle = EmbeddedBundle::open(empty_bundle()).unwrap();

        assert!(bundle.is_empty());
        assert_eq!(bundle.get("anything", b"key"), None);
        bundle.scan("anything", &mut |_, _| panic!("no entries to visit"));
    }

    #[test]
    fn foreign_bytes_are_rejected() {
        assert_eq!(
            EmbeddedBundle::open(Bytes::from_bytes_vec(b"definitely not a bundle".to_vec()))
                .unwrap_err(),
            EmbeddedBundleError::NotABundle
        );
        assert_eq!(
            EmbeddedBundle::open(Bytes::from_bytes_vec(vec![])).unwrap_err(),
            EmbeddedBundleError::NotABundle
        );
    }

    #[test]
    fn another_format_version_is_rejected() {
        let mut bytes = empty_bundle().to_vec();
        bytes[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&99u32.to_le_bytes());
        let bytes = Bytes::from_bytes_vec(bytes);

        assert_eq!(
            EmbeddedBundle::open(bytes).unwrap_err(),
            EmbeddedBundleError::UnsupportedFormat(99)
        );
    }

    /// Truncation must be reported, never panic on a slice out of range.
    #[test]
    fn truncated_bytes_are_rejected() {
        let full = empty_bundle();

        for len in MAGIC.len()..full.len() {
            let result = EmbeddedBundle::open(Bytes::from_bytes_vec(full[..len].to_vec()));
            assert!(result.is_err(), "a {len}-byte prefix must not open");
        }
    }

    /// A blob whose offsets point outside it must be caught at open, since
    /// lookups slice without checking.
    #[test]
    fn out_of_range_spans_are_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // metadata
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one namespace
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(b"ns");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one entry
        bytes.extend_from_slice(&0u32.to_le_bytes()); // namespace id
        bytes.extend_from_slice(&0u32.to_le_bytes()); // key offset
        bytes.extend_from_slice(&99u32.to_le_bytes()); // key len, way past the end
        bytes.extend_from_slice(&0u32.to_le_bytes()); // value offset
        bytes.extend_from_slice(&0u32.to_le_bytes()); // value len

        assert!(matches!(
            EmbeddedBundle::open(Bytes::from_bytes_vec(bytes)).unwrap_err(),
            EmbeddedBundleError::Corrupted(_)
        ));
    }

    #[test]
    fn an_unknown_namespace_id_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // no namespaces
        bytes.extend_from_slice(&1u32.to_le_bytes()); // but one entry
        bytes.extend_from_slice(&7u32.to_le_bytes()); // pointing at namespace 7
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        assert!(matches!(
            EmbeddedBundle::open(Bytes::from_bytes_vec(bytes)).unwrap_err(),
            EmbeddedBundleError::Corrupted(_)
        ));
    }
}
