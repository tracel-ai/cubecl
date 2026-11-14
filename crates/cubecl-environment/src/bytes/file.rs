use super::{AllocationProperty, file_reads};
use crate::bytes::{
    AccessError, AccessPolicy, AllocationController,
    default_controller::{MAX_ALIGN, NativeAllocationController},
};
use spin::Once;
use std::{
    boxed::Box,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    string::ToString,
    sync::{Arc, Mutex, PoisonError},
    vec,
};

/// Where bytes read lazily from a file lie in it: the file they are read
/// from, and the offset of their first byte.
///
/// Public only to appear in [`AllocationController::file`]; the module is
/// private, so nothing outside the crate names it, and its fields are the
/// crate's.
// Read only by the bundle readers, which the native cache builds.
#[cfg_attr(not(native_cache), allow(dead_code))]
pub struct FileWindow<'a> {
    pub(crate) source: &'a FileSource,
    pub(crate) offset: u64,
}

// Read only by the bundle readers, which the native cache builds.
#[cfg_attr(not(native_cache), allow(dead_code))]
impl<'a> FileWindow<'a> {
    /// The file to open in place, when the window is the whole of the one at
    /// its path, `len` bytes long, and that file is the one the bytes are
    /// read from. A file opened and held proves it is still the one at its
    /// path only where the platform says which file a path names (Unix);
    /// elsewhere it answers `None`, and the caller reads the bytes instead.
    pub(crate) fn whole_file(&self, len: u64) -> Option<&'a Path> {
        if self.offset != 0 {
            return None;
        }
        let at_path = std::fs::metadata(self.source.path()).ok()?;
        let whole = match self.source {
            FileSource::Path(_) => at_path.len() == len,
            FileSource::Opened { file, .. } => {
                let held = file
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .metadata()
                    .ok()?;
                held.len() == len && same_file(&held, &at_path)
            }
        };
        whole.then(|| self.source.path())
    }
}

/// What a file allocation reads its bytes from.
// Read only by the bundle readers, which the native cache builds.
#[cfg_attr(not(native_cache), allow(dead_code))]
pub(crate) enum FileSource {
    /// The file at a path, opened at each read: whatever file is there then.
    Path(PathBuf),
    /// A file opened once and held, beside the path it was opened at: a file
    /// later renamed over the path, or the path removed, leaves the bytes
    /// read from this one.
    Opened {
        /// Where the file was opened, which the reads are counted under.
        path: PathBuf,
        /// Locked for each read, since a read seeks it.
        file: Mutex<File>,
    },
}

impl FileSource {
    /// The path the file was given at.
    fn path(&self) -> &Path {
        match self {
            Self::Path(path) | Self::Opened { path, .. } => path,
        }
    }

    /// Fills `buf` from the file, starting `offset` bytes in.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        match self {
            Self::Path(path) => read_at(&mut File::open(path)?, offset, buf),
            Self::Opened { file, .. } => read_at(
                &mut file.lock().unwrap_or_else(PoisonError::into_inner),
                offset,
                buf,
            ),
        }
    }
}

/// The allocation is managed on a file.
///
/// The content of the file is copied into an in-memory buffer (a [`NativeAllocationController`])
/// lazily on first access, behind a [`Once`] so concurrent first accesses materialize exactly
/// once. Duplicating the file allocator gives every version its own buffer.
///
/// # Notes
///
/// Because of that mechanism, dereferencing [`crate::bytes::Bytes`] isn't cost-free when using the
/// file allocator, since it's going to trigger a copy from the file system to an in-memory buffer.
pub(crate) struct FileAllocationController {
    source: Arc<FileSource>,
    size: u64,
    offset: u64,
    controller: Once<Box<dyn AllocationController>>,
}

impl FileAllocationController {
    pub fn new<P: Into<PathBuf>>(file: P, size: u64, offset: u64) -> Self {
        Self::from_source(Arc::new(FileSource::Path(file.into())), size, offset)
    }

    /// The `size` bytes of `file`, opened at `path`, from its start; read
    /// from the open file, whatever is at `path` by then.
    // Read only by the bundle readers, which the native cache builds.
    #[cfg_attr(not(native_cache), allow(dead_code))]
    pub fn opened(file: File, path: PathBuf, size: u64) -> Self {
        let source = FileSource::Opened {
            path,
            file: Mutex::new(file),
        };
        Self::from_source(Arc::new(source), size, 0)
    }

    fn from_source(source: Arc<FileSource>, size: u64, offset: u64) -> Self {
        Self {
            source,
            size,
            controller: Once::new(),
            offset,
        }
    }

    /// Materialize the file content into an in-memory buffer on first call, returning the cached
    /// controller afterwards.
    ///
    /// Honors `policy`: if not yet materialized and the policy forbids copies, returns
    /// [`AccessError::WouldCopy`]. A failed read returns [`AccessError::Read`] and leaves the
    /// cell uninitialized (retryable).
    fn ensure_init(&self, policy: AccessPolicy) -> Result<&dyn AllocationController, AccessError> {
        if let Some(controller) = self.controller.get() {
            return Ok(&**controller);
        }
        if !policy.copy_allowed() {
            return Err(AccessError::WouldCopy);
        }

        let controller = self.controller.try_call_once(
            || -> Result<Box<dyn AllocationController>, AccessError> {
                let mut buf = vec![0u8; self.size as usize];
                self.source
                    .read_at(self.offset, &mut buf)
                    .map_err(|e| AccessError::Read(e.to_string()))?;
                file_reads::count(self.source.path(), self.size);

                Ok(Box::new(NativeAllocationController::from_elems(buf)))
            },
        )?;
        Ok(&**controller)
    }
}

impl AllocationController for FileAllocationController {
    fn alloc_align(&self) -> usize {
        MAX_ALIGN
    }

    fn split(
        &mut self,
        offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), super::SplitError>
    {
        // Use `<` (not `<=`) to allow boundary splits where one side is empty.
        // This is symmetric: both offset==0 (empty left) and offset==size (empty right) are valid.
        if self.size < offset as u64 {
            return Err(super::SplitError::InvalidOffset);
        }

        let left =
            FileAllocationController::from_source(self.source.clone(), offset as u64, self.offset);
        let right = FileAllocationController::from_source(
            self.source.clone(),
            self.size - offset as u64,
            self.offset + offset as u64,
        );

        Ok((Box::new(left), Box::new(right)))
    }

    fn view(&self, start: usize, end: usize) -> Option<Box<dyn AllocationController>> {
        if self.controller.is_completed() {
            // Once materialized, the in-memory buffer may diverge from the file
            // (after a copy-on-write), so no zero-copy file window is offered.
            // The caller can `Bytes::shared()` the current data to view it instead.
            return None;
        }
        if start > end || end as u64 > self.size {
            return None;
        }

        Some(Box::new(FileAllocationController::from_source(
            self.source.clone(),
            (end - start) as u64,
            self.offset + start as u64,
        )))
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::File
    }

    fn may_alias_mutable_storage(&self) -> bool {
        // File contents are copied into a private allocation, never mapped.
        false
    }

    // The file region size, known without reading the file (never materializes).
    fn capacity(&self) -> usize {
        self.size as usize
    }

    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [core::mem::MaybeUninit<u8>], AccessError> {
        self.ensure_init(policy)?;

        // SAFETY: `ensure_init` guarantees the controller is set, and `&mut self` is exclusive.
        let controller = self
            .controller
            .get_mut()
            .expect("controller must be set after init");
        unsafe { controller.memory_mut(policy) }
    }

    fn memory(&self, policy: AccessPolicy) -> Result<&[core::mem::MaybeUninit<u8>], AccessError> {
        self.ensure_init(policy)?.memory(policy)
    }

    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        if self.controller.is_completed() {
            return None;
        }

        let controller = Self::from_source(self.source.clone(), self.size, self.offset);
        Some(Box::new(controller))
    }

    fn file(&self) -> Option<FileWindow<'_>> {
        (!self.controller.is_completed()).then_some(FileWindow {
            source: &self.source,
            offset: self.offset,
        })
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        if self.controller.is_completed() {
            let len = buf.len();
            let memory = self
                .memory(AccessPolicy::default())
                .expect("file: host access failed");
            let memory_slice = &memory[0..len];

            // SAFETY: By construction, bytes up to len are initialized.
            let data = unsafe {
                core::slice::from_raw_parts(memory_slice.as_ptr().cast(), memory_slice.len())
            };
            buf.copy_from_slice(data);
            return;
        }

        self.source
            .read_at(self.offset, buf)
            .expect("file: host access failed");
        file_reads::count(self.source.path(), buf.len() as u64);
    }
}

/// Fills `buf` from `file`, starting `offset` bytes in.
fn read_at(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// Whether two files' metadata are of one file.
#[cfg(unix)]
// Read only by the bundle readers, which the native cache builds.
#[cfg_attr(not(native_cache), allow(dead_code))]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    (a.dev(), a.ino()) == (b.dev(), b.ino())
}

/// A platform whose std does not say which file a path names: never proven
/// the same.
#[cfg(not(unix))]
// Read only by the bundle readers, which the native cache builds.
#[cfg_attr(not(native_cache), allow(dead_code))]
fn same_file(_: &std::fs::Metadata, _: &std::fs::Metadata) -> bool {
    false
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use tempfile::TempDir;

    use super::super::{AccessError, Bytes, Reader, SplitPolicy};
    use std::{io::Write, path::PathBuf, vec::Vec};

    #[test_log::test]
    fn test_no_copy_read_and_length_do_not_materialize() {
        let elems = (0..120).collect();
        let (path, bytes_expected, dir) = with_data(elems);
        let bytes = Bytes::from_file(&path, bytes_expected.len() as u64, 0);

        // Not materialized yet: a no-copy read must refuse rather than read the file.
        assert_eq!(
            bytes.read(Reader::new().no_copy()),
            Err(AccessError::WouldCopy)
        );

        // `len()` and `capacity()` must NOT materialize, so a no-copy read still refuses.
        assert_eq!(bytes.len(), bytes_expected.len());
        assert_eq!(bytes.capacity(), bytes_expected.len());
        assert_eq!(
            bytes.read(Reader::new().no_copy()),
            Err(AccessError::WouldCopy)
        );

        // A copy-allowed read materializes the file...
        assert_eq!(bytes.read(Reader::new()).unwrap(), &bytes_expected[..]);
        // ...after which a no-copy read succeeds.
        assert_eq!(
            bytes.read(Reader::new().no_copy()).unwrap(),
            &bytes_expected[..]
        );
        core::mem::drop(dir);
    }

    #[test_log::test]
    fn test_from_file() {
        let elems = (0..250).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);

        assert_eq!(&bytes, &bytes_file);
        core::mem::drop(dir);
    }

    #[test_log::test]
    fn test_split_file() {
        let elems = (0..250).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        let offset = 40;
        let (left, right) = bytes_file.split(offset, SplitPolicy::Shared).unwrap();

        let left_expected: &[u8] = &bytes[0..offset];
        let right_expected: &[u8] = &bytes[offset..];
        let left_actual: &[u8] = &left;
        let right_actual: &[u8] = &right;

        assert_eq!(left_expected, left_actual);
        assert_eq!(right_expected, right_actual);
        core::mem::drop(dir);
    }

    #[test_log::test]
    fn test_split_file_at_zero() {
        // Boundary case: split at 0 creates empty left, full right
        let elems: Vec<u8> = (0..100).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        let (left, right) = bytes_file.split(0, SplitPolicy::Shared).unwrap();

        assert_eq!(left.len(), 0);
        assert_eq!(&right[..], &bytes[..]);
        core::mem::drop(dir);
    }

    #[test_log::test]
    fn test_view_file_is_zero_copy() {
        let elems: Vec<u8> = (0..100).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        // A file-backed view reads the corresponding sub-range straight from
        // the file, no full load required, and leaves the original usable.
        let view = bytes_file.view(10, 20).unwrap();

        assert_eq!(&view[..], &bytes[10..20]);
        assert_eq!(&bytes_file[..], &bytes[..]);
        core::mem::drop(dir);
    }

    #[test_log::test]
    fn test_split_file_at_len() {
        // Boundary case: split at len creates full left, empty right
        let elems: Vec<u8> = (0..100).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        let len = bytes_file.len();
        let (left, right) = bytes_file.split(len, SplitPolicy::Shared).unwrap();

        assert_eq!(&left[..], &bytes[..]);
        assert_eq!(right.len(), 0);
        core::mem::drop(dir);
    }

    #[test_log::test]
    fn test_memory_mut_on_duplicated_file() {
        let elems = (0..250).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);

        let mut bytes_mut = bytes_file.clone();
        bytes_mut[0] = 5;

        let expected: &[u8] = &bytes[1..];
        let actual: &[u8] = &bytes_mut[1..];

        assert_eq!(&bytes, &bytes_file);
        assert_eq!(bytes_mut[0], 5);
        assert_eq!(expected, actual);
        core::mem::drop(dir);
    }

    fn with_data(elems: Vec<u8>) -> (PathBuf, Bytes, TempDir) {
        let bytes = Bytes::from_bytes_vec(elems);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        (path, bytes, dir)
    }
}
