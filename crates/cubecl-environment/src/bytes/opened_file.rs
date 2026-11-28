use super::{file::FileSource, held_file::HeldFile};
use std::{fs::File, path::PathBuf, sync::Arc};

/// A file opened once, which any number of [`Bytes`](super::Bytes) read
/// windows of through [`Bytes::from_opened_file`](super::Bytes::from_opened_file).
///
/// Cloning it shares the one open file. The file stays open as long as the
/// opened file or any bytes made from it are alive. A file written over in
/// place is read as it is then: only a file renamed over the path, or the
/// path removed, leaves the bytes reading the file that was opened.
///
/// # Guarantees
///
/// - Every read is of the file held, never of whatever is at its path then.
/// - Bytes read on several threads at once each read their own window.
/// - The reads are counted under the path the file was opened at by a
///   [`FileReadCounter`](super::FileReadCounter) of it.
///
/// # Examples
///
/// ```no_run
/// use cubecl_environment::bytes::{Bytes, OpenedFile};
///
/// let path = "weights.bin";
/// let opened = OpenedFile::new(std::fs::File::open(path).unwrap(), path);
/// let first = Bytes::from_opened_file(&opened, 64, 0);
/// let second = Bytes::from_opened_file(&opened, 128, 64);
/// std::fs::remove_file(path).unwrap();
///
/// // Both still read the file that was opened.
/// let first: &[u8] = &first;
/// let second: &[u8] = &second;
/// assert_eq!(first.len() + second.len(), 192);
/// ```
#[derive(Debug, Clone)]
pub struct OpenedFile {
    source: Arc<FileSource>,
}

impl OpenedFile {
    /// Hold `file` to read windows of it.
    ///
    /// # Assumptions
    ///
    /// - `file` was opened for reading at `path`.
    pub fn new(file: File, path: impl Into<PathBuf>) -> Self {
        let source = FileSource::Opened {
            path: path.into(),
            file: HeldFile::new(file),
        };
        Self {
            source: Arc::new(source),
        }
    }
}

impl OpenedFile {
    /// The source the windows of the file read from.
    pub(crate) fn source(&self) -> &Arc<FileSource> {
        &self.source
    }
}
