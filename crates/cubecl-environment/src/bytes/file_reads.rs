//! The bytes file-backed [`Bytes`](super::Bytes) read from their files, for a
//! caller that shows how far a lazy load has got.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    vec::Vec,
};

/// Every counter alive, matched against the file of each read.
static COUNTERS: Mutex<Vec<Weak<Counted>>> = Mutex::new(Vec::new());

/// How many counters are alive: a read with none counts nothing, and skips
/// the lock every other read of a file would take.
static ALIVE: AtomicUsize = AtomicUsize::new(0);

/// The bytes read from one file through file-backed
/// [`Bytes`](super::Bytes) while this counter is alive.
///
/// A file-backed allocation reads its file only when it is first accessed —
/// most often when a runtime copies it to a device, on the runtime's own
/// thread — so whoever made the allocation does not see the read happen. A
/// counter is how they learn how far it has got: create one for the file
/// before its bytes are read, then read [`read`](Self::read) from any thread.
///
/// The file is matched by the path [`Bytes::from_file`](super::Bytes::from_file)
/// was given, compared as given and never canonicalized. Two counters of one
/// file each count every read of it.
pub struct FileReadCounter {
    counted: Arc<Counted>,
}

struct Counted {
    path: PathBuf,
    read: AtomicU64,
}

impl FileReadCounter {
    /// Start counting the reads of `path`, from zero.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let counted = Arc::new(Counted {
            path: path.into(),
            read: AtomicU64::new(0),
        });
        let mut counters = COUNTERS.lock().unwrap_or_else(|poison| poison.into_inner());
        counters.retain(|counter| counter.strong_count() > 0);
        counters.push(Arc::downgrade(&counted));
        ALIVE.fetch_add(1, Ordering::Release);
        Self { counted }
    }

    /// The file counted.
    pub fn path(&self) -> &Path {
        &self.counted.path
    }

    /// The bytes read from the file since this counter was created.
    pub fn read(&self) -> u64 {
        self.counted.read.load(Ordering::Relaxed)
    }
}

impl Drop for FileReadCounter {
    fn drop(&mut self) {
        ALIVE.fetch_sub(1, Ordering::Release);
    }
}

/// Count `size` bytes read from `path` on every counter of it.
pub(crate) fn count(path: &Path, size: u64) {
    if ALIVE.load(Ordering::Acquire) == 0 {
        return;
    }
    let counters = COUNTERS.lock().unwrap_or_else(|poison| poison.into_inner());
    counters
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|counted| counted.path == path)
        .for_each(|counted| {
            counted.read.fetch_add(size, Ordering::Relaxed);
        });
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::super::{Bytes, Reader};
    use super::*;
    use std::io::Write;

    /// A read is counted once, when the file is first accessed, whether it
    /// materializes the bytes or copies them out, and a file nobody counts
    /// is read as before.
    #[test]
    fn reads_are_counted_when_the_file_is_accessed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&[7u8; 64])
            .unwrap();

        let counter = FileReadCounter::new(&path);
        let bytes = Bytes::from_file(&path, 64, 0);
        let view = bytes.view(0, 16).unwrap();
        assert_eq!(counter.read(), 0);

        let mut copied = Bytes::from_bytes_vec(std::vec![0u8; 16]);
        view.copy_into(&mut copied);
        assert_eq!(counter.read(), 16);

        bytes.read(Reader::new()).unwrap();
        bytes.read(Reader::new()).unwrap();
        assert_eq!(counter.read(), 16 + 64);

        let other = dir.path().join("other");
        std::fs::File::create(&other)
            .unwrap()
            .write_all(&[1])
            .unwrap();
        Bytes::from_file(&other, 1, 0).read(Reader::new()).unwrap();
        assert_eq!(counter.read(), 16 + 64);
    }
}
