use std::fs::File;

/// A file held open and read at positions: reads of it on several threads
/// at once each fill their own buffer from their own offset.
///
/// Public only to appear in [`FileSource`](super::file::FileSource); the
/// module is private, so nothing outside the crate names it.
#[derive(Debug)]
pub struct HeldFile {
    /// Read with positioned reads, which move no position shared by the
    /// readers.
    #[cfg(any(unix, windows))]
    file: File,
    /// Locked for each read, since a read on a platform without positioned
    /// reads seeks it.
    #[cfg(not(any(unix, windows)))]
    file: std::sync::Mutex<File>,
}

impl HeldFile {
    /// Hold `file` to read it at positions.
    pub(crate) fn new(file: File) -> Self {
        Self {
            #[cfg(any(unix, windows))]
            file,
            #[cfg(not(any(unix, windows)))]
            file: std::sync::Mutex::new(file),
        }
    }

    /// Fills `buf` from the file, starting `offset` bytes in.
    ///
    /// # Errors
    ///
    /// The file does not read, or ends before `buf` is full.
    pub(crate) fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        #[cfg(any(unix, windows))]
        return read_exact_at(&self.file, offset, buf);
        #[cfg(not(any(unix, windows)))]
        return read_exact_at(
            &mut self
                .file
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            offset,
            buf,
        );
    }

    /// The held file's metadata.
    // Read only by the bundle readers, which the native cache builds.
    #[cfg_attr(not(native_cache), allow(dead_code))]
    pub(crate) fn metadata(&self) -> std::io::Result<std::fs::Metadata> {
        #[cfg(any(unix, windows))]
        return self.file.metadata();
        #[cfg(not(any(unix, windows)))]
        return self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .metadata();
    }
}

/// Fills `buf` from `file`, starting `offset` bytes in, without moving a
/// position other readers of `file` share.
#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// Fills `buf` from `file`, starting `offset` bytes in, each read naming
/// its own offset. A positioned read on Windows still moves the file's
/// position, which no read here relies on.
#[cfg(windows)]
fn read_exact_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                buf = &mut buf[read..];
                offset += read as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Fills `buf` from `file`, starting `offset` bytes in, by seeking it.
#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}
