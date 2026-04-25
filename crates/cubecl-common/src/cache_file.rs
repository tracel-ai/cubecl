use core::{fmt::Display, time::Duration};
use std::{
    format,
    fs::{self, File},
    io::{BufReader, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    string::String,
};

/// Multi-process safe append-only file .
#[derive(Debug)]
pub struct CacheFile {
    path: PathBuf,
    lock: FileLock,
    cursor: u64,
    /// `false` when the underlying file could not be created / opened
    /// (missing parent directory, read-only mount, sandbox / hardened
    /// runtime refusal, etc.). All subsequent ops on an invalid cache
    /// file are no-ops — callers see an empty cache and fall through
    /// to recompute-from-scratch paths rather than panicking.
    ///
    /// This replaces the previous panic-on-unwrap behavior that
    /// cascaded into "Task failed: Any { .. }" warnings across burn's
    /// fusion runtime when a downstream IR operation tried to use a
    /// tensor whose producing kernel never got registered because the
    /// cache file couldn't be opened. See the `fix/fastdivmod-zero-guard`
    /// branch commit log for the motivating bug report.
    valid: bool,
}

impl Display for CacheFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Cache file: {:?}", self.path)
    }
}

impl CacheFile {
    /// Create a new cache file.
    ///
    /// If the parent directory can't be created OR the file can't be
    /// opened (common failure mode in sandboxed / ad-hoc-signed bundle
    /// apps where `$HOME/.cache/<name>/<version>/` write access is
    /// silently denied), we log an error and return an "invalid"
    /// `CacheFile`. Subsequent `lock()` / `write()` calls become no-ops,
    /// so callers silently skip the disk cache and recompute
    /// every time. Slow, but never panics.
    pub fn new<P: Into<PathBuf>>(path: P, lock_max_duration: Duration) -> Self {
        let path: PathBuf = path.into();
        let mut valid = true;

        log::debug!("cubecl cache: CacheFile::new({:?})", path);

        // We check before trying to create the file, since it might erase the content of an
        // existing file.
        if !fs::exists(&path).unwrap_or(false) {
            if let Some(parent) = path.parent()
                && let Err(err) = fs::create_dir_all(parent)
            {
                log::error!(
                    "cubecl cache: create_dir_all({:?}) failed: kind={:?} raw_os_error={:?} err={}; \
                     cache will be disabled for this entry",
                    parent,
                    err.kind(),
                    err.raw_os_error(),
                    err
                );
                valid = false;
            }

            // Even if `create_dir_all` appeared to succeed, the actual
            // `File::create` can still fail (permission, read-only FS,
            // race with a cleanup process). Log and disable rather
            // than panicking — the original `.unwrap()` here is what
            // cascaded into 7-9 burn-fusion panics when the cache
            // subdir was unwritable.
            if valid && let Err(err) = File::create(&path) {
                let parent_exists = path.parent().map(|p| p.exists()).unwrap_or(false);
                log::error!(
                    "cubecl cache: File::create({:?}) failed: kind={:?} raw_os_error={:?} \
                     parent_exists={} err={}; cache will be disabled for this entry",
                    path,
                    err.kind(),
                    err.raw_os_error(),
                    parent_exists,
                    err
                );
                valid = false;
            }
        }

        Self {
            lock: FileLock::new(&path, lock_max_duration),
            path,
            cursor: 0,
            valid,
        }
    }

    /// Locks the file and returns the content that wasn't synced since the last lock.
    ///
    /// Returns `None` when the cache file is invalid OR when the
    /// underlying file cannot be opened (disappeared, permission
    /// changed, etc.). Callers already treat `None` as "no new
    /// content" so degrading to this path is safe.
    pub fn lock(&mut self) -> Option<BufReader<File>> {
        if !self.valid {
            return None;
        }

        if !self.lock.lock() {
            // FileLock gave up (lock file can't be created even after
            // waiting lock_max_duration — typically because the parent
            // dir is missing and create_new keeps returning NotFound,
            // which used to spin forever here).
            log::error!(
                "cubecl cache: FileLock::lock({:?}) gave up; disabling cache entry",
                self.path
            );
            self.valid = false;
            return None;
        }

        let mut file = match File::open(&self.path) {
            Ok(f) => f,
            Err(err) => {
                let path_exists = self.path.exists();
                let parent_exists = self.path.parent().map(|p| p.exists()).unwrap_or(false);
                log::error!(
                    "cubecl cache: File::open({:?}) failed during lock(): kind={:?} \
                     raw_os_error={:?} path_exists={} parent_exists={} err={}",
                    self.path,
                    err.kind(),
                    err.raw_os_error(),
                    path_exists,
                    parent_exists,
                    err
                );
                self.lock.unlock();
                self.valid = false;
                return None;
            }
        };
        let end = match file.metadata() {
            Ok(m) => m.len(),
            Err(err) => {
                log::error!(
                    "cubecl cache: metadata({:?}) failed: kind={:?} raw_os_error={:?} err={}",
                    self.path,
                    err.kind(),
                    err.raw_os_error(),
                    err
                );
                self.lock.unlock();
                self.valid = false;
                return None;
            }
        };
        if let Err(err) = file.seek(SeekFrom::Start(self.cursor)) {
            log::error!(
                "cubecl cache: seek({:?}, cursor={}) failed: kind={:?} raw_os_error={:?} err={}",
                self.path,
                self.cursor,
                err.kind(),
                err.raw_os_error(),
                err
            );
            self.lock.unlock();
            self.valid = false;
            return None;
        }

        if self.cursor < end {
            let buf = BufReader::new(file);
            self.cursor = end;
            Some(buf)
        } else {
            None
        }
    }

    /// Unlock the file.
    pub fn unlock(&mut self) {
        self.lock.unlock();
    }

    /// Write the content to the file.
    ///
    /// The `valid` check comes BEFORE the `is_lock` check: `lock()`'s
    /// error paths call `self.lock.unlock()` on their way out, which
    /// clears `is_lock`. If we checked `is_lock` first, every failed
    /// `lock()` would turn a subsequent `Cache::insert` write into
    /// a panic — exactly the cascade the "gracefully handle cache
    /// I/O failures" patch (commit a784b98) was meant to eliminate.
    /// A failed `lock()` sets `valid = false`, so we silently no-op.
    ///
    /// When the cache IS valid but `write()` is called with no lock
    /// held, that's a genuine caller-side API misuse and we panic.
    /// The root cause of the original cache-lock failure cascade
    /// (autotune cache wrote to `/target/...` on macOS bundles,
    /// read-only filesystem) was fixed in the `Target` fallback
    /// commit (ddf43b7f); the panic is the correct response to a
    /// caller that forgets to `lock()` first.
    pub fn write(&mut self, content: &[u8]) {
        if !self.valid {
            return;
        }
        if !self.lock.is_lock {
            panic!("The cache file should be locked before writing content to it.")
        }

        let mut file = match fs::OpenOptions::new().append(true).open(&self.path) {
            Ok(f) => f,
            Err(err) => {
                let path_exists = self.path.exists();
                log::error!(
                    "cubecl cache: append-open({:?}) failed: kind={:?} raw_os_error={:?} \
                     path_exists={} err={}",
                    self.path,
                    err.kind(),
                    err.raw_os_error(),
                    path_exists,
                    err
                );
                self.valid = false;
                return;
            }
        };

        match file.write(content) {
            Ok(n) => self.cursor += n as u64,
            Err(err) => {
                log::error!(
                    "cubecl cache: write({:?}, {} bytes) failed: kind={:?} raw_os_error={:?} err={}",
                    self.path,
                    content.len(),
                    err.kind(),
                    err.raw_os_error(),
                    err
                );
                self.valid = false;
            }
        }
    }
}

#[derive(Debug)]
/// A very simple file lock that only depends on std.
///
/// The lock is only valid for a fixed duration; after that, there is no guarantee.
/// This is to combat corrupted data, since killing a process might leave the lock file on disk.
///
/// Since it is used with an append-only cache file, we could simply delete the entire cache file
/// when the lock is outdated.
struct FileLock {
    is_lock: bool,
    path_lock: PathBuf,
    lock_max_duration: Duration,
}

impl FileLock {
    /// Create a lock for the given file path.
    pub fn new(path: &Path, lock_max_duration: Duration) -> Self {
        let file_name = path
            .file_name()
            .expect("Path to have a file name.")
            .to_str()
            .expect("File name to be valid");
        let mut path_lock = path.to_path_buf();
        path_lock.set_file_name(format!("{file_name}.lock"));

        Self {
            path_lock,
            is_lock: false,
            lock_max_duration,
        }
    }

    /// Acquire the lock.
    ///
    /// Returns `true` on success, `false` if we gave up waiting —
    /// either because a peer held the lock longer than
    /// `lock_max_duration`, or because `create_new` kept failing
    /// with a non-`AlreadyExists` error (`NotFound` when the parent
    /// directory is missing, `PermissionDenied` under sandboxing,
    /// etc.). The previous implementation spun forever in that
    /// second case; now we log and bail.
    pub fn lock(&mut self) -> bool {
        if self.is_lock {
            return true;
        }

        let waiting_total = std::time::SystemTime::now();
        let mut other_error_logged = false;

        loop {
            match fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&self.path_lock)
            {
                Ok(mut file) => {
                    let timestamp = std::time::SystemTime::now();
                    let content = serde_json::to_vec(&timestamp).unwrap();
                    file.write_all(&content).unwrap();
                    break;
                }
                Err(err) => match err.kind() {
                    std::io::ErrorKind::AlreadyExists => {
                        if let Ok(true) = self.maybe_cleanup_frozen_lock() {
                            log::debug!("Removed frozen lock file");
                        } else {
                            if waiting_total.elapsed().unwrap_or_default() > self.lock_max_duration
                            {
                                log::error!(
                                    "cubecl cache: FileLock::lock({:?}) timed out after {:?} \
                                     waiting for peer to release lock; giving up",
                                    self.path_lock,
                                    self.lock_max_duration
                                );
                                return false;
                            }
                            std::thread::sleep(Duration::from_millis(30));
                        }
                    }
                    _ => {
                        if !other_error_logged {
                            let parent_exists =
                                self.path_lock.parent().map(|p| p.exists()).unwrap_or(false);
                            log::error!(
                                "cubecl cache: FileLock::lock({:?}) unexpected I/O error: \
                                 kind={:?} raw_os_error={:?} parent_exists={} err={}",
                                self.path_lock,
                                err.kind(),
                                err.raw_os_error(),
                                parent_exists,
                                err
                            );
                            other_error_logged = true;
                        }
                        if waiting_total.elapsed().unwrap_or_default() > self.lock_max_duration {
                            fs::remove_file(&self.path_lock).ok();
                            log::error!(
                                "cubecl cache: FileLock::lock({:?}) giving up after {:?} \
                                 (persistent non-AlreadyExists error)",
                                self.path_lock,
                                self.lock_max_duration
                            );
                            return false;
                        }
                        std::thread::sleep(Duration::from_millis(30));
                    }
                },
            };
        }

        self.is_lock = true;
        true
    }

    pub fn unlock(&mut self) {
        if self.is_lock {
            fs::remove_file(&self.path_lock).ok();
        }

        self.is_lock = false;
    }

    fn maybe_cleanup_frozen_lock(&mut self) -> Result<bool, String> {
        let content = fs::read_to_string(&self.path_lock).map_err(|err| format!("{err}"))?;
        let timestamp: std::time::SystemTime =
            serde_json::from_str(&content).map_err(|err| format!("{err}"))?;

        let elapsed = timestamp.elapsed().map_err(|err| format!("{err}"))?;

        if elapsed > self.lock_max_duration {
            fs::remove_file(&self.path_lock).map_err(|err| format!("{err}"))?;
            return Ok(true);
        }

        Ok(false)
    }
}
