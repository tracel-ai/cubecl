use core::{fmt::Display, time::Duration};
use std::{
    fs::{self, File},
    io::{BufReader, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

/// Multi-process safe append-only file .
#[derive(Debug)]
pub struct CacheFile {
    path: PathBuf,
    lock: FileLock,
    cursor: u64,
}

impl Display for CacheFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Cache file: {:?}", self.path)
    }
}

impl CacheFile {
    /// Create a new cache file.
    pub fn new<P: Into<PathBuf>>(path: P, lock_max_duration: Duration) -> Self {
        let path: PathBuf = path.into();

        // We check before trying to create the file, since it might erase the content of an
        // existing file.
        if !fs::exists(&path).unwrap_or(false) {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).ok();
            }

            File::create(&path).unwrap();
        }

        Self {
            lock: FileLock::new(&path, lock_max_duration),
            path,
            cursor: 0,
        }
    }

    /// Locks the file and returns the content that wasn't synced since the last lock.
    pub fn lock(&mut self) -> Option<BufReader<File>> {
        self.lock.lock();

        let mut file = File::open(&self.path).unwrap();
        let end = file.metadata().unwrap().len();
        file.seek(SeekFrom::Start(self.cursor)).unwrap();

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
    /// Panics if the file isn't locked or there is an internal error.
    pub fn write(&mut self, content: &[u8]) {
        if !self.lock.is_lock {
            panic!("The cache file should be locked before writing content to it.")
        }

        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .unwrap();

        self.cursor += file.write(content).unwrap() as u64;
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
    pub fn lock(&mut self) {
        if self.is_lock {
            return;
        }

        let waiting_total = std::time::SystemTime::now();

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
                            std::thread::sleep(Duration::from_millis(30));
                        }
                    }
                    _ => {
                        if waiting_total.elapsed().unwrap() > self.lock_max_duration {
                            fs::remove_file(&self.path_lock).ok();
                        } else {
                            std::thread::sleep(Duration::from_millis(30));
                        }
                    }
                },
            };
        }

        self.is_lock = true;
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
