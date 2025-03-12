use core::{fmt::Display, time::Duration};
use std::{
    fs::{self, File, Metadata},
    io::{BufReader, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

/// A cache file is an append only file that is multi-process safe.
#[derive(Debug)]
pub struct CacheFile {
    path: PathBuf,
    path_lock: PathBuf,
    cursor: u64,
    is_lock: bool,
}

impl Display for CacheFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Cache file: {:?}", self.path)
    }
}

impl CacheFile {
    /// Create a new cache file.
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        let path: PathBuf = path.into();
        let file_name = path
            .file_name()
            .expect("Path to have a file name.")
            .to_str()
            .expect("File name to be valid");
        let mut path_lock = path.clone();
        path_lock.set_file_name(format!("{}.lock", file_name));

        let max_try = 10;
        let waiting_duration = 100; // ms
        let mut current_try = 0;
        let mut metadata: Option<Metadata> = None;

        loop {
            let is_already_init = std::fs::exists(&path).unwrap_or(false);
            if is_already_init {
                break;
            }

            // We hard reset the cache.
            if current_try >= max_try {
                std::fs::remove_file(&path_lock).ok();
                std::fs::remove_file(&path).ok();
                init_file(&path).unwrap();
                break;
            }

            match File::open(&path_lock) {
                Ok(file) => {
                    let metadata_curr = file.metadata().unwrap();
                    match &mut metadata {
                        Some(metadata) => {
                            // We are writing to the cache file from another process, don't need to
                            // initialize it.
                            if metadata.len() == metadata_curr.len() {
                                break;
                            } else {
                                *metadata = metadata_curr;
                            }
                        }
                        None => {
                            metadata = Some(metadata_curr);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(waiting_duration));
                    current_try += 1;
                }
                Err(err) => {
                    if let std::io::ErrorKind::NotFound = err.kind() {
                        if let Ok(value) = std::fs::exists(&path) {
                            // If both the lock and normal path are not created we
                            // initialize the cache file.
                            if !value {
                                if init_file(&path).is_ok() {
                                    break;
                                }
                            }
                        }
                    };
                    current_try += 1;
                    std::thread::sleep(Duration::from_millis(waiting_duration));
                }
            };
        }

        Self {
            path,
            path_lock,
            is_lock: false,
            cursor: 0,
        }
    }

    /// Lock the file and returns the content that wasn't synced since the last lock.
    pub fn lock(&mut self) -> Option<BufReader<File>> {
        loop {
            if std::fs::rename(&self.path, &self.path_lock).is_ok() {
                break;
            } else {
                std::thread::sleep(Duration::from_millis(30));
            }
        }

        self.is_lock = true;

        let mut file = File::open(&self.path_lock).unwrap();
        file.seek(SeekFrom::Start(self.cursor)).unwrap();
        let end = file.metadata().unwrap().len();

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
        loop {
            if std::fs::rename(&self.path_lock, &self.path).is_ok() {
                break;
            } else {
                std::thread::sleep(Duration::from_millis(30));
            }
        }

        self.is_lock = false;
    }

    /// Write the content to the file.
    ///
    /// Panics if the file isn't locked or there is an internal error.
    pub fn write(&mut self, content: &[u8]) {
        if !self.is_lock {
            panic!("The cache file should be locked before writing content to it.")
        }

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&self.path_lock)
            .unwrap();

        self.cursor += file.write(content).unwrap() as u64;
    }
}

fn init_file(file_path: &Path) -> std::io::Result<File> {
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    File::create(file_path)
}
