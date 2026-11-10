//! Scratch files beside an environment.

use std::path::{Path, PathBuf};

/// A file beside an environment for one call's intermediate bytes,
/// deleted with the files the engine writes beside it when dropped.
///
/// The name is drawn at random, so two calls on one environment at once
/// never write or delete each other's file.
pub(crate) struct ScratchFile {
    path: PathBuf,
}

impl ScratchFile {
    pub(crate) fn new(beside: &Path) -> Self {
        let mut path = beside.to_path_buf();
        let suffix: u64 = crate::rand::gen_random();
        path.set_extension(alloc::format!("{suffix:016x}.scratch"));
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm", "-tshm"] {
            let mut file = std::ffi::OsString::from(&self.path);
            file.push(suffix);
            let _ = std::fs::remove_file(file);
        }
    }
}
