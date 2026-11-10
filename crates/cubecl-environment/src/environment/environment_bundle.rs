//! An environment serialized whole, and what a reader learns of one without
//! loading it.

use alloc::string::String;

use crate::bytes::Bytes;

/// An environment serialized whole: what [`EnvironmentId::save`](super::EnvironmentId::save) answers and
/// an [`EnvironmentOrigin::Bundle`](super::EnvironmentOrigin::Bundle) is created from. The bytes are a
/// [`BundleFormat::Sqlite`](crate::bundle::BundleFormat) bundle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentBundle {
    /// The serialized environment.
    pub bytes: Bytes,
}

/// How an environment is saved into a bundle.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SaveOptions {
    /// The name the bundle's manifest gives it, for a person reading it back:
    /// the environment's own name when unset.
    #[serde(default)]
    pub name: Option<String>,
}

#[cfg(native_cache)]
mod reading {
    use super::*;
    use crate::bundle::SqliteBundle;
    use crate::environment::scratch_file::ScratchFile;
    use crate::environment::{BuildStamp, EnvironmentError, EnvironmentSummary};
    use std::path::Path;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    /// What a saved bundle says of itself, read without loading it.
    ///
    /// A trait rather than methods of [`EnvironmentBundle`], as cubecl-server's
    /// `Prune` is for an environment: a layer
    /// that carries the bundle under its own name reads it under its own
    /// error, and an inherent method of the same name would shadow its read.
    pub trait ReadBundle: Sized {
        /// The bundle saved at `path`.
        ///
        /// Its bytes are read from the file when something uses them, and the
        /// readers below open the file in place, so learning a bundle's name
        /// or counts costs no copy of it. The file must therefore stay as it
        /// is while the bundle is held; one replaced meanwhile is the one read.
        ///
        /// # Errors
        ///
        /// The file cannot be read.
        fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError>;

        /// The name the bundle's manifest gives it.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        fn name(&self) -> Result<String, EnvironmentError>;

        /// The executable that saved the bundle, when its manifest names one:
        /// equal to [`BuildStamp::current`] when this executable did.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        fn saved_by(&self) -> Result<Option<BuildStamp>, EnvironmentError>;

        /// When the bundle was saved, when its manifest says.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        fn saved_at(&self) -> Result<Option<SystemTime>, EnvironmentError>;

        /// What the bundle holds, counted.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError>;
    }

    impl ReadBundle for EnvironmentBundle {
        fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError> {
            let path = path.as_ref();
            let size = std::fs::metadata(path)
                .map_err(EnvironmentError::storage)?
                .len();
            Ok(Self {
                bytes: Bytes::from_file(path, size, 0),
            })
        }

        fn name(&self) -> Result<String, EnvironmentError> {
            open(self, |bundle| bundle.manifest().name.clone())
        }

        fn saved_by(&self) -> Result<Option<BuildStamp>, EnvironmentError> {
            open(self, |bundle| bundle.manifest().saved_by)
        }

        fn saved_at(&self) -> Result<Option<SystemTime>, EnvironmentError> {
            open(self, |bundle| {
                let seconds = bundle.manifest().created_unix_secs?;
                UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
            })
        }

        fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError> {
            open(self, |bundle| EnvironmentSummary::new(&bundle.summary()))
        }
    }

    /// Opens `bundle` and answers what `read` takes from it: in place when its
    /// bytes are still the whole of a file, else from a scratch copy, since
    /// the database engine reads files only.
    fn open<T>(
        bundle: &EnvironmentBundle,
        read: impl FnOnce(&SqliteBundle) -> T,
    ) -> Result<T, EnvironmentError> {
        if let Some(file) = whole_file(bundle) {
            let opened = SqliteBundle::open(file).map_err(EnvironmentError::storage)?;
            return Ok(read(&opened));
        }
        let scratch = ScratchFile::new(&std::env::temp_dir().join("cubecl-bundle"));
        std::fs::write(scratch.path(), &*bundle.bytes).map_err(EnvironmentError::storage)?;
        // Declared after the scratch file, so closed before it is deleted.
        let opened = SqliteBundle::open(scratch.path()).map_err(EnvironmentError::storage)?;
        Ok(read(&opened))
    }

    /// The file `bundle`'s bytes are, whole, while they are still read from it.
    fn whole_file(bundle: &EnvironmentBundle) -> Option<&Path> {
        let window = bundle.bytes.file()?;
        let whole = window.offset == 0
            && std::fs::metadata(window.path)
                .is_ok_and(|file| file.len() == bundle.bytes.len() as u64);
        whole.then_some(window.path)
    }
}

#[cfg(native_cache)]
pub use reading::ReadBundle;
