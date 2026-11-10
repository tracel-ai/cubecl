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

    impl EnvironmentBundle {
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
        pub fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError> {
            let path = path.as_ref();
            let size = std::fs::metadata(path)
                .map_err(EnvironmentError::storage)?
                .len();
            Ok(Self {
                bytes: Bytes::from_file(path, size, 0),
            })
        }

        /// The name the bundle's manifest gives it.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        pub fn name(&self) -> Result<String, EnvironmentError> {
            self.open(|bundle| bundle.manifest().name.clone())
        }

        /// The executable that saved the bundle, when its manifest names one:
        /// equal to [`BuildStamp::current`] when this executable did.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        pub fn saved_by(&self) -> Result<Option<BuildStamp>, EnvironmentError> {
            self.open(|bundle| bundle.manifest().saved_by)
        }

        /// What the bundle holds, counted.
        ///
        /// # Errors
        ///
        /// The bytes are not a bundle, or do not read.
        pub fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError> {
            self.open(|bundle| EnvironmentSummary::new(&bundle.summary()))
        }

        /// Opens the bundle and answers what `read` takes from it: in place
        /// when the bytes are still the whole of a file, else from a scratch
        /// copy, since the database engine reads files only.
        fn open<T>(&self, read: impl FnOnce(&SqliteBundle) -> T) -> Result<T, EnvironmentError> {
            if let Some(file) = self.file() {
                let bundle = SqliteBundle::open(file).map_err(EnvironmentError::storage)?;
                return Ok(read(&bundle));
            }
            let scratch = ScratchFile::new(&std::env::temp_dir().join("cubecl-bundle"));
            std::fs::write(scratch.path(), &*self.bytes).map_err(EnvironmentError::storage)?;
            // Declared after the scratch file, so closed before it is deleted.
            let bundle = SqliteBundle::open(scratch.path()).map_err(EnvironmentError::storage)?;
            Ok(read(&bundle))
        }

        /// The file the bytes are, whole, while they are still read from it.
        fn file(&self) -> Option<&Path> {
            let window = self.bytes.file()?;
            let whole = window.offset == 0
                && std::fs::metadata(window.path)
                    .is_ok_and(|file| file.len() == self.bytes.len() as u64);
            whole.then_some(window.path)
        }
    }
}
