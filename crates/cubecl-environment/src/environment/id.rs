//! Environments as values: named in the storage that keeps them, created
//! empty, from a fork or from a bundle, activated and saved whole.
//!
//! The functions of [`environment`](super) act on this process's storage by
//! name. An [`EnvironmentId`] also names the storage, so an environment of
//! another process — a server's, reached over the network — is refused rather
//! than confused with a local one of the same name.

use alloc::string::String;

use crate::bytes::Bytes;

/// Names an environment in the storage that keeps it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentId {
    /// The storage the environment lives in: random, drawn once per process,
    /// so a process never takes another's environment for its own.
    pub storage: u64,
    /// The environment's name in that storage, reduced to a safe file name.
    pub name: String,
    /// The database file the environment lives in, fixed when the id is
    /// made: the id keeps naming that environment after the root changes or a
    /// file is mounted.
    pub location: String,
}

/// An environment serialized whole: what [`EnvironmentId::save`] answers and
/// an [`EnvironmentOrigin::Bundle`] is created from. The bytes are a
/// [`BundleFormat::Sqlite`](crate::bundle::BundleFormat) bundle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentBundle {
    /// The serialized environment.
    pub bytes: Bytes,
}

/// What a new environment holds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EnvironmentOrigin {
    /// Nothing.
    Empty,
    /// What the environment of the id holds now, copied.
    Fork(EnvironmentId),
    /// What the bundle holds.
    Bundle(EnvironmentBundle),
}

/// Why an environment could not be created, activated or saved.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EnvironmentError {
    /// The environment lives in another process's storage. Carry it over by
    /// saving it there and creating one from the bundle here.
    IncompatibleStorage {
        /// The environment's name.
        name: String,
    },
    /// The environment cannot be replaced while it is active.
    Active {
        /// The environment's name.
        name: String,
    },
    /// The environment would be created from itself.
    SelfFork {
        /// The environment's name.
        name: String,
    },
    /// The storage refused a read or a write.
    Storage {
        /// Why, in the storage's words.
        reason: String,
    },
}

impl core::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IncompatibleStorage { name } => write!(
                f,
                "the environment {name} lives in another process's storage"
            ),
            Self::Active { name } => {
                write!(
                    f,
                    "the environment {name} is active, and cannot be replaced"
                )
            }
            Self::SelfFork { name } => {
                write!(f, "the environment {name} cannot be forked into itself")
            }
            Self::Storage { reason } => write!(f, "the environment storage failed: {reason}"),
        }
    }
}

impl core::error::Error for EnvironmentError {}

#[cfg(native_cache)]
mod storage {
    use super::*;
    use crate::bundle::{BundleFormat, ExportOptions, SqliteBundle, export};
    use crate::environment::{activate, active, file_name, load, path, root, sanitize};
    use crate::persistence::turso::same_file;
    use crate::persistence::{Access, Database};
    use crate::sync::LazyLock;
    use alloc::string::ToString;
    use std::path::{Path, PathBuf};

    /// The storage of this process; see [`EnvironmentId::storage`].
    static STORAGE: LazyLock<u64> = LazyLock::new(crate::rand::gen_random);

    impl EnvironmentId {
        /// The environment active in this process: the file mounted by
        /// [`load`](super::super::load), named after it, or the named one.
        pub fn active() -> Self {
            let location = path();
            let named = active();
            let name = match location == root().join(file_name(&named)) {
                true => named.to_string(),
                false => location
                    .file_stem()
                    .map_or_else(|| named.to_string(), |stem| stem.to_string_lossy().into()),
            };
            Self::local(name, &location)
        }

        /// Creates the environment `name` in this process's storage, holding
        /// what `origin` says, replacing one of that name. It is not activated.
        ///
        /// The environment is emptied entry by entry rather than deleted: the
        /// process keeps one connection per database for its whole life, so a
        /// file replaced under it would leave that connection writing into one
        /// nothing reads. Everything that could refuse is checked first, and
        /// the replacement is one transaction, so a failure leaves the
        /// environment as it was.
        ///
        /// # Errors
        ///
        /// The environment to replace is active, a fork's source lives in
        /// another process's storage or is the environment itself, the bundle
        /// does not read, or the storage refused the write.
        pub fn create(name: &str, origin: EnvironmentOrigin) -> Result<Self, EnvironmentError> {
            let name = sanitize(name);
            let id = Self::local(name.clone(), &root().join(file_name(&name)));
            if same_file(&path(), &id.path()) {
                return Err(EnvironmentError::Active { name: id.name });
            }
            std::fs::create_dir_all(root()).map_err(storage_error)?;
            let staged = id.staged("bundle");
            let created = id.fill(origin, &staged);
            discard(&staged);
            created.map(|_| id)
        }

        /// Makes this the active environment of the process, answering the one
        /// it replaces; see [`activate`](super::super::activate). An
        /// environment outside the current root, or one that was mounted, is
        /// mounted again with [`load`](super::super::load).
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage.
        pub fn activate(&self) -> Result<Self, EnvironmentError> {
            self.check_local()?;
            let previous = Self::active();
            match self.path() == root().join(file_name(&self.name)) {
                true => activate(&self.name),
                false => load(self.path()),
            }
            Ok(previous)
        }

        /// The environment serialized whole.
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage, or the storage
        /// refused the read.
        pub fn save(&self) -> Result<EnvironmentBundle, EnvironmentError> {
            self.check_local()?;
            let staged = self.staged("saved");
            let options = ExportOptions {
                name: self.name.clone(),
                format: BundleFormat::Sqlite,
                ..Default::default()
            };
            let saved = export(&[self.path()], &staged, &options)
                .map_err(storage_error)
                .and_then(|_| std::fs::read(&staged).map_err(storage_error));
            discard(&staged);
            Ok(EnvironmentBundle {
                bytes: Bytes::from_bytes_vec(saved?),
            })
        }

        fn local(name: String, location: &Path) -> Self {
            Self {
                storage: *STORAGE,
                name,
                location: location.display().to_string(),
            }
        }

        fn check_local(&self) -> Result<(), EnvironmentError> {
            match self.storage == *STORAGE {
                true => Ok(()),
                false => Err(EnvironmentError::IncompatibleStorage {
                    name: self.name.clone(),
                }),
            }
        }

        /// Replaces what the environment holds with `origin`, once the source
        /// is known to read. A bundle is written to `staged` to be read.
        fn fill(&self, origin: EnvironmentOrigin, staged: &Path) -> Result<(), EnvironmentError> {
            let source = match origin {
                EnvironmentOrigin::Empty => None,
                EnvironmentOrigin::Fork(source) => {
                    source.check_local()?;
                    if same_file(&source.path(), &self.path()) {
                        return Err(EnvironmentError::SelfFork {
                            name: self.name.clone(),
                        });
                    }
                    Some(source.open(Access::ReadOnly)?)
                }
                EnvironmentOrigin::Bundle(bundle) => {
                    std::fs::write(staged, &*bundle.bytes).map_err(storage_error)?;
                    // Validates the manifest; the entries are then read as an
                    // environment's, since a bundle file is one.
                    drop(SqliteBundle::open(staged).map_err(storage_error)?);
                    Some(
                        Database::open_at(staged, Access::ReadOnly)
                            .map_err(|reason| EnvironmentError::Storage { reason })?,
                    )
                }
            };

            self.open(Access::ReadWrite)?
                .replace_with(source.as_ref())
                .map_err(|reason| EnvironmentError::Storage { reason })
        }

        fn open(&self, access: Access) -> Result<Database, EnvironmentError> {
            Database::open_at(&self.path(), access)
                .map_err(|reason| EnvironmentError::Storage { reason })
        }

        fn path(&self) -> PathBuf {
            PathBuf::from(&self.location)
        }

        /// A scratch file beside the environment, for `purpose`.
        fn staged(&self, purpose: &str) -> PathBuf {
            let mut path = self.path();
            path.set_extension(alloc::format!("{purpose}.staged"));
            path
        }
    }

    /// Deletes a scratch file and the files the engine writes beside it.
    fn discard(path: &Path) {
        for suffix in ["", "-wal", "-shm", "-tshm"] {
            let mut file = std::ffi::OsString::from(path);
            file.push(suffix);
            let _ = std::fs::remove_file(file);
        }
    }

    fn storage_error(error: impl ToString) -> EnvironmentError {
        EnvironmentError::Storage {
            reason: error.to_string(),
        }
    }
}
