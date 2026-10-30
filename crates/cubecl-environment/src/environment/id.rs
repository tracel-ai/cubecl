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
            Self::Storage { reason } => write!(f, "the environment storage failed: {reason}"),
        }
    }
}

impl core::error::Error for EnvironmentError {}

#[cfg(native_cache)]
mod storage {
    use super::*;
    use crate::bundle::{BundleFormat, ExportOptions, SqliteBundle, export};
    use crate::environment::{activate, active, file_name, path, root, sanitize};
    use crate::persistence::{Access, Database, Origin};
    use crate::sync::LazyLock;
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use std::path::{Path, PathBuf};

    /// The storage of this process; see [`EnvironmentId::storage`].
    static STORAGE: LazyLock<u64> = LazyLock::new(crate::rand::gen_random);

    impl EnvironmentId {
        /// The environment active in this process.
        pub fn active() -> Self {
            Self::local(active().to_string())
        }

        /// Creates the environment `name` in this process's storage, holding
        /// what `origin` says, replacing one of that name. It is not activated.
        ///
        /// The environment is emptied entry by entry rather than deleted: the
        /// process keeps one connection per database for its whole life, so a
        /// file replaced under it would leave that connection writing into one
        /// nothing reads. Everything that could refuse is checked first.
        ///
        /// # Errors
        ///
        /// The environment to replace is active, a fork's source lives in
        /// another process's storage, the bundle does not read, or the storage
        /// refused the write.
        pub fn create(name: &str, origin: EnvironmentOrigin) -> Result<Self, EnvironmentError> {
            let id = Self::local(sanitize(name));
            if path() == id.path() {
                return Err(EnvironmentError::Active { name: id.name });
            }
            let staged = id.staged("bundle");
            let created = id.fill(origin, &staged);
            discard(&staged);
            created.map(|_| id)
        }

        /// Makes this the active environment of the process, answering the one
        /// it replaces; see [`activate`](super::super::activate).
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage.
        pub fn activate(&self) -> Result<Self, EnvironmentError> {
            self.check_local()?;
            let previous = Self::active();
            activate(&self.name);
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

        fn local(name: String) -> Self {
            Self {
                storage: *STORAGE,
                name,
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

        /// Empties the environment, then copies `origin` into it. A bundle is
        /// written to `staged` to be read.
        fn fill(&self, origin: EnvironmentOrigin, staged: &Path) -> Result<(), EnvironmentError> {
            let source: Option<Box<dyn crate::bundle::Bundle>> = match origin {
                EnvironmentOrigin::Empty => None,
                EnvironmentOrigin::Fork(source) => {
                    source.check_local()?;
                    Some(Box::new(source.open(Access::ReadOnly)?))
                }
                EnvironmentOrigin::Bundle(bundle) => {
                    std::fs::write(staged, &*bundle.bytes).map_err(storage_error)?;
                    Some(Box::new(SqliteBundle::open(staged).map_err(storage_error)?))
                }
            };

            let target = self.open(Access::ReadWrite)?;
            for namespace in target.namespaces() {
                target.purge(&namespace);
            }
            if let Some(source) = source {
                crate::bundle::copy(source.as_ref(), |namespace| {
                    let (target, namespace) = (&target, namespace.to_string());
                    move |batch: &mut dyn Iterator<Item = (Bytes, Bytes)>| {
                        target.insert_many(&namespace, batch, Origin::Imported)
                    }
                });
            }
            Ok(())
        }

        fn open(&self, access: Access) -> Result<Database, EnvironmentError> {
            Database::open_at(&self.path(), access)
                .map_err(|reason| EnvironmentError::Storage { reason })
        }

        fn path(&self) -> PathBuf {
            root().join(file_name(&self.name))
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
