//! Environments as values: named in the storage that keeps them, created
//! empty, from a fork or from a bundle, merged into, activated and saved
//! whole.
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

/// What a new environment holds, or what a merge brings into one.
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
    use crate::bundle::{BundleFormat, ExportOptions, SqliteBundle, export, import_into};
    use crate::environment::{activate, active, file_name, load, path, root, sanitize};
    use crate::persistence::turso::same_file;
    use crate::persistence::{Access, Database, Storage, TursoStorage};
    use crate::sync::LazyLock;
    use alloc::boxed::Box;
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
            let incoming = Incoming::new(&id, origin)?;
            if let Incoming::Itself = incoming {
                return Err(EnvironmentError::SelfFork { name: id.name });
            }
            std::fs::create_dir_all(root()).map_err(storage_error)?;
            let scratch = ScratchFile::new(&id.path());
            let source = incoming.open(&scratch)?;
            id.open(Access::ReadWrite)?
                .replace_with(source.as_ref())
                .map_err(|reason| EnvironmentError::Storage { reason })?;
            Ok(id)
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

        /// Brings what `origin` holds into the environment, under the rules of
        /// [`import`](crate::bundle::import): a key the environment already
        /// holds keeps its value, and every other entry lands as
        /// [`Origin::Imported`](crate::persistence::Origin), which a value
        /// computed here later replaces.
        ///
        /// A merge writes entries and does nothing else: it never activates
        /// an environment, the active one included. Since no stored value
        /// changes, nothing a store already holds in memory goes stale: a lazy
        /// store reads a merged key the first time it misses, and an eager one
        /// at its next [`sync`](crate::persistence::Store::sync). Merging an
        /// environment into itself, or merging nothing, changes nothing.
        ///
        /// The entries are written in batches, so a writer of the same
        /// environment never waits on a whole merge, and a failure can leave
        /// part of `origin` merged. That part is right, not stale: merging
        /// again completes it.
        ///
        /// # Errors
        ///
        /// The environment or a fork's source lives in another process's
        /// storage, the bundle does not read, or the storage refused the
        /// write.
        pub fn merge(&self, origin: EnvironmentOrigin) -> Result<(), EnvironmentError> {
            self.check_local()?;
            let scratch = ScratchFile::new(&self.path());
            let Some(source) = Incoming::new(self, origin)?.open(&scratch)? else {
                return Ok(());
            };
            let path = self.path();
            // One handle per namespace, as `import` opens one storage per
            // namespace; an environment holds few namespaces, so the opens
            // are few.
            let open = |namespace: &str| -> Result<Box<dyn Storage>, EnvironmentError> {
                let database = Database::open_at(&path, Access::ReadWrite)
                    .map_err(|reason| EnvironmentError::Storage { reason })?;
                Ok(Box::new(TursoStorage::new(database, namespace.to_string())))
            };
            match import_into(&source, open)?.failed {
                0 => Ok(()),
                failed => Err(EnvironmentError::Storage {
                    reason: alloc::format!("{failed} entries were refused"),
                }),
            }
        }

        /// The environment serialized whole.
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage, or the storage
        /// refused the read.
        pub fn save(&self) -> Result<EnvironmentBundle, EnvironmentError> {
            self.check_local()?;
            let scratch = ScratchFile::new(&self.path());
            let options = ExportOptions {
                name: self.name.clone(),
                format: BundleFormat::Sqlite,
                ..Default::default()
            };
            export(&[self.path()], scratch.path(), &options).map_err(storage_error)?;
            let saved = std::fs::read(scratch.path()).map_err(storage_error)?;
            Ok(EnvironmentBundle {
                bytes: Bytes::from_bytes_vec(saved),
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

        fn open(&self, access: Access) -> Result<Database, EnvironmentError> {
            Database::open_at(&self.path(), access)
                .map_err(|reason| EnvironmentError::Storage { reason })
        }

        fn path(&self) -> PathBuf {
            PathBuf::from(&self.location)
        }
    }

    /// What an [`EnvironmentOrigin`] brings, read against the environment it
    /// goes into: the one check [`EnvironmentId::create`] and
    /// [`EnvironmentId::merge`] both run before reading anything.
    enum Incoming {
        /// No entries.
        Nothing,
        /// A fork of the environment it goes into, under any spelling of its
        /// name. `create` refuses it, since it would empty its own source;
        /// `merge` has nothing to bring.
        Itself,
        /// A fork of another environment of this process.
        Fork(EnvironmentId),
        /// A serialized environment.
        Bundle(EnvironmentBundle),
    }

    impl Incoming {
        /// Reads `origin` against `target`.
        ///
        /// # Errors
        ///
        /// A fork's source lives in another process's storage.
        fn new(
            target: &EnvironmentId,
            origin: EnvironmentOrigin,
        ) -> Result<Self, EnvironmentError> {
            Ok(match origin {
                EnvironmentOrigin::Empty => Self::Nothing,
                EnvironmentOrigin::Fork(source) => {
                    source.check_local()?;
                    match same_file(&source.path(), &target.path()) {
                        true => Self::Itself,
                        false => Self::Fork(source),
                    }
                }
                EnvironmentOrigin::Bundle(bundle) => Self::Bundle(bundle),
            })
        }

        /// The database the entries are read from, `None` when there are
        /// none: a fork's source, or a bundle written to `scratch` and checked
        /// to read.
        fn open(self, scratch: &ScratchFile) -> Result<Option<Database>, EnvironmentError> {
            Ok(match self {
                Self::Nothing | Self::Itself => None,
                Self::Fork(source) => Some(source.open(Access::ReadOnly)?),
                Self::Bundle(bundle) => {
                    std::fs::write(scratch.path(), &*bundle.bytes).map_err(storage_error)?;
                    // Validates the manifest; the entries are then read as an
                    // environment's, since a bundle file is one.
                    drop(SqliteBundle::open(scratch.path()).map_err(storage_error)?);
                    Some(
                        Database::open_at(scratch.path(), Access::ReadOnly)
                            .map_err(|reason| EnvironmentError::Storage { reason })?,
                    )
                }
            })
        }
    }

    /// A file beside an environment for one call's intermediate bytes,
    /// deleted with the files the engine writes beside it when dropped.
    ///
    /// The name is drawn at random, so two calls on one environment at once
    /// never write or delete each other's file.
    struct ScratchFile {
        path: PathBuf,
    }

    impl ScratchFile {
        fn new(beside: &Path) -> Self {
            let mut path = beside.to_path_buf();
            let suffix: u64 = crate::rand::gen_random();
            path.set_extension(alloc::format!("{suffix:016x}.scratch"));
            Self { path }
        }

        fn path(&self) -> &Path {
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

    fn storage_error(error: impl ToString) -> EnvironmentError {
        EnvironmentError::Storage {
            reason: error.to_string(),
        }
    }
}
