//! Environments as values: named in the storage that keeps them, created
//! empty, from a fork or from a bundle, merged into, activated and saved
//! whole.
//!
//! The functions of [`environment`](super) act on this process's storage by
//! name. An [`EnvironmentId`] also names the storage, so an environment of
//! another process — a server's, reached over the network — is refused rather
//! than confused with a local one of the same name.

use alloc::string::{String, ToString};

use super::EnvironmentBundle;

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

impl EnvironmentError {
    /// The storage refused a read or a write, for the reason `error` gives.
    #[cfg_attr(not(native_cache), allow(dead_code))]
    pub(crate) fn storage(error: impl ToString) -> Self {
        Self::Storage {
            reason: error.to_string(),
        }
    }
}

#[cfg(native_cache)]
mod storage {
    use super::*;
    use crate::bundle::{BundleFormat, ExportOptions, SqliteBundle, export, import_into};
    use crate::bytes::{Bytes, Reader};
    use crate::environment::scratch_file::ScratchFile;
    use crate::environment::{
        EnvironmentSummary, SaveOptions, activate, active, active_namespaces, file_name, load,
        path, root, sanitize,
    };
    use crate::persistence::database::same_file;
    use crate::persistence::{Access, Database, DatabaseStorage, Storage};
    use crate::sync::LazyLock;
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use std::path::{Path, PathBuf};

    /// The storage of this process; see [`EnvironmentId::storage`].
    static STORAGE: LazyLock<u64> = LazyLock::new(crate::rand::gen_random);

    impl EnvironmentId {
        /// The environment active in this process: the file mounted by
        /// [`load`](crate::environment::load), named after it, or the named one.
        pub fn active() -> Self {
            let location = path();
            let named = active();
            let name = if location == root().join(file_name(&named)) {
                named.to_string()
            } else {
                location
                    .file_stem()
                    .map_or_else(|| named.to_string(), |stem| stem.to_string_lossy().into())
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
        /// the replacement lands as a whole ([`Database::replace_with`]), so a
        /// failure leaves the environment as it was.
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
            std::fs::create_dir_all(root()).map_err(EnvironmentError::storage)?;
            let scratch = ScratchFile::new(&id.path());
            let source = incoming.open(&scratch)?;
            id.open(Access::ReadWrite)?
                .replace_with(source.as_ref())
                .map_err(EnvironmentError::storage)?;
            Ok(id)
        }

        /// Makes this the active environment of the process, answering the one
        /// it replaces; see [`activate`](crate::environment::activate). An
        /// environment outside the current root, or one that was mounted, is
        /// mounted again with [`load`](crate::environment::load).
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage.
        pub fn activate(&self) -> Result<Self, EnvironmentError> {
            self.check_local()?;
            let previous = Self::active();
            if self.path() == root().join(file_name(&self.name)) {
                activate(&self.name);
            } else {
                load(self.path());
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
                    .map_err(EnvironmentError::storage)?;
                Ok(Box::new(DatabaseStorage::new(
                    database,
                    namespace.to_string(),
                )))
            };
            match import_into(&source, open)?.failed {
                0 => Ok(()),
                failed => Err(EnvironmentError::Storage {
                    reason: alloc::format!("{failed} entries were refused"),
                }),
            }
        }

        /// The environment serialized whole, under the name `options` gives
        /// it, and stamped with this executable as the one that saved it.
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage, or the storage
        /// refused the read.
        pub fn save(&self, options: SaveOptions) -> Result<EnvironmentBundle, EnvironmentError> {
            self.check_local()?;
            let scratch = ScratchFile::new(&self.path());
            let options = ExportOptions {
                name: options.name.unwrap_or_else(|| self.name.clone()),
                format: BundleFormat::Sqlite,
                ..Default::default()
            };
            export(&[self.path()], scratch.path(), &options).map_err(EnvironmentError::storage)?;
            let saved = std::fs::read(scratch.path()).map_err(EnvironmentError::storage)?;
            Ok(EnvironmentBundle {
                bytes: Bytes::from_bytes_vec(saved),
            })
        }

        /// What the environment holds now, counted without saving it: cheap
        /// enough to ask after every step of a build.
        ///
        /// The active environment is counted where its stores write, which
        /// is memory when its database would not open.
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage, or its
        /// database does not open or refuses the count.
        pub fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError> {
            self.check_local()?;
            let namespaces = if same_file(&path(), &self.path()) {
                active_namespaces()
            } else {
                self.open(Access::ReadOnly)?.summary()
            };
            Ok(EnvironmentSummary::new(
                &namespaces.map_err(EnvironmentError::storage)?,
            ))
        }

        fn local(name: String, location: &Path) -> Self {
            Self {
                storage: *STORAGE,
                name,
                location: location.display().to_string(),
            }
        }

        fn check_local(&self) -> Result<(), EnvironmentError> {
            if self.storage == *STORAGE {
                Ok(())
            } else {
                Err(EnvironmentError::IncompatibleStorage {
                    name: self.name.clone(),
                })
            }
        }

        /// The database the environment lives in, opened with `access`: for
        /// a runtime that edits what it stored there, as a prune removes what
        /// no workload loads.
        ///
        /// # Errors
        ///
        /// The environment lives in another process's storage, or the
        /// database does not open.
        pub fn database(&self, access: Access) -> Result<Database, EnvironmentError> {
            self.check_local()?;
            self.open(access)
        }

        fn open(&self, access: Access) -> Result<Database, EnvironmentError> {
            Database::open_at(&self.path(), access).map_err(EnvironmentError::storage)
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
                    if same_file(&source.path(), &target.path()) {
                        Self::Itself
                    } else {
                        Self::Fork(source)
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
                    let bytes = bundle
                        .bytes
                        .read(Reader::new())
                        .map_err(EnvironmentError::storage)?;
                    std::fs::write(scratch.path(), bytes).map_err(EnvironmentError::storage)?;
                    // Validates the manifest; the entries are then read as an
                    // environment's, since a bundle file is one.
                    drop(SqliteBundle::open(scratch.path()).map_err(EnvironmentError::storage)?);
                    Some(
                        Database::open_at(scratch.path(), Access::ReadOnly)
                            .map_err(EnvironmentError::storage)?,
                    )
                }
            })
        }
    }
}
