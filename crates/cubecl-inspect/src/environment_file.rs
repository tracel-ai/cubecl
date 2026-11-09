use crate::report::NamespaceRoot;
use crate::{InspectError, environment};
use cubecl_environment::bundle::{BundleManifest, ExportOptions, export};
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;
use std::path::{Path, PathBuf};

/// One environment file, opened read-only and checked to be one this build
/// reads: what an [`Inspector`](crate::Inspector) reads the reports of, and
/// what is [pruned](Self::prune) or [stripped](Self::strip).
///
/// Read-only is what makes it safe to point at an environment a running
/// process is writing: the database is shared through WAL, and nothing but
/// [`prune`](Self::prune) writes, migrates or checkpoints it.
pub struct EnvironmentFile {
    database: Database,
    path: PathBuf,
}

impl EnvironmentFile {
    /// Open the file at `path`.
    ///
    /// # Errors
    ///
    /// [`InspectError::Open`] when the file is missing, or is not an
    /// environment at the schema this build reads.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, InspectError> {
        let path = path.into();
        // A read-only open of a missing file fails too, but with the engine's
        // own message, which does not say whether the file is missing or
        // unreadable.
        if !path.is_file() {
            return Err(InspectError::Open {
                path,
                reason: "no such file".to_string(),
            });
        }
        // Refused unless the file is an environment at this build's schema:
        // anything else would read as an empty one.
        match Database::open_at(&path, Access::ReadOnly) {
            Ok(database) => Ok(Self { database, path }),
            Err(reason) => Err(InspectError::Open { path, reason }),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The database the file holds, open read-only.
    pub fn database(&self) -> &Database {
        &self.database
    }

    /// Drop the records of every session but the newest `keep`, in place,
    /// and open what is left.
    ///
    /// Takes the file, whose read-only handle has to be closed first: a file
    /// open for reading in this process cannot be written by it. The write
    /// goes through a connection of its own. A read-write open migrates a
    /// file of another schema by dropping its entries; this one cannot,
    /// because [`open`](Self::open) refused any file this build's cubecl did
    /// not write.
    ///
    /// # Errors
    ///
    /// [`InspectError::Prune`] when the file cannot be opened for writing,
    /// such as one in a read-only directory; [`InspectError::Open`] when the
    /// pruned file cannot be opened again.
    pub fn prune(self, keep: usize) -> Result<Self, InspectError> {
        let Self { database, path } = self;
        let refused = |reason: String| InspectError::Prune {
            path: path.clone(),
            reason,
        };
        // A read-write open switches the file to WAL for good. A file that was
        // not in WAL — shipped, meant for a read-only directory — is shipped
        // again, as `strip` leaves one.
        let shipped = database.is_standalone();
        drop(database);
        let writable = Database::open_at(&path, Access::ReadWrite).map_err(refused)?;
        Records::new(&writable).prune(keep);
        match shipped {
            true => writable.into_standalone().map_err(refused)?,
            false => drop(writable),
        }
        Self::open(path)
    }

    /// Write a copy of the file to `out` without its records — the account
    /// of how it was built, which a distributed environment has no use for —
    /// and open the copy.
    ///
    /// # Errors
    ///
    /// [`InspectError::Export`] when the copy cannot be written to `out`;
    /// [`InspectError::Open`] when it cannot be opened.
    pub fn strip(&self, out: &Path) -> Result<Self, InspectError> {
        let manifest = BundleManifest::new(&self.database).ok();
        let options = ExportOptions {
            name: manifest.as_ref().map_or_else(
                || environment::stem(&self.path),
                |manifest| manifest.name.clone(),
            ),
            environments: manifest
                .map(|manifest| manifest.environments)
                .unwrap_or_default(),
            // Every layout of records, not only the one this build reads.
            excluded_namespaces: vec![NamespaceRoot::records().as_str().to_string()],
            ..Default::default()
        };
        export(&[&self.path], out, &options).map_err(|err| InspectError::Export {
            path: out.to_path_buf(),
            reason: err.to_string(),
        })?;
        Self::open(out)
    }
}
