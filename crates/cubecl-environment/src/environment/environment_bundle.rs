//! An environment serialized whole, and what a reader learns of one without
//! loading it.

use alloc::string::String;
#[cfg(native_cache)]
use alloc::vec::Vec;

#[cfg(native_cache)]
use crate::persistence::NamespaceRoot;

#[cfg(native_cache)]
use crate::bundle::SqliteBundle;
use crate::bytes::Bytes;
#[cfg(native_cache)]
use crate::bytes::{OpenedFile, Reader};
#[cfg(native_cache)]
use crate::environment::scratch_file::ScratchFile;
#[cfg(native_cache)]
use crate::environment::{BuildStamp, EnvironmentError, EnvironmentSummary};
#[cfg(native_cache)]
use std::path::Path;
#[cfg(native_cache)]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The stem of the scratch file a bundle held in memory is described through.
#[cfg(native_cache)]
const SCRATCH_STEM: &str = "cubecl-bundle";

/// An environment serialized whole: what [`EnvironmentId::save`](super::EnvironmentId::save) answers and
/// an [`EnvironmentOrigin::Bundle`](super::EnvironmentOrigin::Bundle) is created from. The bytes are a
/// [`BundleFormat::Sqlite`](crate::bundle::BundleFormat) bundle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentBundle {
    /// The serialized environment.
    pub bytes: Bytes,
}

/// How an environment is saved into a bundle.
///
/// Built with [`new`](Self::new) and a setter per option, so an option added
/// later breaks no caller.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SaveOptions {
    /// The name the bundle's manifest gives it, for a person reading it back:
    /// the environment's own name when unset.
    #[serde(default)]
    pub name: Option<String>,
    /// What of the environment the bundle holds.
    #[serde(default)]
    pub content: SavedContent,
}

impl SaveOptions {
    /// The default save: the bundle carries the environment's name.
    pub fn new() -> Self {
        Self::default()
    }

    /// Name the bundle `name` in its manifest.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Save `content` of the environment.
    pub fn content(mut self, content: SavedContent) -> Self {
        self.content = content;
        self
    }
}

/// What of an environment a saved bundle holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SavedContent {
    /// Everything it holds.
    #[default]
    Everything,
    /// The tunes the work of the executable saving it used, as
    /// their tables store them, and nothing else: no kernel, no build record,
    /// no record of what the work used.
    ///
    /// Merged into another environment, they are loaded instead of measured
    /// by a build of the same cubecl version on the same hardware, which
    /// reads the same tables. A tune the environment holds already stays,
    /// and one measured there later replaces a brought one.
    UsedTunes,
}

#[cfg(native_cache)]
impl SavedContent {
    /// The roots of the namespaces a save of this content exports, before it
    /// selects among their entries: none for every namespace.
    pub(crate) fn exported_roots(&self) -> Vec<String> {
        match self {
            Self::Everything => Vec::new(),
            Self::UsedTunes => alloc::vec![NamespaceRoot::AUTOTUNE.into()],
        }
    }
}

/// What a saved bundle says of itself: everything a reader learns of it
/// without loading it, read in one opening of the bundle.
#[cfg(native_cache)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleDescription {
    /// The name the bundle's manifest gives it.
    pub name: String,
    /// The executable that saved the bundle, when its manifest names one:
    /// equal to [`BuildStamp::current`] when this executable did.
    pub saved_by: Option<BuildStamp>,
    /// When the bundle was saved, when its manifest says.
    pub saved_at: Option<SystemTime>,
    /// What the bundle holds, counted.
    pub summary: EnvironmentSummary,
}

#[cfg(native_cache)]
impl EnvironmentBundle {
    /// The bundle saved at `path`, its file opened now and its bytes read
    /// when something uses them.
    ///
    /// The open file is held as long as the bytes are, so a file renamed
    /// over `path`, or `path` removed, leaves the bundle reading the file
    /// it opened; a file written over in place is not protected. A
    /// [description](Self::read_description) reads the file in place
    /// rather than copy it.
    ///
    /// # Errors
    ///
    /// [`Storage`](EnvironmentError::Storage): the file does not open.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, EnvironmentError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(EnvironmentError::storage)?;
        let size = file.metadata().map_err(EnvironmentError::storage)?.len();
        Ok(Self {
            bytes: Bytes::from_opened_file(&OpenedFile::new(file, path), size, 0),
        })
    }

    /// What the bundle says of itself: its name, which executable saved
    /// it and when, and its counts, read in one opening of the bundle. A
    /// bundle still read from the file it was opened at is read there;
    /// one held in memory, or whose file changed, through a scratch copy,
    /// since the database engine reads files only.
    ///
    /// # Errors
    ///
    /// [`Storage`](EnvironmentError::Storage): the bytes do not read,
    /// they are not a bundle, or the bundle's entries cannot be counted.
    pub fn read_description(&self) -> Result<BundleDescription, EnvironmentError> {
        match self.whole_file() {
            Some(path) => BundleDescription::try_from(&opened(path)?),
            None => self.read_description_from_a_copy(),
        }
    }

    /// The file the bytes are, whole, while they are still read from it.
    fn whole_file(&self) -> Option<&Path> {
        self.bytes
            .file()?
            .whole_file_of_len(self.bytes.len() as u64)
    }

    /// [`read_description`](Self::read_description), from the bytes
    /// written to a scratch file.
    fn read_description_from_a_copy(&self) -> Result<BundleDescription, EnvironmentError> {
        let bytes = self
            .bytes
            .read(Reader::new())
            .map_err(EnvironmentError::storage)?;
        let scratch = ScratchFile::new(&std::env::temp_dir().join(SCRATCH_STEM));
        std::fs::write(scratch.path(), bytes).map_err(EnvironmentError::storage)?;
        // Declared after the scratch file, so closed before it is deleted.
        let bundle = opened(scratch.path())?;
        BundleDescription::try_from(&bundle)
    }
}

/// What `bundle`'s manifest and entries say of it.
#[cfg(native_cache)]
impl TryFrom<&SqliteBundle> for BundleDescription {
    type Error = EnvironmentError;

    fn try_from(bundle: &SqliteBundle) -> Result<Self, Self::Error> {
        let manifest = bundle.manifest();
        let namespaces = bundle.summary().map_err(EnvironmentError::storage)?;
        Ok(Self {
            name: manifest.name.clone(),
            saved_by: manifest.saved_by,
            saved_at: manifest
                .created_unix_secs
                .and_then(|seconds| UNIX_EPOCH.checked_add(Duration::from_secs(seconds))),
            summary: EnvironmentSummary::new(&namespaces),
        })
    }
}

/// The bundle file at `path`, opened.
#[cfg(native_cache)]
fn opened(path: &Path) -> Result<SqliteBundle, EnvironmentError> {
    SqliteBundle::open(path).map_err(EnvironmentError::storage)
}

#[cfg(all(test, native_cache))]
mod tests {
    use super::*;
    use crate::bundle::{BundleManifest, MANIFEST_SCHEMA};
    use crate::persistence::{Access, Database, Origin};
    use alloc::vec::Vec;
    use std::path::PathBuf;

    /// Describing a bundle opened from its file reads the file in place:
    /// its bytes are still unread afterwards, so no copy of it was made.
    #[test]
    fn a_description_reads_the_file_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = EnvironmentBundle::open(bundle_file(&dir)).unwrap();

        let description = bundle.read_description().unwrap();

        assert_eq!(description.name, "fixture");
        assert_eq!(description.summary.tunes, 1);
        assert!(bundle.bytes.file().is_some(), "the bytes are still unread");
    }

    /// A file shortened after the bundle was opened fails to read, as an
    /// error rather than a panic or a description of a prefix.
    #[test]
    fn a_file_truncated_after_opening_fails_to_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = bundle_file(&dir);
        let bundle = EnvironmentBundle::open(&path).unwrap();

        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(bundle.bytes.len() as u64 / 2).unwrap();
        drop(file);

        assert!(matches!(
            bundle.read_description(),
            Err(EnvironmentError::Storage { .. })
        ));
    }

    /// A file renamed over the opened one, or the path removed, leaves the
    /// bundle reading the file it opened.
    #[test]
    #[cfg(unix)]
    fn a_file_replaced_after_opening_is_not_the_one_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = bundle_file(&dir);
        let bundle = EnvironmentBundle::open(&path).unwrap();

        let other = dir.path().join("other");
        std::fs::write(&other, b"not a bundle, and longer than nothing").unwrap();
        std::fs::rename(&other, &path).unwrap();
        assert_eq!(bundle.read_description().unwrap().name, "fixture");

        std::fs::remove_file(&path).unwrap();
        assert_eq!(bundle.read_description().unwrap().name, "fixture");
    }

    /// A bundle whose entries cannot be counted says so, rather than
    /// describing itself as empty.
    #[test]
    fn a_bundle_whose_entries_do_not_count_fails_to_describe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.cubecl");
        let database = Database::open_at(&path, Access::ReadWrite).unwrap();
        manifest().write_to(&database).unwrap();
        database
            .with_connection(|connection| connection.execute("DROP TABLE entries", &[]))
            .unwrap();
        database.into_standalone().unwrap();

        let bundle = EnvironmentBundle::open(&path).unwrap();

        assert!(matches!(
            bundle.read_description(),
            Err(EnvironmentError::Storage { .. })
        ));
    }

    /// A bundle file in `dir` holding one tune.
    fn bundle_file(dir: &tempfile::TempDir) -> PathBuf {
        let path = dir.path().join("fixture.cubecl");
        let database = Database::open_at(&path, Access::ReadWrite).unwrap();
        database.insert("autotune/0.12.0/matmul", b"key", b"tune", Origin::Local);
        manifest().write_to(&database).unwrap();
        database.into_standalone().unwrap();
        path
    }

    fn manifest() -> BundleManifest {
        BundleManifest {
            schema: MANIFEST_SCHEMA,
            name: "fixture".into(),
            cubecl_version: env!("CARGO_PKG_VERSION").into(),
            created_unix_secs: None,
            environments: Vec::new(),
            saved_by: None,
        }
    }
}
