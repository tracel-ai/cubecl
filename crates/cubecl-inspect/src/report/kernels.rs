use cubecl_server::compiler::{CompilationOutcome, CompilationRecord, KernelCacheKey};
use serde::{Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

/// Every kernel the environment's builds compiled or loaded, and what that
/// cost, per instance and per kernel type.
#[derive(Clone, Debug, Serialize)]
pub struct KernelReport {
    pub kernels: Vec<KernelRow>,
    /// The instances folded by kernel type, costliest first.
    pub families: Vec<FamilyRow>,
    /// Artifacts the file stores with no recorded trip: compiled by a build
    /// that recorded nothing.
    pub unrecorded: u64,
}

/// The artifacts a file's compilation store holds, by entry, with their size.
pub type StoredArtifacts = BTreeMap<StoreEntry, u64>;

/// What names an artifact in the compilation store: the kernel id's hash and
/// the build's — a `KernelCacheKey`, ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoreEntry {
    pub id: u128,
    pub build: u128,
}

impl From<&KernelCacheKey> for StoreEntry {
    fn from(key: &KernelCacheKey) -> Self {
        Self {
            id: key.id,
            build: key.build_id,
        }
    }
}

/// A stable handle on one kernel instance: the hash of its id that names its
/// artifact in the compilation store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelHash(pub u128);

/// One kernel instance, its trips folded.
#[derive(Clone, Debug, Serialize)]
pub struct KernelRow {
    pub id: KernelHash,
    /// The kernel's type, in full.
    pub kernel: String,
    /// The instance: its id rendered.
    pub instance: String,
    /// The size of its artifact in the store, when the file holds one.
    pub bytes: Option<u64>,
    /// Times it was compiled fresh.
    pub compiled: u64,
    /// Times it was loaded from the store.
    pub loaded: u64,
    /// Compiling it, over every fresh compile.
    pub compiling: Duration,
    /// Loading it, over every store load.
    pub loading: Duration,
    /// The source, when a record kept it.
    pub source: Option<String>,
}

/// One kernel type, its instances folded.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyRow {
    pub kernel: String,
    pub instances: u64,
    pub compiled: u64,
    pub loaded: u64,
    pub compiling: Duration,
    pub loading: Duration,
    pub bytes: u64,
}

/// The orders a [`KernelReport`] lists its kernels in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum KernelOrder {
    /// Costliest to compile first.
    #[default]
    Compile,
    /// Largest artifact first.
    Size,
    /// By kernel type.
    Name,
}

impl KernelHash {
    /// Whether the hash's hex rendering starts with `prefix`, the way a
    /// command line names an instance.
    pub fn matches(&self, prefix: &str) -> bool {
        format!("{:032x}", self.0).starts_with(&prefix.to_ascii_lowercase())
    }
}

/// Twelve hex digits: enough to tell a file's kernels apart, short enough to
/// type.
impl fmt::Display for KernelHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &format!("{:032x}", self.0)[..12])
    }
}

/// In full, so a JSON reader has the whole hash.
impl Serialize for KernelHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("{:032x}", self.0))
    }
}

impl KernelRow {
    /// The type's last path segment, without its generics: a name a table can
    /// print.
    pub fn short_name(&self) -> &str {
        short_name(&self.kernel)
    }
}

impl FamilyRow {
    /// See [`KernelRow::short_name`].
    pub fn short_name(&self) -> &str {
        short_name(&self.kernel)
    }
}

fn short_name(kernel: &str) -> &str {
    let name = kernel.split('<').next().unwrap_or(kernel);
    name.rsplit("::").next().unwrap_or(name)
}

impl KernelReport {
    /// Fold the recorded trips by instance, sized from the artifacts the
    /// file `stored`.
    pub fn new(trips: &[CompilationRecord], stored: &StoredArtifacts) -> Self {
        let mut rows = BTreeMap::<StoreEntry, KernelRow>::new();
        for trip in trips {
            let entry = StoreEntry::from(&trip.key);
            let row = rows.entry(entry).or_insert_with(|| KernelRow {
                id: KernelHash(trip.key.id),
                kernel: trip.kernel.clone(),
                instance: trip.id.clone(),
                bytes: stored.get(&entry).copied(),
                compiled: 0,
                loaded: 0,
                compiling: Duration::ZERO,
                loading: Duration::ZERO,
                source: None,
            });
            match trip.outcome {
                CompilationOutcome::Compiled { duration } => {
                    row.compiled += 1;
                    row.compiling += duration;
                }
                CompilationOutcome::Loaded { duration } => {
                    row.loaded += 1;
                    row.loading += duration;
                }
            }
            if trip.source.is_some() {
                row.source.clone_from(&trip.source);
            }
        }
        let unrecorded = stored.keys().filter(|key| !rows.contains_key(key)).count() as u64;
        let kernels: Vec<KernelRow> = rows.into_values().collect();

        let mut families = BTreeMap::<&str, FamilyRow>::new();
        for row in &kernels {
            let family = families
                .entry(row.kernel.as_str())
                .or_insert_with(|| FamilyRow {
                    kernel: row.kernel.clone(),
                    instances: 0,
                    compiled: 0,
                    loaded: 0,
                    compiling: Duration::ZERO,
                    loading: Duration::ZERO,
                    bytes: 0,
                });
            family.instances += 1;
            family.compiled += row.compiled;
            family.loaded += row.loaded;
            family.compiling += row.compiling;
            family.loading += row.loading;
            family.bytes += row.bytes.unwrap_or_default();
        }
        let mut families: Vec<FamilyRow> = families.into_values().collect();
        families.sort_by(|a, b| b.compiling.cmp(&a.compiling).then(a.kernel.cmp(&b.kernel)));

        let mut report = Self {
            kernels,
            families,
            unrecorded,
        };
        report.sort(KernelOrder::default());
        report
    }

    /// Put the kernels in `order`.
    pub fn sort(&mut self, order: KernelOrder) {
        match order {
            KernelOrder::Compile => self
                .kernels
                .sort_by(|a, b| b.compiling.cmp(&a.compiling).then(a.id.cmp(&b.id))),
            KernelOrder::Size => self
                .kernels
                .sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.id.cmp(&b.id))),
            KernelOrder::Name => self
                .kernels
                .sort_by(|a, b| a.kernel.cmp(&b.kernel).then(a.id.cmp(&b.id))),
        }
    }

    /// Everything compiling cost, over every instance.
    pub fn compiling(&self) -> Duration {
        self.kernels.iter().map(|row| row.compiling).sum()
    }
}
