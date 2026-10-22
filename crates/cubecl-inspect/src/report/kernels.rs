use cubecl_server::compiler::{CompilationOutcome, CompilationRecord, KernelCacheKey};
use serde::{Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::AddAssign;
use std::time::Duration;

/// Every kernel the environment's sessions compiled or loaded, and what that
/// cost, per instance and per kernel type.
#[derive(Clone, Debug, Serialize)]
pub struct KernelReport {
    pub kernels: Vec<KernelRow>,
    /// The instances folded by kernel type, costliest first.
    pub families: Vec<FamilyRow>,
    /// Artifacts the file stores with no recorded trip: compiled while
    /// nothing was recorded.
    pub unrecorded: u64,
}

/// The artifacts a file's compilation store holds, by the key naming each,
/// with their size.
pub type StoredArtifacts = BTreeMap<KernelCacheKey, u64>;

/// A 128-bit hash naming part of a kernel's artifact, written once for every
/// such hash so they all print, serialize and match alike: twelve hex digits
/// in text, enough to tell a file's kernels apart and short enough to type;
/// in full in JSON; matched by a prefix of the hex, the way a command line
/// names one.
macro_rules! artifact_hash {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u128);

        impl $name {
            /// Whether the hash's hex rendering starts with `prefix`.
            pub fn matches(&self, prefix: &str) -> bool {
                format!("{:032x}", self.0).starts_with(&prefix.to_ascii_lowercase())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", &format!("{:032x}", self.0)[..12])
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_str(&format_args!("{:032x}", self.0))
            }
        }
    };
}

artifact_hash!(
    /// A stable handle on one kernel instance: the hash of its id that names
    /// its artifact in the compilation store.
    KernelHash
);

artifact_hash!(
    /// The hash of the build that compiled or loaded a kernel instance: with
    /// the [`KernelHash`], what names its artifact.
    BuildHash
);

/// What a command line names one kernel instance by: a prefix of its id and,
/// when several builds recorded it, a prefix of the build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelSelector<'a> {
    pub id: &'a str,
    pub build: Option<&'a str>,
}

/// How the instances a [`KernelSelector`] picks answer it.
///
/// A kernel is recorded once per build that compiled or loaded it, so after
/// a rebuild its id alone names several instances: the selector's build
/// names one of them.
#[derive(Clone, Debug)]
pub enum KernelMatch {
    /// No instance's id starts with the selector's.
    Unknown,
    One(KernelRow),
    /// One kernel, recorded by several builds.
    SeveralBuilds(Vec<KernelRow>),
    /// Instances of this many kernels: the id needs more of its digits.
    SeveralKernels(usize),
}

/// One kernel instance as one build recorded it, its trips folded.
#[derive(Clone, Debug, Serialize)]
pub struct KernelRow {
    pub id: KernelHash,
    pub build: BuildHash,
    /// The kernel's type, in full.
    pub kernel: String,
    /// The kernel as cubecl defined it, in the IR's textual form, when a
    /// fresh compile recorded it.
    pub ir: Option<String>,
    /// The size of its artifact in the store, when the file holds one.
    pub bytes: Option<u64>,
    /// Flattened, so a JSON reader finds `compiled` and `compiling` on the
    /// row.
    #[serde(flatten)]
    pub cost: CompilationCost,
    /// The source, when a record kept it.
    pub source: Option<String>,
}

/// One kernel type, its instances folded.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyRow {
    pub kernel: String,
    pub instances: u64,
    /// Flattened, like [`KernelRow::cost`].
    #[serde(flatten)]
    pub cost: CompilationCost,
    pub bytes: u64,
}

/// What obtaining kernels cost: how often each was compiled fresh or loaded
/// from the store, and the time each took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CompilationCost {
    /// Times compiled fresh.
    pub compiled: u64,
    /// Times loaded from the store.
    pub loaded: u64,
    /// Compiling, over every fresh compile.
    pub compiling: Duration,
    /// Loading, over every store load.
    pub loading: Duration,
}

/// The orders a [`KernelReport`] lists its kernels in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KernelOrder {
    /// Costliest to compile first.
    #[default]
    Compile,
    /// Largest artifact first.
    Size,
    /// By kernel type.
    Name,
}

impl KernelMatch {
    /// The instances of `rows` that `selector` names.
    pub fn new(selector: KernelSelector<'_>, rows: impl IntoIterator<Item = KernelRow>) -> Self {
        let mut rows: Vec<KernelRow> = rows
            .into_iter()
            .filter(|row| selector.selects(row))
            .collect();
        let kernels: BTreeSet<KernelHash> = rows.iter().map(|row| row.id).collect();
        match (rows.len(), kernels.len()) {
            (0, _) => Self::Unknown,
            (1, _) => Self::One(rows.remove(0)),
            (_, 1) => Self::SeveralBuilds(rows),
            (_, kernels) => Self::SeveralKernels(kernels),
        }
    }
}

impl KernelSelector<'_> {
    /// Whether `row` is an instance this names.
    pub fn selects(&self, row: &KernelRow) -> bool {
        row.id.matches(self.id) && self.build.is_none_or(|build| row.build.matches(build))
    }
}

/// The id, then `--build` and the build when one is named: what the command
/// line was given.
impl fmt::Display for KernelSelector<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)?;
        if let Some(build) = &self.build {
            write!(f, " --build {build}")?;
        }
        Ok(())
    }
}

impl KernelRow {
    /// The type's last path segment, without its generics: a name a table can
    /// print.
    pub fn short_name(&self) -> &str {
        short_name(&self.kernel)
    }

    /// What names the instance's artifact in the store.
    pub fn key(&self) -> KernelCacheKey {
        KernelCacheKey {
            id: self.id.0,
            build_id: self.build.0,
        }
    }
}

impl CompilationCost {
    /// Count one trip that ended in `outcome` after `work`.
    pub fn record(&mut self, outcome: CompilationOutcome, work: Duration) {
        match outcome {
            CompilationOutcome::Compiled => {
                self.compiled += 1;
                self.compiling += work;
            }
            // Moved under this key from the artifact another key stored:
            // obtained from the store, like a load, without compiling.
            CompilationOutcome::Loaded | CompilationOutcome::Rekeyed => {
                self.loaded += 1;
                self.loading += work;
            }
        }
    }
}

impl AddAssign for CompilationCost {
    fn add_assign(&mut self, other: Self) {
        self.compiled += other.compiled;
        self.loaded += other.loaded;
        self.compiling += other.compiling;
        self.loading += other.loading;
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
    pub fn new<'a>(
        trips: impl IntoIterator<Item = &'a CompilationRecord>,
        stored: &StoredArtifacts,
    ) -> Self {
        let mut rows = BTreeMap::<KernelCacheKey, KernelRow>::new();
        for trip in trips {
            let row = rows.entry(trip.key).or_insert_with(|| KernelRow {
                id: KernelHash(trip.key.id),
                build: BuildHash(trip.key.build_id),
                kernel: trip.kernel.clone(),
                ir: None,
                bytes: stored.get(&trip.key).copied(),
                cost: CompilationCost::default(),
                source: None,
            });
            row.cost.record(trip.outcome, trip.work);
            if trip.ir.is_some() {
                row.ir.clone_from(&trip.ir);
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
                    cost: CompilationCost::default(),
                    bytes: 0,
                });
            family.instances += 1;
            family.cost += row.cost;
            family.bytes += row.bytes.unwrap_or_default();
        }
        let mut families: Vec<FamilyRow> = families.into_values().collect();
        families.sort_by(|a, b| {
            b.cost
                .compiling
                .cmp(&a.cost.compiling)
                .then(a.kernel.cmp(&b.kernel))
        });

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
            KernelOrder::Compile => self.kernels.sort_by(|a, b| {
                b.cost
                    .compiling
                    .cmp(&a.cost.compiling)
                    .then(a.key().cmp(&b.key()))
            }),
            KernelOrder::Size => self
                .kernels
                .sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.key().cmp(&b.key()))),
            KernelOrder::Name => self
                .kernels
                .sort_by(|a, b| a.kernel.cmp(&b.kernel).then(a.key().cmp(&b.key()))),
        }
    }

    /// Everything compiling cost, over every instance.
    pub fn compiling(&self) -> Duration {
        self.kernels.iter().map(|row| row.cost.compiling).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prefix that names one kernel twice, once per build, asks for the
    /// build; one that names two kernels asks for more of the id.
    #[test]
    fn a_selector_matches_one_instance_or_says_what_it_lacks() {
        let rows = [row(0xab01, 1), row(0xab01, 2), row(0xcd02, 1)];
        let select = |id, build| KernelMatch::new(KernelSelector { id, build }, rows.clone());
        let full = format!("{:032x}", 0xab01u128);
        let one = format!("{:032x}", 1u128);

        assert!(matches!(select("ff", None), KernelMatch::Unknown));
        assert!(
            matches!(select(&full, Some(&one)), KernelMatch::One(row) if row.build == BuildHash(1))
        );
        assert!(matches!(select(&full, None), KernelMatch::SeveralBuilds(rows) if rows.len() == 2));
        assert!(matches!(select("", None), KernelMatch::SeveralKernels(2)));
    }

    fn row(id: u128, build: u128) -> KernelRow {
        KernelRow {
            id: KernelHash(id),
            build: BuildHash(build),
            kernel: "kernel".to_string(),
            ir: None,
            bytes: None,
            cost: CompilationCost {
                compiled: 1,
                ..CompilationCost::default()
            },
            source: None,
        }
    }
}
