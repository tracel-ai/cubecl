//! What the work used, recorded per build in the environment it ran in.
//!
//! A tune races every candidate it measures, and each race fills the
//! environment with what only that candidate needs. What the work itself
//! uses is recorded apart, under the build that ran it, so a prune keeps
//! what the work will use again and removes the rest.

use super::BuildStamp;
#[cfg(any(native_cache, browser_cache))]
use super::PruneTally;
use crate::persistence::{Namespace, Store, StoreKey};
use alloc::string::String;

#[cfg(any(native_cache, browser_cache))]
use crate::collections::{HashMap, HashSet};
#[cfg(any(native_cache, browser_cache))]
use crate::persistence::{Database, encode};
#[cfg(any(native_cache, browser_cache))]
use alloc::vec::Vec;
#[cfg(any(native_cache, browser_cache))]
use serde::Serialize;

#[cfg(any(native_cache, browser_cache))]
mod tune_pruner;
#[cfg(any(native_cache, browser_cache))]
use tune_pruner::TunePruner;

#[cfg(all(test, native_cache))]
mod medium_tests;

/// The version every namespace this build writes carries, as
/// [`Namespace::scoped`] injects it.
#[cfg(any(native_cache, browser_cache))]
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The root the kernels the work launched were recorded under before the
/// uses had a root of their own: read only to be removed.
#[cfg(any(native_cache, browser_cache))]
const LEGACY_KERNELS_ROOT: &str = "launched";

/// The last segment of the namespace recording the kernels.
const KERNELS: &str = "kernels";

/// The segment before the namespace whose entries a record names.
const ENTRIES: &str = "entries";

/// What a use of the work names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UsedContent {
    /// Compiled kernels, by the stable hash of their id, whatever variant
    /// they were compiled in.
    Kernels,
    /// The entries of one namespace, by their key.
    Entries {
        /// The namespace whose entries the work used, in full.
        namespace: String,
    },
}

impl UsedContent {
    /// The first segment of every namespace a use is recorded under.
    pub const ROOT: &'static str = "used";

    /// The namespace `build` records its uses of this content under:
    /// `used/<version>/<build>/kernels`, or
    /// `used/<version>/<build>/entries/<namespace>`.
    pub fn namespace(&self, build: BuildStamp) -> Namespace {
        match self {
            Self::Kernels => Namespace::scoped(Self::ROOT, alloc::format!("{build}/{KERNELS}")),
            Self::Entries { namespace } => {
                Namespace::scoped(Self::ROOT, alloc::format!("{build}/{ENTRIES}/{namespace}"))
            }
        }
    }
}

impl UsedContent {
    /// The content the segments after a record's build name, or `None` when
    /// they name none.
    #[cfg(any(native_cache, browser_cache))]
    fn from_segments(segments: &str) -> Option<Self> {
        match segments {
            KERNELS => Some(Self::Kernels),
            _ => segments
                .strip_prefix(ENTRIES)
                .and_then(|rest| rest.strip_prefix('/'))
                .map(|namespace| Self::Entries {
                    namespace: namespace.into(),
                }),
        }
    }
}

/// One build's record of what its work used of one [content](UsedContent),
/// in the environment it runs in.
///
/// # Assumptions
///
/// - `K` encodes as the keys of what it records: for the kernels, the stable
///   hash of their id; for the entries of a namespace, the key type of that
///   namespace's store.
/// - The caller records only the work's uses, never a race's.
/// - The caller holds any switch that turns recording off: it builds the
///   store only when recording is on.
///
/// # Guarantees
///
/// - A key is written at most once per environment, so a warm start only
///   reads.
/// - A refused write is logged, never raised: a lost record costs a recompile
///   or a retune after a prune, never a result, and the next record of that
///   key tries again.
/// - When the store was opened with
///   [`StoreOptions::storage`](crate::persistence::StoreOptions::storage), it
///   follows environment switches as a [`Store`] does.
#[derive(Debug)]
pub struct WorkUses<K> {
    /// Lazy: it memoizes each key it finds recorded, and resets itself when
    /// the environment switches.
    store: Store<K, ()>,
}

impl<K: StoreKey> WorkUses<K> {
    /// The record kept in `store`, which the caller opened lazily over the
    /// namespace [`UsedContent::namespace`] names.
    pub fn from_store(store: Store<K, ()>) -> Self {
        Self { store }
    }

    /// Record that the work used `key`. A key recorded already, by this
    /// process or another, is read once and not written again; `key` is
    /// cloned only when it is not recorded yet.
    pub fn record(&mut self, key: &K) {
        // `get_mut`, not `get`: a lazy store reads through to its storage
        // only there, and memoizes what it finds.
        if self.store.get_mut(key).is_some() {
            return;
        }
        write(&mut self.store, key);
    }
}

/// Write `key` into `store`, logging a refusal.
fn write<K: StoreKey>(store: &mut Store<K, ()>, key: &K) {
    if let Err(error) = store.insert(key.clone(), ()) {
        log::debug!("a use of the work was not recorded: {}", error.reason());
    }
}

/// What a database records of the work of one build: the keys it recorded
/// as used, by content, and the records other builds left.
///
/// # Guarantees
///
/// - It holds exactly the keys recorded under the build it was read for by
///   this cubecl version: a record is this build's when both its version
///   and its build are, and a record of another version is another build's.
/// - Only the namespaces under [`UsedContent::ROOT`] are read as uses. Those
///   under the root the kernels were recorded under before are other builds'
///   records of their kernels, read only to be purged, and no other
///   namespace is read.
/// - A row of this build's kernels record counts as a recorded use, whether
///   or not it decodes as a key.
#[cfg(any(native_cache, browser_cache))]
#[derive(Debug)]
pub struct RecordedUses {
    /// This build's recorded keys, by content, as their store encoded them:
    /// read by content (Keyed) and asked whether a key is in them
    /// (Membership). A content is a key only when it records a use.
    uses: HashMap<UsedContent, HashSet<Vec<u8>>>,
    /// Other builds' namespaces, by the content they record: read by content
    /// (Keyed) and purged in the order they were found (Arrival).
    other_builds: HashMap<UsedContent, Vec<String>>,
}

#[cfg(any(native_cache, browser_cache))]
impl RecordedUses {
    /// What `database` records of the work of `build`.
    pub fn read(database: &Database, build: BuildStamp) -> Self {
        let mut uses: HashMap<UsedContent, HashSet<Vec<u8>>> = HashMap::default();
        let mut other_builds: HashMap<UsedContent, Vec<String>> = HashMap::default();
        let records = database.namespaces().into_iter().filter_map(|namespace| {
            UseRecord::of(&namespace, build).map(|record| (namespace, record))
        });
        for (namespace, record) in records {
            match record {
                UseRecord::ThisBuild(content) => {
                    read_keys(database, &namespace, uses.entry(content).or_default())
                }
                UseRecord::OtherBuild(content) => {
                    other_builds.entry(content).or_default().push(namespace)
                }
            }
        }
        Self { uses, other_builds }
    }

    /// Whether this build recorded any use of `content`.
    pub fn recorded(&self, content: &UsedContent) -> bool {
        self.uses.contains_key(content)
    }

    /// Whether this build recorded a use of `key` of `content`, `key`
    /// encoded as a [`Store`] encodes its keys.
    pub fn used<K: Serialize>(&self, content: &UsedContent, key: &K) -> bool {
        holds(self.uses.get(content), key)
    }

    /// Remove from `database`, in each autotune table this build's work used,
    /// the tunes it did not use, with the records other builds left of
    /// that table, and tally the tunes.
    ///
    /// # Guarantees
    ///
    /// - A table this build's work never used stays whole, with every record
    ///   of it: another hardware's, another cubecl version's, a tuner's this
    ///   build never ran.
    /// - When this build recorded no use of any table, nothing is removed.
    /// - No row is decoded: a tune is told by its key as its table's store
    ///   encoded it.
    pub fn prune_unused_tunes(&self, database: &Database) -> PruneTally {
        TunePruner::new(self).prune_used_tables(database)
    }

    /// Remove from `database` every tune this build's work did
    /// not use, in every table, and count the tunes left: what a save of
    /// the used tunes leaves in its copy.
    #[cfg(native_cache)]
    pub(crate) fn keep_only_used_tunes(&self, database: &Database) -> u64 {
        TunePruner::new(self).keep_only_used(database)
    }

    /// Delete from `database` the records other builds left of `content`:
    /// exactly their namespaces, never one that only starts like them.
    pub fn purge_other_builds(&self, database: &Database, content: &UsedContent) {
        let Some(namespaces) = self.other_builds.get(content) else {
            return;
        };
        for namespace in namespaces {
            database.purge(namespace);
        }
    }
}

/// Whether `keys` holds `key`, encoded as a [`Store`] encodes it.
#[cfg(any(native_cache, browser_cache))]
fn holds<K: Serialize>(keys: Option<&HashSet<Vec<u8>>>, key: &K) -> bool {
    keys.is_some_and(|keys| keys.contains(&encode(key)[..]))
}

/// Add to `keys` every key the rows of `namespace` hold, as stored.
#[cfg(any(native_cache, browser_cache))]
fn read_keys(database: &Database, namespace: &str, keys: &mut HashSet<Vec<u8>>) {
    database.scan(namespace, &mut |key, _| {
        keys.insert(key.to_vec());
    });
}

/// Whose record of the work a namespace is, read off its name.
#[cfg(any(native_cache, browser_cache))]
#[derive(Debug, PartialEq, Eq)]
enum UseRecord {
    /// The record of the build read for, of uses of the content.
    ThisBuild(UsedContent),
    /// The record of another build, or of another version.
    OtherBuild(UsedContent),
}

#[cfg(any(native_cache, browser_cache))]
impl UseRecord {
    /// Whose record `namespace` is, read for `build`: `None` when it is not
    /// a record of the work.
    fn of(namespace: &str, build: BuildStamp) -> Option<Self> {
        let mut segments = namespace.splitn(4, '/');
        let root = segments.next()?;
        if root == LEGACY_KERNELS_ROOT {
            return Some(Self::OtherBuild(UsedContent::Kernels));
        }
        if root != UsedContent::ROOT {
            return None;
        }
        let (version, stamp) = (segments.next()?, segments.next()?);
        let content = UsedContent::from_segments(segments.next()?)?;
        match version == VERSION && stamp.parse() == Ok(build) {
            true => Some(Self::ThisBuild(content)),
            false => Some(Self::OtherBuild(content)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::{
        CacheOption, Insertion, MemoryStorage, Origin, Storage, StoreOptions,
    };
    use crate::sync::Arc;
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use core::sync::atomic::{AtomicUsize, Ordering};

    const BUILD: BuildStamp = BuildStamp { hash: 7 };

    #[test]
    fn kernels_are_recorded_under_the_build() {
        let version = env!("CARGO_PKG_VERSION");

        assert_eq!(
            UsedContent::Kernels.namespace(BUILD).as_str(),
            alloc::format!("used/{version}/{BUILD}/kernels")
        );
    }

    #[test]
    fn entries_are_recorded_under_the_build_and_their_namespace() {
        let version = env!("CARGO_PKG_VERSION");
        let content = UsedContent::Entries {
            namespace: "autotune/0.1.0/matmul".to_string(),
        };

        assert_eq!(
            content.namespace(BUILD).as_str(),
            alloc::format!("used/{version}/{BUILD}/entries/autotune/0.1.0/matmul")
        );
    }

    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn this_builds_kernels_record_reads_as_its_own() {
        let namespace = UsedContent::Kernels.namespace(BUILD);

        assert_eq!(
            UseRecord::of(namespace.as_str(), BUILD),
            Some(UseRecord::ThisBuild(UsedContent::Kernels))
        );
    }

    /// The remainder after `entries/` is the whole namespace, `/` and all.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn an_entries_record_reads_back_its_whole_namespace() {
        let content = UsedContent::Entries {
            namespace: "autotune/0.1.0/layout-v1/matmul".to_string(),
        };
        let namespace = content.namespace(BUILD);

        assert_eq!(
            UseRecord::of(namespace.as_str(), BUILD),
            Some(UseRecord::ThisBuild(content))
        );
    }

    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn another_builds_record_is_not_this_builds() {
        let namespace = UsedContent::Kernels.namespace(BuildStamp { hash: 8 });

        assert_eq!(
            UseRecord::of(namespace.as_str(), BUILD),
            Some(UseRecord::OtherBuild(UsedContent::Kernels))
        );
    }

    /// The same build stamp under another cubecl version is another build.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn another_versions_record_is_another_builds() {
        let namespace = alloc::format!("used/0.0.0-other/{BUILD}/kernels");

        assert_eq!(
            UseRecord::of(&namespace, BUILD),
            Some(UseRecord::OtherBuild(UsedContent::Kernels))
        );
    }

    /// The records written before the uses had a root of their own are
    /// other builds' records of their kernels, whatever build they name.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn a_legacy_record_is_another_builds_kernels_record() {
        let version = env!("CARGO_PKG_VERSION");
        let namespace = alloc::format!("launched/{version}/{BUILD}");

        assert_eq!(
            UseRecord::of(&namespace, BUILD),
            Some(UseRecord::OtherBuild(UsedContent::Kernels))
        );
    }

    /// A namespace under another root is no record, even one whose root
    /// starts like the uses' root.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn a_namespace_under_another_root_is_no_record() {
        let version = env!("CARGO_PKG_VERSION");
        for namespace in [
            alloc::format!("autotune/{version}/{BUILD}/kernels"),
            alloc::format!("used-by-an-application/{version}/{BUILD}/kernels"),
        ] {
            assert_eq!(UseRecord::of(&namespace, BUILD), None, "{namespace}");
        }
    }

    /// A namespace under the uses' root that names no content is no record.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn a_record_naming_no_content_is_no_record() {
        let version = env!("CARGO_PKG_VERSION");
        for namespace in [
            alloc::format!("used/{version}/{BUILD}"),
            alloc::format!("used/{version}/{BUILD}/tunes"),
            alloc::format!("used/{version}/{BUILD}/entries"),
        ] {
            assert_eq!(UseRecord::of(&namespace, BUILD), None, "{namespace}");
        }
    }

    /// A key past 64 bits matches as the store that recorded it encoded it.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn a_wide_key_is_used_as_its_store_wrote_it() {
        let wide: u128 = (1 << 70) | 3;
        let memory = MemoryStorage::new("work-uses-wide-key");
        memory.purge();
        let mut store: Store<u128, ()> =
            Store::new(StoreOptions::new().storage_with(Box::new(memory.clone()), "wide"));
        store.insert(wide, ()).unwrap();
        let mut keys = HashSet::default();
        memory.scan(&mut |key, _| {
            keys.insert(key.to_vec());
        });
        let uses = RecordedUses {
            uses: [(UsedContent::Kernels, keys)].into_iter().collect(),
            other_builds: HashMap::default(),
        };

        assert!(uses.used(&UsedContent::Kernels, &wide));
        assert!(!uses.used(&UsedContent::Kernels, &(wide ^ (1 << 70))));
    }

    /// A content this build recorded no use of holds no used key.
    #[cfg(any(native_cache, browser_cache))]
    #[test]
    fn a_content_with_no_record_uses_nothing() {
        let uses = RecordedUses {
            uses: HashMap::default(),
            other_builds: HashMap::default(),
        };

        assert!(!uses.recorded(&UsedContent::Kernels));
        assert!(!uses.used(&UsedContent::Kernels, &1_u128));
    }

    /// However often the work uses a key, this process writes it once.
    #[test]
    fn a_key_used_again_is_not_written_again() {
        let storage = Counting::new("work-uses-written-once", false);
        let mut uses = storage.uses();

        uses.record(&1_u128);
        uses.record(&1_u128);
        uses.record(&1_u128);

        assert_eq!(storage.writes(), 1);
    }

    /// A process that finds a key recorded already writes nothing: a warm
    /// start only reads.
    #[test]
    fn a_key_recorded_by_another_process_is_only_read() {
        let storage = Counting::new("work-uses-warm-start", false);
        storage.uses().record(&1_u128);

        storage.uses().record(&1_u128);

        assert_eq!(storage.writes(), 1);
    }

    /// A refused write is no panic, and the next record of the key tries
    /// again.
    #[test]
    fn a_refused_record_is_tried_again() {
        let storage = Counting::new("work-uses-refused", true);
        let mut uses = storage.uses();

        uses.record(&1_u128);
        uses.record(&1_u128);

        assert_eq!(storage.writes(), 2);
    }

    /// A storage over shared memory, counting the writes it is asked for,
    /// and refusing each one when it is told to.
    #[derive(Debug, Clone)]
    struct Counting {
        memory: MemoryStorage,
        writes: Arc<AtomicUsize>,
        refusing: bool,
    }

    impl Counting {
        fn new(namespace: &str, refusing: bool) -> Self {
            let memory = MemoryStorage::new(namespace);
            memory.purge();
            Self {
                memory,
                writes: Arc::new(AtomicUsize::new(0)),
                refusing,
            }
        }

        /// A record over this storage, as a process opening it would.
        fn uses(&self) -> WorkUses<u128> {
            WorkUses::from_store(Store::new(
                StoreOptions::new()
                    .storage_with(
                        Box::new(self.clone()),
                        UsedContent::Kernels.namespace(BUILD),
                    )
                    .cache(CacheOption::Lazy),
            ))
        }

        fn writes(&self) -> usize {
            self.writes.load(Ordering::Relaxed)
        }
    }

    impl Storage for Counting {
        fn get(&self, key: &[u8]) -> Option<crate::bytes::Bytes> {
            self.memory.get(key)
        }

        fn insert(&self, key: &[u8], value: crate::bytes::Bytes, origin: Origin) -> Insertion {
            self.writes.fetch_add(1, Ordering::Relaxed);
            match self.refusing {
                true => Insertion::Failed("refused".to_string()),
                false => self.memory.insert(key, value, origin),
            }
        }

        fn replace(&self, key: &[u8], value: crate::bytes::Bytes, origin: Origin) -> Insertion {
            self.insert(key, value, origin)
        }

        fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
            self.memory.scan(visit)
        }

        fn purge(&self) {
            self.memory.purge()
        }

        fn purge_key(&self, key: &[u8]) {
            self.memory.purge_key(key)
        }

        fn describe(&self) -> String {
            self.memory.describe()
        }
    }
}
