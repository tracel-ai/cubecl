#[cfg(persistence)]
use alloc::string::String;
#[cfg(persistence)]
use alloc::vec::Vec;
#[cfg(persistence)]
use cubecl_environment::persistence::{Namespace, NamespaceRoot};
#[cfg(persistence)]
use serde::{Deserialize, Serialize};

#[cfg(persistence)]
use crate::namespace_version::NamespaceVersion;
#[cfg(persistence)]
use crate::tune::AutotuneResult;

/// The layout version of the tables: how they are named and what their rows hold, bumped when
/// either changes.
///
/// Version 2 stores `decided_index` and `threshold_fraction` where version 1 stored
/// `fastest_index` and `threshold`: a table of version 1 is dropped rather than misread.
#[cfg(persistence)]
pub static TABLE_LAYOUT_VERSION: NamespaceVersion =
    NamespaceVersion::new(NamespaceRoot::AUTOTUNE, LAYOUT_PREFIX, 2);

/// The table entry an answer is stored under.
#[cfg(persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub struct PersistentCacheKey<K> {
    /// The autotune key identifying the operation.
    pub key: K,
    /// The checksum of the candidate list the key was tuned under: an answer to another list is
    /// an answer to another question.
    pub checksum: String,
}

/// The answer a table stores for one entry.
///
/// Only [`decided_index`](Self::decided_index) is read back: a look-up seeds the in-memory cache
/// from it and nothing else. Everything below it is stored so a cache entry can be inspected after
/// the fact — why a kernel won, against which measurements, and under which bounds — which is the
/// question that cannot be answered from a live process once tuning is over. That is also why the
/// type is `pub`: reading an entry back is the point.
#[cfg(persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PersistentCacheValue {
    /// The index of the candidate the key runs: the one the tune decided, which a round robin or
    /// a short circuit may pick over the best-scored one.
    pub decided_index: usize,
    /// Every candidate's result, measured ones first, fastest first (Arrival Usage).
    pub results: Vec<AutotuneResult>,
    /// The bounds the tune was measured against, where the set has a bounds generator.
    pub bounds: Option<crate::tune::Bounds>,
    /// The time limit the bounds set, if any.
    pub limit: Option<core::time::Duration>,
}

/// The table a [`TuneTracker`](crate::tune::cache::tracker::TuneTracker) stores its answers in,
/// whether or not it persists them.
#[cfg(persistence)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneTable {
    /// The namespace the table is stored under: what a [record](crate::tune::TuneRecord) names it by.
    pub namespace: String,
    /// The environment generation the in-memory answers were built under, or last reset to; see
    /// [`cubecl_environment::environment::generation`].
    pub generation: u32,
}

/// What the segment numbering a table layout starts with, as in `layout-v1`.
pub const LAYOUT_PREFIX: &str = "layout-v";

/// What names the table one tuner stores its answers in on one hardware.
///
/// # Assumptions
///
/// - Neither name holds a `/`: the namespace joins them with one.
#[cfg(persistence)]
#[derive(Clone, Copy, Debug)]
pub struct TuneTableName<'a> {
    /// The hardware the answers were measured on, as its
    /// [`HardwareIdentity`](crate::device::HardwareIdentity) displays.
    pub hardware: &'a str,
    /// The tuner, named by the code that declared it.
    pub tuner: &'a str,
    /// Whether the answers were measured with
    /// [register spills refused](crate::config::compilation::CompilationConfig::refuse_register_spills),
    /// which drops the candidates that spill from the race.
    pub refuses_register_spills: bool,
}

#[cfg(persistence)]
impl TuneTableName<'_> {
    /// The namespace the table is stored under.
    ///
    /// # Guarantees
    ///
    /// - It reads `autotune/<version>/layout-v<layout>/<hardware>/<tuner>`, the layout a reader
    ///   of the environment parses.
    /// - With spills refused, the tuner segment ends in `@no-spills`: an answer a spilling
    ///   candidate won is never served to a process that refuses it.
    pub fn namespace(&self) -> Namespace {
        let Self {
            hardware,
            tuner,
            refuses_register_spills,
        } = self;
        let policy = if *refuses_register_spills {
            NO_SPILLS_SUFFIX
        } else {
            ""
        };
        TABLE_LAYOUT_VERSION.namespace(&alloc::format!("{hardware}/{tuner}{policy}"))
    }
}

/// What a table's tuner segment ends with when its answers were measured with spills refused.
#[cfg(persistence)]
const NO_SPILLS_SUFFIX: &str = "@no-spills";

#[cfg(all(test, persistence))]
mod tests {
    use super::*;

    #[test]
    fn refused_spills_name_a_table_of_their_own() {
        let name = |refuses_register_spills| {
            TuneTableName {
                hardware: "gfx1151",
                tuner: "matmul",
                refuses_register_spills,
            }
            .namespace()
        };

        assert_ne!(name(true).as_str(), name(false).as_str());
        assert_eq!(
            name(false).as_str(),
            TABLE_LAYOUT_VERSION.namespace("gfx1151/matmul").as_str(),
            "the table of answers with spills accepted keeps its name"
        );
    }
}
