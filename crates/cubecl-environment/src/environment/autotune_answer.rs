//! The answers of an autotune table, outside of the environment that held
//! them.

use alloc::vec::Vec;

use crate::bytes::Bytes;
use crate::persistence::TuneTableName;
use crate::persistence::decode;
#[cfg(native_cache)]
use crate::persistence::encode;

/// The answers one autotune table holds, trimmed to their picks: what another
/// environment needs of the table to run the same picks, without the
/// measurements they were picked from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TunedTable {
    /// The table that holds them.
    pub table: TuneTableName,
    /// Its answers, one per key.
    pub answers: Vec<AutotuneAnswer>,
}

/// One answer of an autotune table, trimmed to its pick.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AutotuneAnswer {
    /// The key it answers and the checksum of the candidate list it was picked
    /// from, as the table stores them.
    pub key: Bytes,
    /// The index of the pick in that list.
    pub fastest_index: usize,
}

impl AutotuneAnswer {
    /// The answer a table stores as `value` under `key`, or `None` when the
    /// value does not decode.
    pub(crate) fn stored(key: &[u8], value: &[u8]) -> Option<Self> {
        let StoredPick { fastest_index } = decode(value)?;
        Some(Self {
            key: Bytes::from_bytes_vec(key.to_vec()),
            fastest_index,
        })
    }

    /// The value its table stores for it: a stored answer without its
    /// measurements.
    #[cfg(native_cache)]
    pub(crate) fn stored_value(&self) -> Bytes {
        let Self { fastest_index, .. } = *self;
        encode(&StoredPick { fastest_index })
    }
}

/// The part of a stored answer that names its pick: the rest of what a table
/// stores says how the pick was measured.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StoredPick {
    /// The index of the pick in its candidate list.
    pub fastest_index: usize,
}
