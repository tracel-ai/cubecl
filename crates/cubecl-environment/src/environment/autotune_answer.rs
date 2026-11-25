//! One autotune answer, outside of the environment that held it.

use alloc::string::{String, ToString};

use crate::bytes::Bytes;
use crate::persistence::decode;
#[cfg(native_cache)]
use crate::persistence::encode;

/// One answer of an autotune table, trimmed to its pick: what another
/// environment needs of it to run the same pick, without the measurements it
/// was picked from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AutotuneAnswer {
    /// The table that holds it, by its whole namespace: the cubecl version and
    /// the table layout, the hardware and the tuner.
    pub table: String,
    /// The key it answers and the checksum of the candidate list it was picked
    /// from, as the table stores them.
    pub key: Bytes,
    /// The index of the pick in that list.
    pub fastest_index: usize,
}

impl AutotuneAnswer {
    /// The answer `table` stores as `value` under `key`, or `None` when the
    /// value does not decode.
    pub(crate) fn stored(table: &str, key: &[u8], value: &[u8]) -> Option<Self> {
        let StoredPick { fastest_index } = decode(value)?;
        Some(Self {
            table: table.to_string(),
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
