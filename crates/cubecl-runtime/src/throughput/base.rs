use crate::throughput::{CmmaDims, ComputeCmmaConfig};
use alloc::{format, string::String};
use core::time::Duration;
use cubecl_ir::{ElemType, FloatKind};

/// Bytes per buffer of the single-size memory probes, [`ThroughputMode::Memory`]
/// and [`ThroughputMode::MemoryRead`]. Clamped to the device's maximum
/// allocation when the probe runs.
pub const DEFAULT_BUFFER_BYTES: u64 = 512 * 1024 * 1024;

/// Which directions of traffic a memory probe issues.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryAccess {
    /// Reads every byte and writes it back out. The ceiling for a kernel that
    /// both loads and stores.
    Copy,
    /// Reads only, storing nothing. The ceiling for a kernel that streams data
    /// it does not write back — a weight stream, a reduction, a gather. Such a
    /// kernel legitimately exceeds [`Copy`](Self::Copy), because half of the
    /// copy's traffic is a direction it never uses.
    Read,
}

impl MemoryAccess {
    /// How many buffers of equal size one pass touches: two for a copy (one in,
    /// one out), one for a read.
    pub const fn buffers(&self) -> u64 {
        match self {
            Self::Copy => 2,
            Self::Read => 1,
        }
    }

    /// The working set of the single-size probe for this access, in bytes moved
    /// per pass: [`DEFAULT_BUFFER_BYTES`] per buffer touched.
    pub const fn default_working_set(&self) -> u64 {
        DEFAULT_BUFFER_BYTES * self.buffers()
    }
}

/// Represents the mode of a throughput computation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ThroughputMode {
    /// Compute direct calculation without special hardware acceleration.
    ComputeDirect {
        /// The data type of the computation.
        dtype: ElemType,
    },
    /// Compute cmma calculation with CMMA hardware acceleration.
    ComputeCmma {
        /// The data type of the computation.
        dtype: ElemType,
        /// The configuration of the CMMA operation.
        config: ComputeCmmaConfig,
    },
    /// Memory input reads and output writes — a copy, at the default working
    /// set. `ops_count` counts both directions, so this is total traffic across
    /// the memory interface.
    ///
    /// Equivalent to [`MemoryWorkingSet`](Self::MemoryWorkingSet) with
    /// [`MemoryAccess::Copy`] at [`MemoryAccess::default_working_set`].
    Memory,
    /// Memory input reads only, no store, at the default working set. The
    /// ceiling for a kernel that streams data it does not write back — a weight
    /// stream, a reduction, a gather. Such a kernel can legitimately exceed
    /// [`Memory`](Self::Memory), which is why it needs its own probe rather than
    /// a correction factor.
    ///
    /// Equivalent to [`MemoryWorkingSet`](Self::MemoryWorkingSet) with
    /// [`MemoryAccess::Read`] at [`MemoryAccess::default_working_set`].
    MemoryRead,
    /// One point of a memory curve: `access` over a working set of exactly
    /// `bytes`.
    ///
    /// `bytes` is the traffic one pass moves across the interface, the same
    /// currency the measured rate is in — a [`Copy`](MemoryAccess::Copy) splits
    /// it evenly between two buffers of `bytes / 2`, a
    /// [`Read`](MemoryAccess::Read) moves all of it out of one buffer of
    /// `bytes`. So a kernel that touches N bytes asks about N regardless of
    /// which access it resembles.
    ///
    /// Small working sets report *cache* bandwidth rather than bus bandwidth,
    /// and a size too small to keep the interface busy reports less than the
    /// hardware can do. Neither is an error — it is what a kernel of that size
    /// can actually reach — but the two are not interchangeable, which is why
    /// [`MemoryCurve`](crate::throughput::MemoryCurve) hands out a rate and a
    /// [`MemoryRegime`](crate::throughput::MemoryRegime) together rather than a
    /// bare number.
    MemoryWorkingSet {
        /// Which directions of traffic the probe issues.
        access: MemoryAccess,
        /// The bytes moved in one pass.
        bytes: u64,
    },
    /// Launch overhead measurement.
    Launch,
}

impl ThroughputMode {
    /// The mode probing `access` over a working set of `bytes`.
    ///
    /// Folds onto [`Memory`](Self::Memory) and [`MemoryRead`](Self::MemoryRead)
    /// at their default working set: those are the same measurement, so a curve
    /// reuses the cache entry the single-size call already filled instead of
    /// measuring it a second time under a second key.
    pub const fn memory(access: MemoryAccess, bytes: u64) -> Self {
        if bytes == access.default_working_set() {
            match access {
                MemoryAccess::Copy => Self::Memory,
                MemoryAccess::Read => Self::MemoryRead,
            }
        } else {
            Self::MemoryWorkingSet { access, bytes }
        }
    }

    /// The access pattern and working set this mode probes, or `None` for the
    /// modes that don't measure memory.
    ///
    /// The one place the single-size variants are mapped onto their access and
    /// size, so nothing downstream has to repeat the equivalence.
    pub const fn memory_probe(&self) -> Option<(MemoryAccess, u64)> {
        match self {
            Self::Memory => Some((MemoryAccess::Copy, MemoryAccess::Copy.default_working_set())),
            Self::MemoryRead => {
                Some((MemoryAccess::Read, MemoryAccess::Read.default_working_set()))
            }
            Self::MemoryWorkingSet { access, bytes } => Some((*access, *bytes)),
            Self::ComputeDirect { .. } | Self::ComputeCmma { .. } | Self::Launch => None,
        }
    }
}

/// Represents a key/configuration used to identify the throughput of a computation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
// Reject cached entries from an older key layout instead of silently ignoring their extra fields.
#[cfg_attr(std_io, serde(deny_unknown_fields))]
pub struct ThroughputKey {
    /// The mode of the throughput computation.
    pub mode: ThroughputMode,
}

impl ThroughputKey {
    /// Returns the data type of the computation.
    pub fn dtype(&self) -> ElemType {
        match self.mode {
            ThroughputMode::ComputeDirect { dtype } => dtype,
            ThroughputMode::ComputeCmma { dtype, .. } => dtype,
            // For memory and launch throughput, we use a default element type (F32).
            ThroughputMode::Memory
            | ThroughputMode::MemoryRead
            | ThroughputMode::MemoryWorkingSet { .. }
            | ThroughputMode::Launch => ElemType::Float(FloatKind::F32),
        }
    }
}

/// Represents the throughput of a computation, including the number of operations and the duration.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct ThroughputValue {
    /// The number of operations performed depending of the mode during the computation.
    pub ops_count: usize,
    /// The duration of the computation.
    pub duration: Duration,
}

impl ThroughputValue {
    /// A zero-initialized throughput value, representing no operations or duration.
    pub const ZERO: Self = Self {
        ops_count: 0,
        duration: Duration::ZERO,
    };

    /// Returns the operations per second.
    pub fn ops_per_s(&self) -> f64 {
        if self.duration.is_zero() {
            return f64::NAN;
        }
        self.ops_count as f64 / self.duration.as_secs_f64()
    }

    /// Returns the bytes per second.
    pub fn bytes_per_s(&self, key: &ThroughputKey) -> f64 {
        if self.duration.is_zero() {
            return f64::NAN;
        }
        (self.ops_count * key.dtype().size()) as f64 / self.duration.as_secs_f64()
    }

    /// Returns the duration per operation.
    pub fn duration_per_op(&self) -> Duration {
        if self.ops_count == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(self.duration.as_secs_f64() / self.ops_count as f64)
        }
    }

    /// Formats the throughput value as a clean human-readable string.
    pub fn format(&self, key: &ThroughputKey) -> String {
        let (mut val_per_s, unit) = match key.mode {
            ThroughputMode::ComputeDirect { .. } | ThroughputMode::ComputeCmma { .. } => {
                (self.ops_per_s(), "OPS")
            }
            ThroughputMode::Memory
            | ThroughputMode::MemoryRead
            | ThroughputMode::MemoryWorkingSet { .. } => (self.bytes_per_s(key), "bytes"),
            ThroughputMode::Launch => {
                let dur = self.duration_per_op();
                if dur.is_zero() {
                    return String::from("N/A");
                }
                return format!("{dur:?}/launch");
            }
        };

        if val_per_s.is_nan() {
            return String::from("N/A");
        }

        let suffixes = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"];
        let mut suffix_idx = 0;

        for _ in 0..suffixes.len() - 1 {
            if val_per_s < 1000.0 {
                break;
            }
            val_per_s /= 1000.0;
            suffix_idx += 1;
        }

        format!("{val_per_s:.4} {}{unit}/s", suffixes[suffix_idx])
    }
}

/// Constructs a compute [`ThroughputKey`] based on CMMA tile availability and types.
pub fn compute_throughput_key(
    cmma_tile: Option<(u32, u32, u32)>,
    input_elem_type: ElemType,
    acc_elem_type: ElemType,
) -> ThroughputKey {
    let mode = match cmma_tile {
        Some((tile_m, tile_n, tile_k)) => ThroughputMode::ComputeCmma {
            dtype: input_elem_type,
            config: ComputeCmmaConfig {
                accumulator_type: acc_elem_type,
                cmma_dims: CmmaDims {
                    m: tile_m as usize,
                    n: tile_n as usize,
                    k: tile_k as usize,
                },
            },
        },
        None => ThroughputMode::ComputeDirect {
            dtype: acc_elem_type,
        },
    };

    ThroughputKey { mode }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_folds_the_default_working_set_onto_the_single_size_modes() {
        // A curve reaching the default working set must land on the same key
        // the single-size call uses, or it measures what is already cached.
        let copy = MemoryAccess::Copy.default_working_set();
        let read = MemoryAccess::Read.default_working_set();

        assert_eq!(
            ThroughputMode::memory(MemoryAccess::Copy, copy),
            ThroughputMode::Memory
        );
        assert_eq!(
            ThroughputMode::memory(MemoryAccess::Read, read),
            ThroughputMode::MemoryRead
        );

        // A copy moves the same bytes twice, so the two accesses reach their
        // default at different working sets and must not fold onto each other.
        assert_eq!(copy, 2 * read);
        assert_eq!(
            ThroughputMode::memory(MemoryAccess::Copy, read),
            ThroughputMode::MemoryWorkingSet {
                access: MemoryAccess::Copy,
                bytes: read,
            }
        );
    }

    #[test]
    fn memory_probe_reports_the_access_and_working_set() {
        // Every memory mode describes a probe, and no other mode does.
        for mode in [
            ThroughputMode::Memory,
            ThroughputMode::MemoryRead,
            ThroughputMode::MemoryWorkingSet {
                access: MemoryAccess::Read,
                bytes: 4096,
            },
        ] {
            let (access, bytes) = mode.memory_probe().expect("A memory mode");
            assert_eq!(ThroughputMode::memory(access, bytes), mode);
        }

        assert_eq!(ThroughputMode::Launch.memory_probe(), None);
        assert_eq!(
            ThroughputMode::ComputeDirect {
                dtype: ElemType::Float(FloatKind::F32)
            }
            .memory_probe(),
            None
        );
    }

    /// The throughput cache keys on the serialized key, so a layout change
    /// drops every measurement users already paid for. Adding a variant must
    /// leave the existing ones encoded exactly as before.
    #[cfg(std_io)]
    #[test]
    fn existing_keys_keep_their_serialized_form() {
        let encode = |mode| serde_json::to_string(&ThroughputKey { mode }).unwrap();

        assert_eq!(encode(ThroughputMode::Memory), r#"{"mode":"Memory"}"#);
        assert_eq!(
            encode(ThroughputMode::MemoryRead),
            r#"{"mode":"MemoryRead"}"#
        );
        assert_eq!(encode(ThroughputMode::Launch), r#"{"mode":"Launch"}"#);

        // And an entry written before the variant existed still reads back.
        let cached: ThroughputKey = serde_json::from_str(r#"{"mode":"Memory"}"#).unwrap();
        assert_eq!(cached.mode, ThroughputMode::Memory);
    }
}
