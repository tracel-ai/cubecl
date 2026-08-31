use crate::throughput::{CmmaDims, ComputeCmmaConfig};
use alloc::{format, string::String};
use core::time::Duration;
use cubecl_ir::{ElemType, FloatKind};

/// Bytes per buffer of a [`ThroughputMode::Memory`] probe left at its default
/// working set. Clamped to the device's maximum allocation when the probe runs.
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
    /// Writes only, reading nothing at the software level. The ceiling for a
    /// kernel that streams stores it never reads back: an RNG fill, a memset,
    /// a broadcast. Ordinary stores still carry read-for-ownership traffic on
    /// cache-coherent hardware, and this probe's stores do too, which is what
    /// makes it the honest ceiling for a kernel that uses ordinary stores
    /// rather than a non-temporal one.
    Write,
}

impl MemoryAccess {
    /// How many buffers of equal size one pass touches: two for a copy (one in,
    /// one out), one for a read or a write.
    pub const fn buffers(&self) -> u64 {
        match self {
            Self::Copy => 2,
            Self::Read | Self::Write => 1,
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
    /// Traffic across the memory interface, as described by its [`MemorySpec`].
    ///
    /// `bytes` is the total one pass moves, so a
    /// [`Copy`](MemoryAccess::Copy) splits it across two buffers where a
    /// [`Read`](MemoryAccess::Read) takes it all from one.
    Memory(MemorySpec),
    /// Launch overhead measurement.
    Launch,
}

/// What a memory mode asks of a probe.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(std_io, serde(deny_unknown_fields))]
pub struct MemorySpec {
    /// Which directions of traffic to issue.
    pub access: MemoryAccess,
    /// The bytes one pass moves across the interface.
    pub bytes: u64,
}

impl ThroughputMode {
    /// `access` over as much as the interface will take at once.
    pub const fn memory(access: MemoryAccess) -> Self {
        Self::Memory(MemorySpec::new(access, access.default_working_set()))
    }

    /// What this mode asks of a memory probe, or `None` for the modes that do
    /// not measure memory.
    pub const fn memory_probe(&self) -> Option<MemorySpec> {
        match self {
            Self::Memory(spec) => Some(*spec),
            Self::ComputeDirect { .. } | Self::ComputeCmma { .. } | Self::Launch => None,
        }
    }
}

impl MemorySpec {
    /// A probe moving `bytes` per pass in the directions `access` names.
    pub const fn new(access: MemoryAccess, bytes: u64) -> Self {
        Self { access, bytes }
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
            ThroughputMode::Memory(_) | ThroughputMode::Launch => ElemType::Float(FloatKind::F32),
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
            ThroughputMode::Memory(_) => (self.bytes_per_s(key), "bytes"),
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

#[cfg(all(test, std_io))]
mod tests {
    use super::*;

    /// The throughput cache keys on the serialized key, so a layout change
    /// drops every measurement users have already paid for. That is what a
    /// version bump is for; a change that is not one must leave these alone.
    #[test]
    fn a_memory_key_keeps_its_serialized_form() {
        let encode = |mode| serde_json::to_string(&ThroughputKey { mode }).unwrap();

        assert_eq!(
            encode(ThroughputMode::memory(MemoryAccess::Copy)),
            r#"{"mode":{"Memory":{"access":"Copy","bytes":1073741824}}}"#
        );
        assert_eq!(
            encode(ThroughputMode::Memory(MemorySpec::new(
                MemoryAccess::Read,
                8192
            ))),
            r#"{"mode":{"Memory":{"access":"Read","bytes":8192}}}"#
        );
    }

    /// A working set is part of the key, so two sizes of the same access are
    /// separate measurements rather than one overwriting the other.
    #[test]
    fn a_working_set_is_part_of_the_key() {
        let small = ThroughputMode::Memory(MemorySpec::new(MemoryAccess::Read, 8192));
        let large = ThroughputMode::Memory(MemorySpec::new(MemoryAccess::Read, 16384));

        assert_ne!(small, large);
    }
}
