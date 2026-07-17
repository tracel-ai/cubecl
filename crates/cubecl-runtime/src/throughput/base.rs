use alloc::{format, string::String};

use cubecl_ir::{ElemType, FloatKind};

use crate::throughput::{CmmaDims, ComputeCmmaConfig};

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
    /// Memory input reads and output writes.
    Memory,
    /// Launch overhead measurement.
    Launch,
}

/// Represents a key/configuration used to identify the throughput of a computation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
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
            ThroughputMode::Memory | ThroughputMode::Launch => ElemType::Float(FloatKind::F32),
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
    pub duration: core::time::Duration,
}

impl ThroughputValue {
    /// A zero-initialized throughput value, representing no operations or duration.
    pub const ZERO: Self = Self {
        ops_count: 0,
        duration: core::time::Duration::ZERO,
    };

    /// Returns the operations per second.
    pub fn ops_per_s(&self) -> f64 {
        if self.duration.is_zero() {
            return f64::NAN;
        }
        self.ops_count as f64 / self.duration.as_secs_f64()
    }

    /// Returns the bytes per second.
    pub fn bytes_per_s(&self) -> f64 {
        if self.duration.is_zero() {
            return f64::NAN;
        }
        (self.ops_count * ElemType::Float(FloatKind::F32).size()) as f64
            / self.duration.as_secs_f64()
    }

    /// Returns the duration per operation.
    pub fn duration_per_op(&self) -> core::time::Duration {
        if self.ops_count == 0 {
            core::time::Duration::ZERO
        } else {
            core::time::Duration::from_secs_f64(self.duration.as_secs_f64() / self.ops_count as f64)
        }
    }

    /// Formats the throughput value as a clean human-readable string.
    pub fn format(&self, key: &ThroughputKey) -> String {
        let (mut val_per_s, unit) = match key.mode {
            ThroughputMode::ComputeDirect { .. } | ThroughputMode::ComputeCmma { .. } => {
                (self.ops_per_s(), "OPS")
            }
            ThroughputMode::Memory => (self.bytes_per_s(), "bytes"),
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
