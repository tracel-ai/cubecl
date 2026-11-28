use crate::throughput::{CmmaDims, ComputeCmmaConfig};
use alloc::{format, string::String};
use core::time::Duration;
use cubecl_ir::{ElemType, FloatKind};
use thiserror::Error;

/// What the probes measure, as opposed to which release ran them. Bump it when
/// a probe changes what it reports.
pub const PROBE_VERSION: u32 = 4;

/// Bytes one buffer of a [`ThroughputMode::Memory`] probe moves per pass at its
/// default working set. The probe's buffer is a multiple of this, and both are
/// clamped to the device's maximum allocation when the probe runs.
pub const DEFAULT_WORKING_SET_BYTES: u64 = 512 * 1024 * 1024;

/// Which directions of traffic a memory probe issues.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
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
    pub const fn buffer_count(&self) -> u64 {
        match self {
            Self::Copy => 2,
            Self::Read | Self::Write => 1,
        }
    }

    /// The working set of the single-size probe for this access, in bytes moved
    /// per pass: [`DEFAULT_WORKING_SET_BYTES`] per buffer touched.
    pub const fn default_working_set_bytes(&self) -> u64 {
        DEFAULT_WORKING_SET_BYTES * self.buffer_count()
    }
}

/// What a throughput probe measures: a kind of arithmetic, traffic across the memory interface,
/// or the cost of a launch.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum ThroughputMode {
    /// Compute direct calculation without special hardware acceleration.
    ///
    /// The ceiling is for operands of `dtype`, not arithmetic performed in it:
    /// where converting to f32 retires more, that is what the probe reports.
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
    /// How fast the device runs cubes of `units` units that each make one round trip to global
    /// memory, a value a unit: the ceiling of a grid of many small cubes, however little each
    /// of them moves or computes, set by how many of them the device keeps waiting on memory at
    /// once. Counted in cubes.
    CubeRoundTrips {
        /// The units of every cube the probe launches.
        units: u32,
    },
    /// Reads of a window of `bytes`, by every unit of a grid that fills the device, again and
    /// again: the bandwidth a kernel reads at what it or another cube read before, from
    /// whichever level of the cache holds a working set that size. Counted in `f32`s read.
    CacheRead {
        /// The window the probe reads, the size of the working set read again.
        bytes: u64,
    },
    /// Reads of shared memory by every unit of a grid that fills the device, each cube reading
    /// an array it filled: the bandwidth a kernel reads what its cube staged at. Counted in `f32`s
    /// read.
    SharedRead,
}

/// What a memory mode asks of a probe.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(serializable, serde(deny_unknown_fields))]
pub struct MemorySpec {
    /// Which directions of traffic to issue.
    pub access: MemoryAccess,
    /// The bytes one pass moves across the interface.
    pub bytes: u64,
}

impl ThroughputMode {
    /// `access` over as much as the interface will take at once.
    pub const fn memory(access: MemoryAccess) -> Self {
        Self::Memory(MemorySpec::new(access, access.default_working_set_bytes()))
    }

    /// What this mode asks of a memory probe, or `None` for the modes that do
    /// not measure memory.
    pub const fn memory_probe(&self) -> Option<MemorySpec> {
        match self {
            Self::Memory(spec) => Some(*spec),
            Self::ComputeDirect { .. }
            | Self::ComputeCmma { .. }
            | Self::Launch
            | Self::CubeRoundTrips { .. }
            | Self::CacheRead { .. }
            | Self::SharedRead => None,
        }
    }
}

impl MemorySpec {
    /// A probe moving `bytes` per pass in the directions `access` names.
    pub const fn new(access: MemoryAccess, bytes: u64) -> Self {
        Self { access, bytes }
    }
}

/// What one measured throughput is the throughput of, as its cache keys it.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
// Reject cached entries from an older key layout instead of silently ignoring their extra fields.
#[cfg_attr(serializable, serde(deny_unknown_fields))]
pub struct ThroughputKey {
    /// What the probe measured.
    pub mode: ThroughputMode,
}

impl ThroughputKey {
    /// The key of compute in `input` accumulated in `accumulator`: on the matrix instruction of
    /// `cmma` dims where there is one, else on plain arithmetic in `accumulator`.
    pub fn compute(cmma: Option<CmmaDims>, input: ElemType, accumulator: ElemType) -> Self {
        let mode = match cmma {
            Some(cmma_dims) => ThroughputMode::ComputeCmma {
                dtype: input,
                config: ComputeCmmaConfig {
                    accumulator_type: accumulator,
                    cmma_dims,
                },
            },
            None => ThroughputMode::ComputeDirect { dtype: accumulator },
        };
        Self { mode }
    }

    /// The element type the probe's operations are counted in, `f32` for memory and launches.
    pub fn dtype(&self) -> ElemType {
        match self.mode {
            ThroughputMode::ComputeDirect { dtype } => dtype,
            ThroughputMode::ComputeCmma { dtype, .. } => dtype,
            // For memory, launch and cube throughput, we use a default element type (F32).
            ThroughputMode::Memory(_)
            | ThroughputMode::Launch
            | ThroughputMode::CubeRoundTrips { .. }
            | ThroughputMode::CacheRead { .. }
            | ThroughputMode::SharedRead => ElemType::Float(FloatKind::F32),
        }
    }
}

/// Why a device has no peak to report for a probe.
#[derive(Error, Eq, PartialEq, Clone, Copy, Debug)]
pub enum ThroughputError {
    /// The device implements no such operation.
    #[error("unsupported")]
    Unsupported,
    /// The device's timer reported no elapsed time for any shape of the probe.
    #[error("no timing")]
    NoTiming,
    /// The device could not allocate the buffers the probe runs over.
    #[error("allocation failed")]
    Allocation,
    /// The probe's kernel did not run: it failed to compile or to launch, or
    /// the device faulted under it.
    #[error("launch failed")]
    Launch,
}

impl ThroughputError {
    /// Whether probing again in this process would fail the same way: what the device implements
    /// and times does not change, while the room it has and whether a kernel runs depend on what
    /// else holds it.
    pub(crate) fn is_lasting(&self) -> bool {
        match self {
            Self::Unsupported | Self::NoTiming => true,
            Self::Allocation | Self::Launch => false,
        }
    }
}

/// A measured throughput: how many operations a pass did, and how long the fastest pass took.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct ThroughputValue {
    /// The operations one pass did, counted as the probe's mode counts them.
    pub ops_count: usize,
    /// The duration of the computation, at the fastest sample.
    pub duration: Duration,
    /// The statistics of the samples taken once the device was warm, or `None` where fewer than two
    /// were taken.
    pub samples: Option<SampleStats>,
}

/// The mean, standard deviation and slowest of a probe's samples, each a whole sample at one iteration
/// count, so neither is lost to rounding a per-iteration time.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct SampleStats {
    /// How many samples were taken.
    pub count: usize,
    /// Their mean.
    pub mean: Duration,
    /// Their sample standard deviation.
    pub std_dev: Duration,
    /// The slowest of them.
    pub max: Duration,
}

impl SampleStats {
    /// The statistics of `samples`, or `None` for fewer than two, which have no variance.
    pub fn new(samples: &[Duration]) -> Option<Self> {
        if samples.len() < 2 {
            return None;
        }
        let count = samples.len() as f64;
        let mean = samples.iter().map(Duration::as_secs_f64).sum::<f64>() / count;
        let squares = samples
            .iter()
            .map(|sample| {
                let deviation = sample.as_secs_f64() - mean;
                deviation * deviation
            })
            .sum::<f64>();
        // A sample near `Duration::MAX` does not survive the round trip through `f64`.
        let duration = |secs: f64| Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX);
        Some(Self {
            count: samples.len(),
            mean: duration(mean),
            std_dev: duration(num_traits::Float::sqrt(squares / (count - 1.0))),
            max: samples
                .iter()
                .copied()
                .max()
                .expect("two samples should have a slowest"),
        })
    }

    /// The standard deviation over the mean, or `None` where the timer read zero, which says
    /// nothing about the device.
    pub fn relative_std_dev(&self) -> Option<f64> {
        self.relative_to_mean(self.std_dev)
    }

    /// How far the slowest sample sits above the mean, as a fraction of the mean, or `None` where
    /// the timer read zero.
    pub fn relative_max(&self) -> Option<f64> {
        self.relative_to_mean(self.max)
            .map(|relative| relative - 1.0)
    }
}

impl SampleStats {
    /// `duration` over the mean, or `None` where the timer read zero.
    fn relative_to_mean(&self, duration: Duration) -> Option<f64> {
        if self.mean.is_zero() {
            return None;
        }
        Some(duration.as_secs_f64() / self.mean.as_secs_f64())
    }
}

const NANOS_PER_SEC: u128 = 1_000_000_000;

/// The step between two decimal unit prefixes.
const PREFIX_STEP: f64 = 1000.0;

/// The decimal unit prefixes from none up, in order.
const PREFIXES: [&str; 11] = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"];

/// What [`ThroughputValue::format`] writes for a value that measured nothing.
const NOT_AVAILABLE: &str = "N/A";

impl ThroughputValue {
    /// A zero-initialized throughput value, representing no operations or duration.
    pub const ZERO: Self = Self {
        ops_count: 0,
        duration: Duration::ZERO,
        samples: None,
    };

    /// The value of samples each `iterations` long: the fastest per iteration, and the statistics
    /// of all of them.
    ///
    /// # Assumptions
    ///
    /// `iterations` is at least one, and `samples` holds at least one sample.
    ///
    /// # Panics
    ///
    /// When either Assumption is broken: no sample, or no iteration.
    pub fn from_samples(ops_count: usize, iterations: usize, samples: &[Duration]) -> Self {
        let fastest = samples
            .iter()
            .copied()
            .min()
            .expect("a throughput should be measured from at least one sample");
        let iterations = iterations as u128;
        // To the nearest nanosecond: a kernel under half a nanosecond a pass would otherwise read
        // as taking no time.
        let nanos = (fastest.as_nanos() + iterations / 2) / iterations;

        Self {
            ops_count,
            duration: Duration::new(
                (nanos / NANOS_PER_SEC) as u64,
                (nanos % NANOS_PER_SEC) as u32,
            ),
            samples: SampleStats::new(samples),
        }
    }

    /// The samples' standard deviation over their mean, or `None` where fewer than two were
    /// taken.
    pub fn relative_std_dev(&self) -> Option<f64> {
        self.samples.and_then(|samples| samples.relative_std_dev())
    }

    /// How far the slowest sample sits above the mean, or `None` where fewer than two were taken.
    pub fn relative_max(&self) -> Option<f64> {
        self.samples.and_then(|samples| samples.relative_max())
    }

    /// The operations a second, `NaN` where the timer read zero.
    pub fn ops_per_s(&self) -> f64 {
        self.per_second(self.ops_count)
    }

    /// The bytes a second, each operation of `key` moving one of its element type, `NaN` where
    /// the timer read zero.
    pub fn bytes_per_s(&self, key: &ThroughputKey) -> f64 {
        self.per_second(self.ops_count * key.dtype().size())
    }

    /// How long one operation took, zero for a value without operations.
    pub fn duration_per_op(&self) -> Duration {
        if self.ops_count == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(self.duration.as_secs_f64() / self.ops_count as f64)
        }
    }

    /// The value written for a person, in the unit `key`'s mode counts: `1.2340 GOPS/s`,
    /// `512.0000 Gbytes/s`, `3µs/launch`, `20.0000 Mcubes/s`, or `N/A` where it measured nothing.
    pub fn format(&self, key: &ThroughputKey) -> String {
        match key.mode {
            ThroughputMode::ComputeDirect { .. } | ThroughputMode::ComputeCmma { .. } => {
                format_rate(self.ops_per_s(), "OPS")
            }
            ThroughputMode::Memory(_) => format_rate(self.bytes_per_s(key), "bytes"),
            ThroughputMode::Launch => format_per_launch(self.duration_per_op()),
            ThroughputMode::CubeRoundTrips { .. } => format_rate(self.ops_per_s(), "cubes"),
            ThroughputMode::CacheRead { .. } | ThroughputMode::SharedRead => {
                format_rate(self.bytes_per_s(key), "bytes")
            }
        }
    }
}

impl ThroughputValue {
    /// `amount` over the duration, `NaN` where the timer read zero.
    fn per_second(&self, amount: usize) -> f64 {
        if self.duration.is_zero() {
            return f64::NAN;
        }
        amount as f64 / self.duration.as_secs_f64()
    }
}

/// `rate` of `unit` a second, with the largest decimal prefix that keeps it under a step.
fn format_rate(rate: f64, unit: &str) -> String {
    if rate.is_nan() {
        return String::from(NOT_AVAILABLE);
    }
    let (scaled, prefix) = decimal_prefix(rate);
    format!("{scaled:.4} {prefix}{unit}/s")
}

/// `rate` divided down by the largest prefix that keeps it under [`PREFIX_STEP`], and that
/// prefix.
fn decimal_prefix(rate: f64) -> (f64, &'static str) {
    let mut scaled = rate;
    let mut index = 0;
    while index < PREFIXES.len() - 1 && scaled >= PREFIX_STEP {
        scaled /= PREFIX_STEP;
        index += 1;
    }
    (scaled, PREFIXES[index])
}

/// One launch's `duration`, or [`NOT_AVAILABLE`] where it is zero.
fn format_per_launch(duration: Duration) -> String {
    if duration.is_zero() {
        return String::from(NOT_AVAILABLE);
    }
    format!("{duration:?}/launch")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(ops_count: usize, duration: Duration) -> ThroughputValue {
        ThroughputValue {
            ops_count,
            duration,
            samples: None,
        }
    }

    fn compute_key() -> ThroughputKey {
        ThroughputKey::compute(
            None,
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F32),
        )
    }

    /// The throughput cache keys on the serialized key, so a layout change
    /// drops every measurement users have already paid for. That is what a
    /// version bump is for; a change that is not one must leave these alone.
    #[cfg(std_io)]
    #[test]
    fn a_default_memory_key_keeps_its_serialized_form() {
        let key = ThroughputKey {
            mode: ThroughputMode::memory(MemoryAccess::Copy),
        };

        assert_eq!(
            serde_json::to_string(&key).unwrap(),
            r#"{"mode":{"Memory":{"access":"Copy","bytes":1073741824}}}"#
        );
    }

    #[cfg(std_io)]
    #[test]
    fn a_sized_memory_key_keeps_its_serialized_form() {
        let key = ThroughputKey {
            mode: ThroughputMode::Memory(MemorySpec::new(MemoryAccess::Read, 8192)),
        };

        assert_eq!(
            serde_json::to_string(&key).unwrap(),
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

    #[test]
    fn a_copy_touches_two_buffers() {
        assert_eq!(MemoryAccess::Copy.buffer_count(), 2);
    }

    #[test]
    fn a_read_moves_one_default_working_set() {
        assert_eq!(
            MemoryAccess::Read.default_working_set_bytes(),
            DEFAULT_WORKING_SET_BYTES
        );
    }

    #[test]
    fn a_memory_mode_asks_a_memory_probe() {
        assert_eq!(
            ThroughputMode::memory(MemoryAccess::Write).memory_probe(),
            Some(MemorySpec::new(
                MemoryAccess::Write,
                DEFAULT_WORKING_SET_BYTES
            ))
        );
    }

    #[test]
    fn a_launch_asks_no_memory_probe() {
        assert_eq!(ThroughputMode::Launch.memory_probe(), None);
    }

    #[test]
    fn memory_is_counted_in_f32() {
        let key = ThroughputKey {
            mode: ThroughputMode::Launch,
        };

        assert_eq!(key.dtype(), ElemType::Float(FloatKind::F32));
    }

    #[test]
    fn compute_without_a_matrix_instruction_is_direct_in_the_accumulator() {
        let key = ThroughputKey::compute(
            None,
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F32),
        );

        assert_eq!(
            key.mode,
            ThroughputMode::ComputeDirect {
                dtype: ElemType::Float(FloatKind::F32)
            }
        );
    }

    #[test]
    fn compute_on_a_matrix_instruction_is_counted_in_the_input() {
        let dims = CmmaDims {
            m: 16,
            n: 16,
            k: 16,
        };

        let key = ThroughputKey::compute(
            Some(dims),
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F32),
        );

        assert_eq!(key.dtype(), ElemType::Float(FloatKind::F16));
    }

    #[test]
    fn ops_per_second_divide_by_the_duration() {
        assert_eq!(value(8, Duration::from_secs(2)).ops_per_s(), 4.0);
    }

    #[test]
    fn a_zero_duration_has_no_rate() {
        assert!(value(8, Duration::ZERO).ops_per_s().is_nan());
    }

    #[test]
    fn bytes_per_second_count_the_element_size() {
        assert_eq!(
            value(8, Duration::from_secs(2)).bytes_per_s(&compute_key()),
            16.0
        );
    }

    #[test]
    fn duration_per_op_splits_the_duration() {
        assert_eq!(
            value(4, Duration::from_secs(2)).duration_per_op(),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn no_operations_take_no_time() {
        assert_eq!(
            value(0, Duration::from_secs(2)).duration_per_op(),
            Duration::ZERO
        );
    }

    #[test]
    fn a_rate_is_written_with_its_prefix() {
        assert_eq!(
            value(2_000_000, Duration::from_secs(1)).format(&compute_key()),
            "2.0000 MOPS/s"
        );
    }

    #[test]
    fn a_rate_that_measured_nothing_is_not_available() {
        assert_eq!(
            value(1, Duration::ZERO).format(&compute_key()),
            NOT_AVAILABLE
        );
    }

    #[test]
    fn a_launch_is_written_per_launch() {
        let key = ThroughputKey {
            mode: ThroughputMode::Launch,
        };

        assert_eq!(
            value(2, Duration::from_micros(6)).format(&key),
            "3µs/launch"
        );
    }

    #[test]
    fn the_largest_prefix_holds_any_rate() {
        assert_eq!(decimal_prefix(f64::INFINITY).1, "Q");
    }

    #[test]
    fn one_sample_has_no_statistics() {
        assert_eq!(SampleStats::new(&[Duration::from_millis(5)]), None);
    }

    #[test]
    fn a_timer_reading_zero_has_no_relative_deviation() {
        let samples = SampleStats::new(&[Duration::ZERO, Duration::ZERO]).expect("two samples");

        assert_eq!(samples.relative_std_dev(), None);
    }

    #[test]
    fn a_sample_at_the_largest_duration_does_not_panic() {
        let samples = SampleStats::new(&[Duration::from_millis(1), Duration::MAX]);

        assert!(samples.is_some());
    }

    fn spread() -> SampleStats {
        SampleStats::new(&[
            Duration::from_micros(10),
            Duration::from_micros(30),
            Duration::from_micros(20),
        ])
        .expect("three samples")
    }

    #[test]
    fn the_slowest_sample_is_kept() {
        assert_eq!(spread().max, Duration::from_micros(30));
    }

    #[test]
    fn the_relative_max_is_how_far_the_slowest_sits_above_the_mean() {
        let above_mean = spread().relative_max().expect("a mean above zero");

        assert!(
            (above_mean - 0.5).abs() < 1e-9,
            "max sits {above_mean} above the mean"
        );
    }

    #[test]
    fn the_relative_deviation_is_the_deviation_over_the_mean() {
        let relative = spread().relative_std_dev().expect("a mean above zero");

        assert!((relative - 0.5).abs() < 1e-9, "deviates by {relative}");
    }

    #[test]
    fn a_value_from_the_largest_sample_does_not_panic() {
        let value = ThroughputValue::from_samples(1, 1, &[Duration::MAX]);

        assert_eq!(value.duration, Duration::MAX);
    }

    #[test]
    fn a_pass_under_a_nanosecond_rounds_to_the_nearest() {
        let value = ThroughputValue::from_samples(1, 4_000_000, &[Duration::from_millis(3)]);

        assert_eq!(value.duration, Duration::from_nanos(1));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn a_count_past_u32_divides_whole() {
        let iterations = 1usize << 33;
        let value = ThroughputValue::from_samples(1, iterations, &[Duration::from_nanos(1 << 33)]);

        assert_eq!(value.duration, Duration::from_nanos(1));
    }

    #[test]
    fn identical_samples_deviate_by_nothing() {
        let sample = Duration::from_micros(7);
        let samples = SampleStats::new(&[sample; 4]).expect("four samples");

        assert_eq!(samples.std_dev, Duration::ZERO);
    }
}
