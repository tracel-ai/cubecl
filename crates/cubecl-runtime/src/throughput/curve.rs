//! Memory throughput as a function of working set size.
//!
//! A single bandwidth number describes one working set — in practice a large
//! one, chosen so the interface is saturated. A kernel that touches far less
//! than that cannot reach it no matter how it is written: there is not enough
//! traffic in flight to keep the interface busy. Scoring such a kernel against
//! the large-working-set figure reports good code as bad code.
//!
//! So the probes measure a *curve*: the same kernel at a range of sizes, from a
//! few kilobytes to hundreds of megabytes, every point on data no earlier pass
//! left in cache. [`MemoryCurve::ceiling_at`] answers the question a consumer
//! actually has — what can a kernel moving this many bytes reach.

use alloc::vec::Vec;

use crate::throughput::{
    LinkRoute, MemoryAccess, MemorySpec, ThroughputKey, ThroughputMode, ThroughputValue,
};

/// The smallest working set a curve is measured at, in bytes moved per pass.
///
/// The ramp bottoms out further down than it looks: an Arc iGPU reads 102 GB/s
/// at 64 KiB, 66 at 32 KiB and 17 at 8 KiB, against 110 sustained.
pub const MIN_WORKING_SET_BYTES: u64 = 8 * 1024;

/// One measured point of a [`MemoryCurve`].
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct MemoryPoint {
    /// The working set the probe ran at, in bytes moved per pass.
    pub bytes: u64,
    /// What the probe measured.
    pub value: ThroughputValue,
}

/// Which traffic a [`MemoryCurve`] measured.
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub enum Traffic {
    /// Across one device's memory interface.
    Memory(MemoryAccess),
    /// Between devices, whose [`LinkKey`](super::LinkKey) values count bytes as their operations.
    Link(LinkRoute),
}

impl From<MemoryAccess> for Traffic {
    fn from(access: MemoryAccess) -> Self {
        Self::Memory(access)
    }
}

/// Memory throughput measured across a range of working sets, on one device or between several.
///
/// Built by sweeping one probe over [`working_set_sweep`]; ask it for the
/// ceiling of a given working set with [`ceiling_at`](Self::ceiling_at).
#[derive(PartialEq, Clone, Debug)]
pub struct MemoryCurve {
    traffic: Traffic,
    /// Ascending by `bytes`, deduplicated, and every rate finite and positive.
    points: Vec<MemoryPoint>,
}

impl MemoryCurve {
    /// Assembles a curve from points measured over `traffic`.
    ///
    /// A point with an empty working set, or a rate that isn't finite and
    /// positive (an unsupported or failed probe), describes nothing and is
    /// dropped; duplicated working sets keep the first point.
    pub fn new(traffic: impl Into<Traffic>, points: impl IntoIterator<Item = MemoryPoint>) -> Self {
        let traffic = traffic.into();
        let mut points: Vec<MemoryPoint> = points
            .into_iter()
            .filter(|point| describes_something(traffic, point))
            .collect();

        points.sort_unstable_by_key(|point| point.bytes);
        points.dedup_by_key(|point| point.bytes);

        Self { traffic, points }
    }

    /// The working sets a curve is measured at: powers of two from
    /// [`MIN_WORKING_SET_BYTES`] up to `cap` bytes, which is where the device runs out of
    /// allocation.
    ///
    /// Powers of two because the interesting structure is spread over orders of
    /// magnitude, not over any linear span. A `cap` below [`MIN_WORKING_SET_BYTES`]
    /// yields the cap alone, so a tiny device still gets a curve rather than an
    /// empty one.
    pub fn sweep_up_to_bytes(cap: u64) -> Vec<u64> {
        sweep(cap)
    }

    /// The sweep size a working set of `bytes` is probed at: the power of two at or
    /// below it, never below [`MIN_WORKING_SET_BYTES`].
    ///
    /// One cache entry an octave, rather than one per distinct byte count. Down, so
    /// the ceiling a consumer reads is one the working set can reach: the rate
    /// climbs steeply per octave along the ramp, and the size above would report
    /// close to twice what these bytes move.
    pub fn sweep_point_for_bytes(bytes: u64) -> u64 {
        octave_at_or_below(bytes)
    }

    /// The measured points, ascending by working set.
    pub fn points(&self) -> &[MemoryPoint] {
        &self.points
    }

    /// The ceiling for a kernel moving `bytes`, in bytes moved per second,
    /// interpolated between the two measured points that bracket it and clamped
    /// to the ends of the sweep. `None` if nothing was measured.
    ///
    /// Interpolation is linear in `log2(bytes)`, matching the geometric spacing
    /// the curve is sampled at; it is exact at the measured points.
    ///
    /// Below the smallest measured working set the answer is that point's rate,
    /// the least the sweep saw. Above the largest it is the bus figure — that
    /// part of the curve is flat, which is why the sweep stops there.
    pub fn ceiling_at_bytes(&self, bytes: u64) -> Option<f64> {
        ceiling(self.traffic, &self.points, bytes)
    }
}

fn sweep(cap: u64) -> Vec<u64> {
    if cap < MIN_WORKING_SET_BYTES {
        return alloc::vec![cap];
    }

    let mut sizes = Vec::new();
    let mut bytes = MIN_WORKING_SET_BYTES;

    while bytes <= cap {
        sizes.push(bytes);
        // The last doubling before overflow would wrap to zero and loop forever.
        match bytes.checked_mul(2) {
            Some(next) => bytes = next,
            None => break,
        }
    }

    sizes
}

fn octave_at_or_below(bytes: u64) -> u64 {
    let bytes = bytes.max(MIN_WORKING_SET_BYTES);

    1 << (u64::BITS - 1 - bytes.leading_zeros())
}

/// Whether `point` measured a working set at a finite, positive rate.
fn describes_something(traffic: Traffic, point: &MemoryPoint) -> bool {
    let rate = rate(traffic, point);
    point.bytes > 0 && rate.is_finite() && rate > 0.0
}

fn ceiling(traffic: Traffic, points: &[MemoryPoint], bytes: u64) -> Option<f64> {
    let first = points.first()?;
    let last = points.last()?;

    if bytes <= first.bytes {
        return Some(rate(traffic, first));
    }
    if bytes >= last.bytes {
        return Some(rate(traffic, last));
    }

    // `bytes` is strictly inside the sweep, so this lands on an interior
    // index and both neighbours exist.
    let above = points.partition_point(|point| point.bytes <= bytes);
    let (low, high) = (&points[above - 1], &points[above]);

    let span = log2(high.bytes) - log2(low.bytes);
    let weight = (log2(bytes) - log2(low.bytes)) / span;

    Some(rate(traffic, low) + weight * (rate(traffic, high) - rate(traffic, low)))
}

/// What `point` measured over `traffic`, in bytes moved per second.
fn rate(traffic: Traffic, point: &MemoryPoint) -> f64 {
    match traffic {
        Traffic::Memory(access) => point.value.bytes_per_s(&ThroughputKey {
            mode: ThroughputMode::Memory(MemorySpec::new(access, point.bytes)),
        }),
        Traffic::Link(_) => point.value.ops_per_s(),
    }
}

/// `log2`, piecewise-linear across each octave.
///
/// `f64::log2` lives in `std` and this crate is `no_std`, so the exponent comes
/// from the bit width and the mantissa is interpolated linearly. Exact on
/// powers of two — which is every point the sweep measures — and monotonic
/// everywhere, which is all the interpolation needs.
fn log2(bytes: u64) -> f64 {
    let bytes = bytes.max(1);
    let exponent = 63 - bytes.leading_zeros();
    let mantissa = bytes as f64 / (1u64 << exponent) as f64;

    exponent as f64 + (mantissa - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    const MB: u64 = 1024 * 1024;
    const MIN: u64 = MIN_WORKING_SET_BYTES;

    /// A curve whose points measured the given rates.
    fn curve(points: &[(u64, f64)]) -> MemoryCurve {
        MemoryCurve::new(
            MemoryAccess::Read,
            points.iter().map(|&(bytes, bytes_per_s)| MemoryPoint {
                bytes,
                // The rate is `ops_count * dtype.size() / duration`, and every
                // memory mode keys on F32.
                value: ThroughputValue {
                    ops_count: (bytes / 4) as usize,
                    duration: Duration::from_secs_f64(bytes as f64 / bytes_per_s),
                    samples: None,
                },
            }),
        )
    }

    fn close(actual: Option<f64>, expected: f64) -> bool {
        actual.is_some_and(|actual| (actual - expected).abs() < 1e-6)
    }

    #[test]
    fn a_sweep_point_is_the_octave_at_or_below_the_working_set() {
        for bytes in [16 * 1024, 16 * 1024 + 1, 32 * 1024 - 1] {
            assert_eq!(
                MemoryCurve::sweep_point_for_bytes(bytes),
                16 * 1024,
                "at {bytes}"
            );
        }
    }

    /// Reading a ceiling from the octave above would credit a kernel with a
    /// rate its own working set cannot reach.
    #[test]
    fn a_sweep_point_never_rounds_up() {
        for bytes in [MIN, 9 * 1024, 100_000, 1 << 30] {
            assert!(
                MemoryCurve::sweep_point_for_bytes(bytes) <= bytes,
                "at {bytes}"
            );
        }
    }

    /// Every size lands on the grid the curve is measured at, so a probe is
    /// shared rather than added.
    #[test]
    fn a_sweep_point_lands_on_the_measured_grid() {
        let grid = MemoryCurve::sweep_up_to_bytes(1 << 30);

        for bytes in [1, 9 * 1024, 100_000, 1 << 20, 1 << 30] {
            assert!(
                grid.contains(&MemoryCurve::sweep_point_for_bytes(bytes)),
                "at {bytes}"
            );
        }
    }

    /// Below the smallest measured point there is nothing to round down to.
    #[test]
    fn a_working_set_under_the_floor_gets_the_floor() {
        assert_eq!(MemoryCurve::sweep_point_for_bytes(1), MIN);
    }

    #[test]
    fn a_sweep_covers_powers_of_two_up_to_the_cap() {
        assert_eq!(
            MemoryCurve::sweep_up_to_bytes(8 * MIN),
            alloc::vec![MIN, 2 * MIN, 4 * MIN, 8 * MIN]
        );
    }

    /// The probe must never be asked for more than the device can allocate.
    #[test]
    fn a_sweep_stops_at_the_last_power_under_the_cap() {
        assert_eq!(
            MemoryCurve::sweep_up_to_bytes(12 * MIN),
            alloc::vec![MIN, 2 * MIN, 4 * MIN, 8 * MIN]
        );
    }

    /// Below the minimum the cap is all there is, and a curve of one point
    /// still answers queries.
    #[test]
    fn a_cap_under_the_floor_is_swept_alone() {
        assert_eq!(
            MemoryCurve::sweep_up_to_bytes(MIN / 4),
            alloc::vec![MIN / 4]
        );
    }

    /// The geometric midpoint of the octave span, so half the rate span.
    #[test]
    fn the_ceiling_interpolates_between_measured_points() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        assert!(close(curve.ceiling_at_bytes(2 * MB), 150.0));
    }

    /// Measured points come back exactly, not smoothed.
    #[test]
    fn the_ceiling_is_exact_at_a_measured_point() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        assert!(close(curve.ceiling_at_bytes(4 * MB), 200.0));
    }

    /// Below the sweep: the smallest measured rate.
    #[test]
    fn the_ceiling_below_the_sweep_is_its_smallest_rate() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        assert!(close(curve.ceiling_at_bytes(1), 100.0));
    }

    /// Above the sweep: the bus figure, since the curve is flat past saturation.
    #[test]
    fn the_ceiling_above_the_sweep_is_the_bus_figure() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        assert!(close(curve.ceiling_at_bytes(u64::MAX), 200.0));
    }

    /// A probe that never ran reports a zero duration, whose rate is NaN.
    #[test]
    fn unusable_points_are_dropped() {
        let curve = MemoryCurve::new(
            MemoryAccess::Read,
            [MemoryPoint {
                bytes: MB,
                value: ThroughputValue::ZERO,
            }],
        );

        assert!(curve.points().is_empty());
    }

    #[test]
    fn points_come_back_ascending() {
        let curve = curve(&[(4 * MB, 200.0), (MB, 100.0)]);

        assert_eq!(curve.points()[0].bytes, MB);
    }

    #[test]
    fn log2_is_exact_on_powers_of_two() {
        assert_eq!([log2(1), log2(MB), log2(1 << 63)], [0.0, 20.0, 63.0]);
    }

    /// Zero has no logarithm; the floor keeps the interpolation finite.
    #[test]
    fn log2_of_zero_is_floored() {
        assert_eq!(log2(0), 0.0);
    }

    #[test]
    fn log2_is_monotonic_between_powers() {
        assert!(log2(2 * MB) < log2(3 * MB) && log2(3 * MB) < log2(4 * MB));
    }
}
