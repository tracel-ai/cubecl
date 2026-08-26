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

use crate::throughput::{MemoryAccess, ThroughputKey, ThroughputMode, ThroughputValue};

/// The smallest working set a curve is measured at, in bytes moved per pass.
///
/// Low enough to catch the bottom of the ramp, which is further down than it
/// looks: an Arc iGPU reads within 5% of its sustained figure at 128 KiB and
/// only falls away below that, 102 GB/s at 64 KiB, 66 at 32 KiB and 17 at
/// 8 KiB against 110. A sweep starting a few hundred kilobytes up would report
/// an almost flat curve and miss the effect it exists to measure.
pub const MIN_WORKING_SET: u64 = 8 * 1024;

/// The working sets a curve is measured at: powers of two from
/// [`MIN_WORKING_SET`] up to `cap`, which is where the device runs out of
/// allocation.
///
/// Powers of two because the interesting structure is spread over orders of
/// magnitude, not over any linear span. A `cap` below [`MIN_WORKING_SET`]
/// yields the cap alone, so a tiny device still gets a curve rather than an
/// empty one.
pub fn working_set_sweep(cap: u64) -> Vec<u64> {
    if cap < MIN_WORKING_SET {
        return alloc::vec![cap];
    }

    let mut sizes = Vec::new();
    let mut bytes = MIN_WORKING_SET;

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

/// One measured point of a [`MemoryCurve`].
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct MemoryPoint {
    /// The working set the probe ran at, in bytes moved per pass.
    pub bytes: u64,
    /// What the probe measured.
    pub value: ThroughputValue,
}

/// Memory throughput measured across a range of working sets.
///
/// Built by sweeping one probe over [`working_set_sweep`]; ask it for the
/// ceiling of a given working set with [`ceiling_at`](Self::ceiling_at).
#[derive(PartialEq, Clone, Debug)]
pub struct MemoryCurve {
    access: MemoryAccess,
    /// Ascending by `bytes`, deduplicated, and every rate finite and positive.
    points: Vec<MemoryPoint>,
}

impl MemoryCurve {
    /// Assembles a curve from points measured with `access`.
    ///
    /// A point with an empty working set, or a rate that isn't finite and
    /// positive (an unsupported or failed probe), describes nothing and is
    /// dropped; duplicated working sets keep the first point.
    pub fn new(access: MemoryAccess, points: impl IntoIterator<Item = MemoryPoint>) -> Self {
        let mut curve = Self {
            access,
            points: Vec::new(),
        };

        curve.points = points
            .into_iter()
            .filter(|point| {
                let rate = curve.rate(point);
                point.bytes > 0 && rate.is_finite() && rate > 0.0
            })
            .collect();

        curve.points.sort_unstable_by_key(|point| point.bytes);
        curve.points.dedup_by_key(|point| point.bytes);

        curve
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
    pub fn ceiling_at(&self, bytes: u64) -> Option<f64> {
        let first = self.points.first()?;
        let last = self.points.last()?;

        if bytes <= first.bytes {
            return Some(self.rate(first));
        }
        if bytes >= last.bytes {
            return Some(self.rate(last));
        }

        // `bytes` is strictly inside the sweep, so this lands on an interior
        // index and both neighbours exist.
        let above = self.points.partition_point(|point| point.bytes <= bytes);
        let (low, high) = (&self.points[above - 1], &self.points[above]);

        let span = log2(high.bytes) - log2(low.bytes);
        let weight = (log2(bytes) - log2(low.bytes)) / span;

        Some(self.rate(low) + weight * (self.rate(high) - self.rate(low)))
    }

    /// What a point measured, in bytes moved per second.
    fn rate(&self, point: &MemoryPoint) -> f64 {
        point.value.bytes_per_s(&ThroughputKey {
            mode: ThroughputMode::MemoryWorkingSet {
                access: self.access,
                bytes: point.bytes,
            },
        })
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
                },
            }),
        )
    }

    #[test]
    fn sweep_covers_powers_of_two_up_to_the_cap() {
        let min = MIN_WORKING_SET;
        let expected = alloc::vec![min, 2 * min, 4 * min, 8 * min];

        assert_eq!(working_set_sweep(8 * min), expected);

        // A cap between two powers of two stops at the last one that fits: the
        // probe must never be asked for more than the device can allocate.
        assert_eq!(working_set_sweep(12 * min), expected);

        // Below the minimum the cap is all there is, and a curve of one point
        // still answers queries.
        assert_eq!(working_set_sweep(min / 4), alloc::vec![min / 4]);
    }

    #[test]
    fn ceiling_interpolates_between_measured_points() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        // The geometric midpoint of the octave span, so half the rate span.
        assert!((curve.ceiling_at(2 * MB).unwrap() - 150.0).abs() < 1e-6);

        // Measured points come back exactly, not smoothed.
        assert!((curve.ceiling_at(MB).unwrap() - 100.0).abs() < 1e-6);
        assert!((curve.ceiling_at(4 * MB).unwrap() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn ceiling_clamps_outside_the_sweep() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        // Below the sweep: the smallest measured rate. Above it: the bus
        // figure, since the curve is flat past saturation.
        assert!((curve.ceiling_at(1).unwrap() - 100.0).abs() < 1e-6);
        assert!((curve.ceiling_at(u64::MAX).unwrap() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn unusable_points_are_dropped() {
        // A probe that never ran reports a zero duration, whose rate is NaN.
        let curve = MemoryCurve::new(
            MemoryAccess::Read,
            [MemoryPoint {
                bytes: MB,
                value: ThroughputValue::ZERO,
            }],
        );

        assert!(curve.points().is_empty());
        assert_eq!(curve.ceiling_at(MB), None);
    }

    #[test]
    fn log2_is_exact_on_powers_of_two_and_monotonic_between() {
        assert_eq!(log2(1), 0.0);
        assert_eq!(log2(MB), 20.0);
        assert_eq!(log2(1 << 63), 63.0);
        // Zero has no logarithm; the floor keeps the interpolation finite.
        assert_eq!(log2(0), 0.0);

        assert!(log2(3 * MB) > log2(2 * MB));
        assert!(log2(3 * MB) < log2(4 * MB));
    }
}
