//! Memory throughput as a function of working set size.
//!
//! A single bandwidth number describes one working set — in practice a large
//! one, chosen so the interface is saturated. A kernel that touches far less
//! than that cannot reach it no matter how it is written: there is not enough
//! traffic in flight to keep the interface busy. Scoring such a kernel against
//! the large-working-set figure reports good code as bad code.
//!
//! So the probes measure a *curve*: the same kernel at a range of sizes, from a
//! few kilobytes to hundreds of megabytes. [`MemoryCurve::ceiling_at`]
//! answers the question a consumer actually has — what can a kernel moving this
//! many bytes reach — by interpolating between the measured points.
//!
//! Every point is measured cold, on data no earlier pass left in cache, because
//! the question is what a kernel of that size moves and not how fast something
//! already resident can be read again. The curve therefore climbs with size and
//! flattens where the interface saturates.

use alloc::vec::Vec;

use crate::throughput::{MemoryAccess, ThroughputKey, ThroughputMode, ThroughputValue};

/// The smallest working set a curve is measured at, in bytes moved per pass.
///
/// Low enough to catch the bottom of the ramp, which is further down than it
/// looks: on an M2 Pro the curve is already within 12% of the bus figure at
/// 256 KiB and only falls away below that — 146 GB/s at 128 KiB, 94 at 64 KiB,
/// 11 at 8 KiB. A sweep starting a few hundred kilobytes up would report an
/// almost flat curve and miss the effect it exists to measure.
///
/// Below this a pass is a handful of cubes finishing before the next can be
/// issued, so what is measured is latency rather than bandwidth.
pub const MIN_WORKING_SET: u64 = 8 * 1024;

/// The working sets a curve is measured at: powers of two from
/// [`MIN_WORKING_SET`] up to `cap`, which is where the device runs out of
/// allocation.
///
/// Powers of two because the interesting structure — a cache level ending, the
/// interface saturating — is spread over orders of magnitude, not over any
/// linear span. A `cap` below [`MIN_WORKING_SET`] yields the cap alone, so a
/// tiny device still gets a curve rather than an empty one.
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

/// What limits a memory rate at a given working set.
///
/// Classified from the measured curve alone — a rate is compared against the
/// one measured at the largest working set, which is the figure that describes
/// the bus. Nothing here knows the device's cache sizes.
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash)]
pub enum MemoryRegime {
    /// Above the bus figure: the working set was served by a cache after all,
    /// despite the probe reading cold. A real number for a kernel of this size,
    /// and one that must never be compared against — or reported as — bus
    /// bandwidth.
    Cached,
    /// Below the bus figure with the working set still climbing towards it.
    /// Not enough traffic in flight to saturate the interface, which is a
    /// property of the size rather than of the kernel.
    Ramp,
    /// At the bus figure: the interface is saturated and this is the ceiling in
    /// the usual sense.
    Saturated,
}

/// A memory ceiling, with what it is a ceiling *of*.
///
/// The two fields travel together on purpose. A bare rate from the small end of
/// a curve is cache bandwidth and reads exactly like a bus figure, so this type
/// never hands one out without the [`MemoryRegime`] that says which it is.
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct MemoryCeiling {
    /// Which directions of traffic the underlying probe issued.
    pub access: MemoryAccess,
    /// The working set this ceiling applies to, in bytes moved.
    pub working_set: u64,
    /// The rate, in bytes moved per second.
    pub bytes_per_s: f64,
    /// What limits the rate here.
    pub regime: MemoryRegime,
}

/// One measured point of a [`MemoryCurve`].
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct MemoryPoint {
    /// Which directions of traffic the probe issued.
    pub access: MemoryAccess,
    /// The working set the probe ran at, in bytes moved per pass.
    pub bytes: u64,
    /// What the probe measured.
    pub value: ThroughputValue,
}

impl MemoryPoint {
    /// Creates a point from a probe result.
    pub fn new(access: MemoryAccess, bytes: u64, value: ThroughputValue) -> Self {
        Self {
            access,
            bytes,
            value,
        }
    }

    /// The key that produced this point.
    pub fn key(&self) -> ThroughputKey {
        ThroughputKey {
            mode: ThroughputMode::memory(self.access, self.bytes),
        }
    }

    /// The measured rate, in bytes moved per second.
    pub fn bytes_per_s(&self) -> f64 {
        self.value.bytes_per_s(&self.key())
    }
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
    /// A rate this close to the largest working set's is the bus figure rather
    /// than a size that failed to reach it.
    const SATURATED: f64 = 0.95;
    /// A rate this far above the largest working set's cannot be coming from
    /// the bus, so the working set is cache-resident.
    const CACHED: f64 = 1.15;

    /// Assembles a curve from measured points.
    ///
    /// Points measuring another access, an empty working set, or a rate that
    /// isn't finite and positive (an unsupported or failed probe) describe
    /// nothing and are dropped; duplicated working sets keep the first point.
    pub fn new(access: MemoryAccess, points: impl IntoIterator<Item = MemoryPoint>) -> Self {
        let mut points: Vec<_> = points
            .into_iter()
            .filter(|point| {
                let rate = point.bytes_per_s();
                point.access == access && point.bytes > 0 && rate.is_finite() && rate > 0.0
            })
            .collect();

        points.sort_unstable_by_key(|point| point.bytes);
        points.dedup_by_key(|point| point.bytes);

        Self { access, points }
    }

    /// Which directions of traffic this curve's probe issued.
    pub fn access(&self) -> MemoryAccess {
        self.access
    }

    /// The measured points, ascending by working set.
    pub fn points(&self) -> &[MemoryPoint] {
        &self.points
    }

    /// Whether nothing was measured. Every query on an empty curve is `None`.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The ceiling for a kernel moving `bytes`, interpolated between the two
    /// measured points that bracket it and clamped to the ends of the sweep.
    ///
    /// Interpolation is linear in `log2(bytes)`, matching the geometric spacing
    /// the curve is sampled at; between two adjacent samples it is exact at
    /// both ends and monotonic in between.
    ///
    /// A working set below the smallest measured point gets that point's rate,
    /// which is the least the sweep saw. Above the largest, the bus figure
    /// itself — that part of the curve is flat, which is why the sweep stops
    /// there.
    pub fn ceiling_at(&self, bytes: u64) -> Option<MemoryCeiling> {
        let first = self.points.first()?;
        let last = self.points.last()?;

        let bytes_per_s = if bytes <= first.bytes {
            first.bytes_per_s()
        } else if bytes >= last.bytes {
            last.bytes_per_s()
        } else {
            // `bytes` is strictly inside the sweep, so this lands on an
            // interior index and both neighbours exist.
            let above = self.points.partition_point(|point| point.bytes <= bytes);
            let (low, high) = (&self.points[above - 1], &self.points[above]);

            let span = log2(high.bytes) - log2(low.bytes);
            let weight = (log2(bytes) - log2(low.bytes)) / span;

            low.bytes_per_s() + weight * (high.bytes_per_s() - low.bytes_per_s())
        };

        Some(MemoryCeiling {
            access: self.access,
            working_set: bytes,
            bytes_per_s,
            regime: self.regime(bytes_per_s),
        })
    }

    /// The largest working set measured, which is the bus figure: the classic
    /// single number, and the reference every regime is classified against.
    pub fn peak(&self) -> Option<MemoryCeiling> {
        let last = self.points.last()?;

        Some(MemoryCeiling {
            access: self.access,
            working_set: last.bytes,
            bytes_per_s: last.bytes_per_s(),
            regime: MemoryRegime::Saturated,
        })
    }

    /// The smallest measured working set that reaches the bus figure — where a
    /// kernel stops being limited by how little it moves.
    ///
    /// `None` when no measured point is [`Saturated`](MemoryRegime::Saturated),
    /// which for a non-empty curve cannot happen: the largest point defines the
    /// figure and always reaches it.
    pub fn saturation_point(&self) -> Option<MemoryCeiling> {
        self.points
            .iter()
            .map(|point| point.bytes)
            .find_map(|bytes| {
                self.ceiling_at(bytes)
                    .filter(|ceiling| ceiling.regime == MemoryRegime::Saturated)
            })
    }

    /// Classifies a rate against the bus figure. See [`MemoryRegime`].
    fn regime(&self, bytes_per_s: f64) -> MemoryRegime {
        let Some(bus) = self.points.last().map(|point| point.bytes_per_s()) else {
            return MemoryRegime::Ramp;
        };

        if bytes_per_s > bus * Self::CACHED {
            MemoryRegime::Cached
        } else if bytes_per_s >= bus * Self::SATURATED {
            MemoryRegime::Saturated
        } else {
            MemoryRegime::Ramp
        }
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

    /// A point moving `bytes` at `bytes_per_s`.
    fn point(access: MemoryAccess, bytes: u64, bytes_per_s: f64) -> MemoryPoint {
        // `bytes_per_s` is `ops_count * dtype.size() / duration`, and every
        // memory mode keys on F32.
        let ops_count = (bytes / 4) as usize;
        let duration = Duration::from_secs_f64(bytes as f64 / bytes_per_s);

        MemoryPoint::new(
            access,
            bytes,
            ThroughputValue {
                ops_count,
                duration,
            },
        )
    }

    fn curve(points: &[(u64, f64)]) -> MemoryCurve {
        MemoryCurve::new(
            MemoryAccess::Read,
            points
                .iter()
                .map(|&(bytes, rate)| point(MemoryAccess::Read, bytes, rate)),
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
        let ceiling = curve.ceiling_at(2 * MB).unwrap();
        assert!((ceiling.bytes_per_s - 150.0).abs() < 1e-6);
        assert_eq!(ceiling.working_set, 2 * MB);

        // Measured points come back exactly, not smoothed.
        assert!((curve.ceiling_at(MB).unwrap().bytes_per_s - 100.0).abs() < 1e-6);
        assert!((curve.ceiling_at(4 * MB).unwrap().bytes_per_s - 200.0).abs() < 1e-6);
    }

    #[test]
    fn ceiling_clamps_outside_the_sweep() {
        let curve = curve(&[(MB, 100.0), (4 * MB, 200.0)]);

        // Below the sweep: the smallest measured rate. Above it: the bus
        // figure, since the curve is flat past saturation.
        assert!((curve.ceiling_at(1).unwrap().bytes_per_s - 100.0).abs() < 1e-6);
        assert!((curve.ceiling_at(u64::MAX).unwrap().bytes_per_s - 200.0).abs() < 1e-6);
    }

    #[test]
    fn regimes_separate_cache_from_bus() {
        // A cache-resident small end, a ramp, then the bus figure.
        let curve = curve(&[(MB, 400.0), (16 * MB, 120.0), (256 * MB, 200.0)]);

        assert_eq!(curve.ceiling_at(MB).unwrap().regime, MemoryRegime::Cached);
        assert_eq!(
            curve.ceiling_at(16 * MB).unwrap().regime,
            MemoryRegime::Ramp
        );
        assert_eq!(
            curve.ceiling_at(256 * MB).unwrap().regime,
            MemoryRegime::Saturated
        );

        // Within tolerance of the largest point counts as saturated: the sweep
        // is a measurement, not an exact function.
        let near_bus = curve.ceiling_at(230 * MB).unwrap();
        assert!(near_bus.bytes_per_s < 200.0);
        assert_eq!(near_bus.regime, MemoryRegime::Saturated);
    }

    #[test]
    fn peak_and_saturation_point_describe_the_ends() {
        let curve = curve(&[
            (MB, 400.0),
            (16 * MB, 120.0),
            (64 * MB, 199.0),
            (256 * MB, 200.0),
        ]);

        let peak = curve.peak().unwrap();
        assert_eq!(peak.working_set, 256 * MB);
        assert!((peak.bytes_per_s - 200.0).abs() < 1e-6);

        // 64 MB is within tolerance of the bus figure, so that is where a
        // kernel stops being limited by its size. The cache-resident 1 MB point
        // is faster still and must not be mistaken for it.
        assert_eq!(curve.saturation_point().unwrap().working_set, 64 * MB);
    }

    #[test]
    fn unusable_points_are_dropped() {
        // A probe that never ran reports a zero duration, whose rate is NaN,
        // and a point measuring another access belongs to another curve.
        let zero = MemoryPoint::new(MemoryAccess::Read, MB, ThroughputValue::ZERO);
        let other = point(MemoryAccess::Copy, 2 * MB, 100.0);
        let good = point(MemoryAccess::Read, 4 * MB, 100.0);

        let curve = MemoryCurve::new(MemoryAccess::Read, [zero, other, good]);

        assert_eq!(curve.points().len(), 1);
        assert_eq!(curve.points()[0].bytes, 4 * MB);

        assert!(MemoryCurve::new(MemoryAccess::Read, []).is_empty());
        assert_eq!(
            MemoryCurve::new(MemoryAccess::Read, []).ceiling_at(MB),
            None
        );
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
