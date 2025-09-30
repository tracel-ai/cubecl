use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

#[cube]
/// Minimal aggregator: supports local update and merging (for cooperative reduce).
pub trait Aggregator<E: Float>: CubeType {
    /// create empty aggregator (e.g. -inf for max, 0 for sum)
    fn new() -> Self;
    /// incorporate a single element
    fn update(&mut self, e: E);
    /// merge another aggregator into self (associative combine)
    fn merge(this: &mut Self, other: &Self);
    /// finalize result
    fn finalize(&self) -> E;
}

#[cube]
/// Cooperative reduction primitive that knows how to reduce per logical group.
/// Implemented for a target (warp) using the warp intrinsics available in CubeCL.
pub trait CoopReduce: CubeType {
    type Agg: Aggregator<Self::Elem>;
    type Elem: Float;
    /// Reduce `local` into the group's root lane and return root's final aggregator.
    /// semantics: returns `Some(result)` in the lane designated as group owner, else `None`.
    fn reduce_group(local: &Self::Agg, lane: u32, group_id: u32) -> CubeOption<Self::Agg>;
    // /// Optional: reduce across entire warp if group_id == global_group.
}

#[cube]
/// Layout that maps a lane (thread index) to a logical `(row, col_offset, count)`
/// and provides group identifiers for cooperative ops.
/// You can implement multiple layouts (one-row-per-warp, interleaved, split-rows etc).
pub trait PlaneLayout: CubeType {
    type Elem: Float;
    /// lane index in [0..warp_size)
    fn lane_to_group(lane: u32) -> u32;
    /// for debug / mapping: which logical row does lane contribute to?
    fn lane_to_row(lane: u32) -> u32;
    /// how many elements this lane processes for the logical row
    fn lane_local_count(lane: u32) -> u32;
    /// index within row for element `k` (if needed)
    fn lane_local_index(lane: u32, k: u32) -> u32;
}

// Simple max aggregator
#[derive(CubeType, Clone, Copy, Debug)]
pub struct MaxAgg<E: Float> {
    max: E,
}

#[cube]
impl<E: Float> Aggregator<E> for MaxAgg<E> {
    fn new() -> Self {
        MaxAgg::<E> {
            max: E::from_int(-99999999),
        }
    }
    fn update(&mut self, e: E) {
        if e > self.max {
            self.max = e;
        }
    }
    fn merge(this: &mut Self, other: &Self) {
        if other.max > this.max {
            this.max = other.max;
        }
    }
    fn finalize(&self) -> E {
        self.max
    }
}

// Sum aggregator (optionally compensated)
#[derive(CubeType, Clone, Copy, Debug)]
pub struct SumAgg<E: Float> {
    sum: E,
}

#[cube]
impl<E: Float> Aggregator<E> for SumAgg<E> {
    fn new() -> Self {
        SumAgg::<E> {
            sum: E::from_int(0),
        }
    }
    fn update(&mut self, e: E) {
        self.sum += e;
    }
    fn merge(this: &mut Self, other: &Self) {
        this.sum += other.sum;
    }
    fn finalize(&self) -> E {
        self.sum
    }
}

