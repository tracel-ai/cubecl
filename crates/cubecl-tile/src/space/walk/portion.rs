//! This instance's share of a level distributed as one flat index.

use cubecl::prelude::*;

use crate::{ComputeScope, Coverage, CubeAxis, Level, Region, Walk, WalkExpand};

/// The regions this instance touches and how much of the first and last is its own.
/// Counted in the level below's joint steps.
///
/// The steps are laid end to end and cut into one run an instance, every run `ceil(steps /
/// instances)` long but the last that holds any, which holds what is left: the runs holding steps
/// come first, so no run between two that hold steps is empty, and the longest run, which is what
/// the level takes, is as short as an even split's.
#[derive(CubeType)]
pub struct Portion {
    /// The regions the portion touches, in order.
    regions: Walk,
    /// The first of them, in the level's flat index.
    first: usize,
    /// The portion's bounds on the joint step index.
    start: usize,
    end: usize,
    /// Steps of the level below per region.
    stride: usize,
    /// Steps every run holds but the last that holds any.
    run_length: usize,
    /// This instance's run, counted from the first.
    run: usize,
}

#[cube]
impl Portion {
    /// How many regions the portion touches.
    pub fn touched(&self) -> usize {
        self.regions.total()
    }

    /// The `i`-th region the portion touches.
    pub fn region(&self, i: usize) -> Region {
        self.regions.region(i)
    }

    /// The `i`-th region the portion touches, in the level's flat index: one for every region of
    /// the level, whichever instance touches it.
    pub fn index(&self, i: usize) -> usize {
        self.first + i
    }

    /// Start and length of this portion's steps within the `i`-th region's walk below.
    pub fn steps(&self, i: usize) -> (usize, usize) {
        let base = (self.first + i) * self.stride;
        let from = select(base < self.start, self.start - base, 0);
        let to = select(self.end < base + self.stride, self.end - base, self.stride);
        (from, to - from)
    }

    /// The first run holding steps of the `i`-th region the portion touches, and how many runs
    /// hold them. This run is one of them.
    pub fn holders(&self, i: usize) -> (usize, usize) {
        let base = (self.first + i) * self.stride;
        let first_run = self.run_holding(base);
        let last_run = self.run_holding(base + self.stride - 1);
        (first_run, last_run + 1 - first_run)
    }

    /// This instance's run, counted from the first.
    pub fn run(&self) -> usize {
        self.run
    }

    /// Whether `run`, one of the [`holders`](Portion::holders) of the `i`-th region the portion
    /// touches, starts there rather than in a region before it.
    pub fn starts_in(&self, i: usize, run: usize) -> bool {
        run * self.run_length >= (self.first + i) * self.stride
    }

    /// The run holding `step`.
    fn run_holding(&self, step: usize) -> usize {
        step / self.run_length
    }
}

#[cube]
impl Walk {
    /// This instance's portion of the level's flat index, counted in steps of `below`.
    pub fn portion(self, #[comptime] below: Level) -> Portion {
        let instances = comptime!(
            self.level
                .shared_by()
                .expect("Walk::portion: this level distributes no grid as one; say `shared_by`")
        );
        let stride = self.region(0).over(&below).total();
        let steps = self.total() * stride;
        let run = match comptime!(self.level.coverage()) {
            Coverage::Distribute(ComputeScope::Cube) => CubeAxis::position(CubeAxis::X),
            Coverage::Distribute(ComputeScope::Plane) => {
                ComputeScope::position(ComputeScope::Plane)
            }
            Coverage::Distribute(ComputeScope::PlaneGroup { planes }) => {
                ComputeScope::position(comptime!(ComputeScope::PlaneGroup { planes }))
            }
            Coverage::Distribute(ComputeScope::Unit) | Coverage::Walk => {
                panic!(
                    "Walk::portion: only cubes, plane groups or planes share a grid as one index"
                )
            }
        };
        let run_length = steps.div_ceil(instances);
        let start = (run * run_length).min(steps);
        let end = (start + run_length).min(steps);
        let first = start / stride;
        // `end` is exclusive; an empty portion touches nothing.
        let touched = select(start < end, (end - 1) / stride + 1 - first, 0);
        Portion {
            regions: self.range(first, touched),
            first,
            start,
            end,
            stride,
            run_length,
            run,
        }
    }
}
