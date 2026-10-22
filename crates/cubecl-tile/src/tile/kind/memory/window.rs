//! Addressing the bytes: [`Window`] and [`SourceWindow`].

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;
use cubecl_std::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// The layout [`Tile::at`] applies: shift every axis to `origin` and crop it to `extent`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Window {
    pub(crate) origin: Coords<i32>,
    pub(crate) extent: Coords<u32>,
    /// Absolute logical extent; `is_in_bounds` clips against it, while `shape()` stays `extent`.
    pub(crate) bound: Coords<u32>,
    /// Whether the origin can be negative.
    #[cube(comptime)]
    pub(crate) signed: bool,
    /// Per-axis boundary handling; `None` (or an empty list) means in bounds by construction.
    #[cube(comptime)]
    pub(crate) boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

#[cube]
impl Window {
    pub(crate) fn new(
        origin: Coords<i32>,
        extent: Coords<u32>,
        bound: Coords<u32>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    ) -> Self {
        // `bound` is left out: a sub-window inherits its parent's, which may differ in rank.
        let origin_rank = origin.len();
        let extent_rank = extent.len();
        comptime!(assert!(
            extent_rank == origin_rank
                && (boundaries.is_empty() || boundaries.len() == origin_rank),
            "Window: origin ({origin_rank}), extent ({extent_rank}) and boundaries ({}) index the \
             same axes and must agree in rank",
            boundaries.len()
        ));
        Window {
            origin,
            extent,
            bound,
            signed,
            boundaries,
        }
    }

    /// Whether `pos` is valid on the selected physical axes.
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    pub(crate) fn axes_in_bounds(&self, pos: &CoordsDyn, #[comptime] axes: Vec<usize>) -> bool {
        let mut valid = true;
        #[unroll]
        for a in 0..comptime!(axes.len()) {
            let i = comptime!(axes[a]);
            if comptime!(self.boundaries.get(i).copied().flatten() == Some(Boundary::Zero)) {
                valid = valid && self.axis_in_bounds(pos[i], i);
            }
        }
        valid
    }

    /// The scalar physical-axis check behind [`axes_in_bounds`](Self::axes_in_bounds).
    pub(crate) fn axis_in_bounds(&self, pos: u32, #[comptime] axis: usize) -> bool {
        if comptime!(self.boundaries.get(axis).copied().flatten() == Some(Boundary::Zero)) {
            self.lands(pos, axis)
        } else {
            true.runtime()
        }
    }

    /// Whether `pos` lands inside the buffer on every axis a read does not clamp onto its edge.
    ///
    /// Every axis, not only those the window names a [`Boundary::Zero`] on: a window opened at a
    /// box larger than its buffer, as a fragment taller than the rows it is drained into, leaves
    /// the buffer along an axis that names none.
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    pub(crate) fn holds(&self, pos: &CoordsDyn) -> bool {
        let mut valid = true;
        #[unroll]
        for i in 0..self.origin.len() {
            if comptime!(self.boundaries.get(i).copied().flatten() != Some(Boundary::Clamp)) {
                valid = valid && self.lands(pos[i], i);
            }
        }
        valid
    }

    /// Whether `pos` along `axis` lands inside the buffer.
    fn lands(&self, pos: u32, #[comptime] axis: usize) -> bool {
        let abs = self.origin.at(axis).plus(pos.cast::<i32>());
        if comptime!(self.signed) {
            abs >= 0i32 && abs.cast::<u32>() < self.bound.at(axis)
        } else {
            abs.cast::<u32>() < self.bound.at(axis)
        }
    }

    /// This window under `guard`: [`Guard::Proved`] drops clamping and [`Boundary`] modes.
    pub(crate) fn with_guard(self, #[comptime] guard: Guard) -> Window {
        Window {
            origin: self.origin,
            extent: self.extent,
            bound: self.bound,
            signed: comptime!(guard.checks() && self.signed),
            boundaries: comptime!(match guard {
                Guard::Checked => self.boundaries.clone(),
                Guard::Proved => SmallVec::new(),
            }),
        }
    }

    /// This window addressing as a read that zeroes past the edge must: every axis clamps onto
    /// the edge cell, so the address stays in the buffer whatever the read keeps. Every axis, as
    /// [`holds`](Self::holds) checks every axis.
    #[allow(
        dead_code,
        reason = "called through its expansion, by `WindowedView`'s reads"
    )]
    pub(crate) fn edge_clamped(self) -> Window {
        let rank = self.origin.len();
        Window {
            origin: self.origin,
            extent: self.extent,
            bound: self.bound,
            signed: comptime!(self.signed),
            boundaries: comptime!(SmallVec::from_elem(Some(Boundary::Clamp), rank)),
        }
    }
}

/// Where a gathered stage sits inside the buffer it was filled from; invariant under
/// [`at`](Memory::at).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct SourceWindow {
    /// The source window's origin.
    pub(crate) origin: Coords<i32>,
    /// The source buffer's logical extent.
    pub(crate) bound: Coords<u32>,
    /// Per physical axis, what a stage coordinate is multiplied by to land on the source.
    #[cube(comptime)]
    pub(crate) steps: SmallVec<[usize; Space::MAX_RANK]>,
    /// Whether the source origin can be negative.
    #[cube(comptime)]
    pub(crate) signed: bool,
    /// The source's per-axis boundary handling.
    #[cube(comptime)]
    pub(crate) boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

#[cube]
impl SourceWindow {
    /// Whether `pos` of the staged window at `stage_origin` lands inside the source on `axes`.
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    pub(crate) fn axes_in_bounds(
        &self,
        stage_origin: &Coords<i32>,
        pos: &CoordsDyn,
        #[comptime] axes: Vec<usize>,
    ) -> bool {
        let mut valid = true;
        #[unroll]
        for a in 0..comptime!(axes.len()) {
            let i = comptime!(axes[a]);
            if comptime!(self.boundaries.get(i).copied().flatten() == Some(Boundary::Zero)) {
                valid = valid && self.axis_in_bounds(stage_origin.at(i), pos[i], i);
            }
        }
        valid
    }

    /// Whether `stage_origin + pos` on `axis` lands inside the source.
    pub(crate) fn axis_in_bounds(
        &self,
        stage_origin: i32,
        pos: u32,
        #[comptime] axis: usize,
    ) -> bool {
        if comptime!(self.boundaries.get(axis).copied().flatten() == Some(Boundary::Zero)) {
            let step = comptime!(self.steps.get(axis).copied().unwrap_or(1) as i32);
            let cell = (stage_origin + pos.cast::<i32>()) * step;
            let abs = self.origin.at(axis) + cell;
            if comptime!(self.signed) {
                abs >= 0i32 && abs.cast::<u32>() < self.bound.at(axis)
            } else {
                abs.cast::<u32>() < self.bound.at(axis)
            }
        } else {
            true.runtime()
        }
    }
}

#[cube]
impl Layout for Window {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for i in 0..self.origin.len() {
            let abs = self.origin.at(i).plus(pos[i].cast::<i32>());
            // Branchless: this runs per tap of every gathered read.
            let shifted = if comptime!(self.signed) {
                select(abs >= 0i32, abs.cast::<u32>(), 0u32)
            } else {
                abs.cast::<u32>()
            };
            let shifted = match comptime!(self.boundaries.get(i).copied().flatten()) {
                Some(Boundary::Clamp) => {
                    let bound_i = self.bound.at(i);
                    let edge = select(shifted >= bound_i, bound_i.minus(1u32), shifted);
                    // Both arms evaluate: a zero-extent axis's wrapped `bound - 1` is discarded.
                    select(bound_i == 0u32, 0u32, edge)
                }
                None | Some(Boundary::Zero) => shifted,
            };
            out.push(shifted);
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.holds(&pos)
    }
}
