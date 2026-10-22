//! The runtime half of a [`Projection`] over one buffer: its shape and strides, dotted in-kernel.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// In-kernel twin of cubecl's `TiledViewLayout` over a physical coordinate, `projection` being
/// [`Projection::positional`]'s synthetic map.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct BufferLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// Where a stage keeps each line of a block row.
    #[cube(comptime)]
    pub(crate) rows: RowArrangement,
    /// The table a pool of pages is read as sequences through ([`Paging`]); `None` for a buffer
    /// read where it lies.
    pub(crate) paging: ComptimeOption<PageTable>,
}

#[cube]
impl Layout for BufferLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = comptime!(self.projection.physical_rank());
        let mut digits = Coords::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            digits.push(self.physical_digit(&pos, pa));
        }
        let (digits, base) = self.paged(digits);
        let digits = self.placed(&digits);
        let mut terms = Sequence::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            terms.push(digits.at(pa).times(self.physical_strides.at(pa)));
        }
        terms
            .sum(comptime!((0..rank).collect::<Vec<_>>()))
            .plus(base)
            .cast::<usize>()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        logical_extent(comptime!(self.projection.clone()), &self.physical_shape).to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;

        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }

        valid
    }
}

#[cube]
impl BufferLayout {
    /// The logical coordinate line `i` of this buffer holds.
    pub(crate) fn line_coords(&self, i: usize) -> CoordsDyn {
        let digits = self.placed(&self.line_digits(i));
        fold_physical(
            comptime!(self.projection.clone()),
            &digits,
            &self.physical_shape,
        )
    }

    /// Where line `i` of this buffer sits: `i` itself unless rows are padded.
    pub(crate) fn line_offset(&self, i: usize) -> usize {
        if comptime!(self.rows.is_pitched()) {
            let digits = self.line_digits(i);
            let mut offset = 0u32;
            #[unroll]
            for pa in 0..digits.len() {
                offset = offset.plus(digits.at(pa).times(self.physical_strides.at(pa)));
            }
            offset.cast::<usize>()
        } else {
            i
        }
    }

    /// The logical coordinate of line `i` counted row-major over the logical extents: the
    /// source's order, whatever blocking or swizzle places the line here.
    pub(crate) fn row_major_coords(&self, i: usize) -> CoordsDyn {
        let rank = comptime!(self.projection.logical_rank());
        let extent = logical_extent(comptime!(self.projection.clone()), &self.physical_shape);
        let x = i.cast::<u32>();
        let mut coords = CoordsDyn::new();
        #[unroll]
        for p in 0..rank {
            let inner = comptime!((p + 1..rank).collect::<Vec<_>>());
            let quot = x.divided_by(extent.product(inner));
            let digit = if comptime!(p == 0) {
                quot
            } else {
                quot.remainder(extent.at(p))
            };
            coords.push(digit);
        }
        coords
    }

    /// The digits line `i` sits at, one per physical axis, row-major over its extents.
    fn line_digits(&self, i: usize) -> Coords<u32> {
        let x = i.cast::<u32>();
        let mut digits = Coords::<u32>::new();
        #[unroll]
        for pa in 0..self.physical_shape.len() {
            digits.push(line_digit(x, &self.physical_shape, pa));
        }
        digits
    }

    /// This layout for a window whose origin is `origin`, cut as `cut` says
    /// ([`PageTable::pinned_at`]): the pages the window reads resolved once where the buffer is a
    /// pool of pages, unchanged elsewhere.
    pub(crate) fn pinned_at(
        &self,
        origin: &Coords<i32>,
        #[comptime] cut: WindowCut,
    ) -> BufferLayout {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => {
                self.with_paging(ComptimeOption::new_Some(table.pinned_at(
                    origin,
                    cut,
                    self.physical_strides.at(comptime!(table.sequence_dim)),
                    self.physical_strides.at(comptime!(table.position_dim)),
                )))
            }
            ComptimeOption::None => self.clone(),
        }
    }

    /// The entry this layout's window was cut with, [`PageEntry::unlisted`] where the buffer is not
    /// a pool of pages.
    pub(crate) fn pinned_entry(&self) -> PageEntry {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => table.pinned_entry(),
            ComptimeOption::None => PageEntry::unlisted(),
        }
    }

    /// This layout taking its pages from `entry` where a cut names its cell; unchanged where the
    /// buffer is not a pool of pages.
    pub(crate) fn with_page_entry(&self, entry: &PageEntry) -> BufferLayout {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => {
                self.with_paging(ComptimeOption::new_Some(table.with_page_entry(entry)))
            }
            ComptimeOption::None => self.clone(),
        }
    }

    /// This layout over `paging`, the rest unchanged.
    fn with_paging(&self, paging: ComptimeOption<PageTable>) -> BufferLayout {
        BufferLayout {
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            projection: comptime!(self.projection.clone()),
            rows: comptime!(self.rows),
            paging,
        }
    }

    /// `digits` located through the page table where the buffer is a pool of pages
    /// ([`PageTable::located`]), with the base the line's offset adds past them.
    fn paged(&self, digits: Coords<u32>) -> (Coords<u32>, u32) {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => table.located(&digits),
            ComptimeOption::None => (digits, 0u32),
        }
    }

    /// `digits` with a swizzled stage's line digit `XORed` by its row's key.
    fn placed(&self, digits: &Coords<u32>) -> Coords<u32> {
        let mut placed = Coords::<u32>::new();
        #[unroll]
        for pa in 0..digits.len() {
            let mut digit = digits.at(pa);
            // The cube macro branches on a comptime value only through a `match`.
            #[allow(clippy::single_match)]
            match comptime!(self.rows.swizzle_along(pa)) {
                Some(swizzle) => {
                    digit = swizzled_line(swizzle, digit, digits.at(comptime!(swizzle.row_axis())));
                }
                None => {}
            }
            placed.push(digit);
        }
        placed
    }

    /// The digit `pos` sits at along physical axis `pa`.
    fn physical_digit(&self, pos: &CoordsDyn, #[comptime] pa: usize) -> u32 {
        let map = comptime!(self.projection.physical_axis(pa).clone());
        let picks = comptime!((0..map.terms().len()).collect::<Vec<_>>());
        let mut parts = Sequence::<u32>::new();
        #[unroll]
        for t in 0..comptime!(map.terms().len()) {
            let term = comptime!(map.terms()[t]);
            let p = comptime!(self.projection.position(term.axis));
            let (finer, modulo) = comptime!(self.projection.digit(pa, term.axis));
            // Strip the finer digits, then take this one; the outermost keeps the full quotient.
            let quot = pos[p].divided_by(self.physical_shape.product(comptime!(finer.to_vec())));
            let digit = match comptime!(modulo) {
                Some(m) => quot.remainder(self.physical_shape.at(m)),
                None => quot,
            };
            parts.push(digit.times(comptime!(term.scale.get() as u32).runtime()));
        }
        parts.sum(picks)
    }
}
